// Joiner-side worker actor that consumes the inbound control stream
// from one coordinator and drives the rank-local model in lock step
// with the coordinator's KV cache.
//
// Symmetric counterpart to `CoordinatorInferenceSession`. Where the
// coordinator owns request lifecycle, sampling, and stream emission,
// the joiner is a passive dispatch loop:
//
//   - Receive an `InferenceControlMessage` from the inbound stream.
//   - Dispatch synchronously inside the actor to a per-case handler
//     that mutates per-session state and / or invokes the rank-local
//     forward closure.
//   - Repeat until the channel's inbound stream terminates or the
//     surrounding `Task` is cancelled.
//
// The forward pass produces logits the joiner discards. What matters
// is that the forward call participates in the model's `allSum`
// collectives via `DistributedGroup`; that is what advances the
// rank-local KV cache and contributes the joiner's shard of the
// session compute. Sampling lives only on the coordinator.
//
// Closure injection. The actor takes a `forward(tokenIds, cache)`
// function and a `makeCache()` function rather than a direct reference
// to `DistributedQwenModel`. This mirrors the coordinator's design
// and keeps the worker loop unit-testable without standing up a real
// distributed model. Production callers wrap the real model:
//
//     let worker = JoinerInferenceWorker(
//         channel: controlChannel,
//         forward: { ids, cache in model(ids, cache: cache) },
//         makeCache: { model.makeCache() })
//     await worker.run()
//
// Ordering invariant. The worker assumes the coordinator broadcasts
// the canonical sequence:
//
//     sessionStart, prefill, injectToken(t1), decode,
//     injectToken(t2), decode, ..., sessionEnd
//
// At each `decode`, the joiner forwards the most-recently-injected
// token through the model so its KV cache appends the same step the
// coordinator's cache appended on the matching round.

import Foundation
import MLX
import MLXLMCommon

/// One distributed inference joiner. Construct one instance per
/// coordinator-joiner pairing; the actor owns per-session state across
/// successive sessions over the same control channel and must remain
/// alive between sessions so the next `sessionStart` allocates a fresh
/// cache without tearing down the worker.
public actor JoinerInferenceWorker {

    // MARK: - Closure-injected model boundary

    /// Forward-pass closure. Mirrors
    /// `DistributedQwenModel.callAsFunction(_:cache:)`: takes a
    /// `[batch, length]` integer-typed token-id tensor plus the
    /// per-block KV cache list and returns `[batch, length, vocabSize]`
    /// logits. The joiner discards the returned logits but the call
    /// participates in `allSum` collectives that drive the rank-local
    /// cache forward.
    public typealias ForwardFn = @Sendable (MLXArray, [KVCache]) -> MLXArray

    /// Cache-allocation closure. Mirrors
    /// `DistributedQwenModel.makeCache()`: produces a fresh per-block
    /// KV cache array sized to the model's layer count.
    public typealias MakeCacheFn = @Sendable () -> [KVCache]

    // MARK: - Stored state

    private let channel: any InferenceControlChannel
    private let forward: ForwardFn
    private let makeCache: MakeCacheFn

    // MARK: - Per-session state

    /// Identifier of the in-flight session. `nil` when no session is
    /// active. Stored for symmetry with the coordinator and to support
    /// a future drain / resume protocol that needs to identify which
    /// session a recovery message references.
    private var sessionID: UUID?

    /// Sampling parameters carried in `sessionStart`. Joiners do not
    /// sample in v1, but the parameters are retained so a future
    /// speculative-decode side channel that needs joiner-side branching
    /// on the sampling policy has them available without re-fetching.
    private var samplingParams: SamplingParams?

    /// Deterministic seed carried in `sessionStart`. Reserved for
    /// future joiner-side stochastic state; unused in v1.
    private var seed: UInt64?

    /// Per-block KV cache for the active session. Empty between
    /// sessions; reallocated on every `sessionStart` so no state from
    /// the previous session leaks into the next.
    private var cache: [KVCache] = []

    /// Token-id to feed at the next `.decode` step. Populated by
    /// `injectToken`, consumed by `decode`. `nil` between sessions and
    /// after each `decode` round consumes it; the next coordinator
    /// `injectToken` repopulates it before the following `decode`.
    /// Tracking this explicitly catches a coordinator-bug regression
    /// where a `decode` arrives without a preceding `injectToken`.
    private var pendingToken: Int?

    // MARK: - Init

    /// Construct a worker. The channel plus closures are the only
    /// collaborators; the actor owns no model reference and no further
    /// state until `run` consumes its first inbound message.
    public init(
        channel: any InferenceControlChannel,
        forward: @escaping ForwardFn,
        makeCache: @escaping MakeCacheFn
    ) {
        self.channel = channel
        self.forward = forward
        self.makeCache = makeCache
    }

    // MARK: - Worker loop

    /// Consume the channel's inbound stream until it terminates or the
    /// surrounding task is cancelled. Each message is dispatched
    /// synchronously inside the actor's isolation domain; messages do
    /// not overlap. On stream termination, any in-flight session state
    /// is released so a torn-down channel does not leave a dangling
    /// cache attached to the worker.
    public func run() async {
        for await message in channel.inbound {
            handle(message)
        }
        // Inbound stream closed. If a session was still in flight,
        // release its state so the worker can be reused over a fresh
        // channel without carrying stale per-session bookkeeping.
        releaseSessionState()
    }

    // MARK: - Dispatch

    /// Per-case handler. Exhaustive switch keeps the routing close to
    /// the message-shape definition and lets the compiler enforce
    /// coverage when new cases are introduced.
    private func handle(_ message: InferenceControlMessage) {
        switch message {
        case .sessionStart(let sessionID, let sampling, let seed):
            startSession(sessionID: sessionID, sampling: sampling, seed: seed)
        case .prefill(let promptTokens):
            handlePrefill(promptTokens: promptTokens)
        case .decode:
            handleDecode()
        case .injectToken(let sampledToken):
            handleInjectToken(sampledToken: sampledToken)
        case .sessionEnd:
            releaseSessionState()
        }
    }

    // MARK: - Per-case handlers

    /// Allocate fresh per-session state. Any leftover cache or pending
    /// token from a previous session is dropped first so a malformed
    /// sequence (sessionStart without a preceding sessionEnd) cannot
    /// silently leak state into the new session.
    private func startSession(
        sessionID: UUID,
        sampling: SamplingParams,
        seed: UInt64
    ) {
        self.sessionID = sessionID
        self.samplingParams = sampling
        self.seed = seed
        self.cache = makeCache()
        self.pendingToken = nil
    }

    /// Run the prompt through the rank-local forward pass with the
    /// session's cache. Joiners do not sample, so the returned logits
    /// are discarded; the side effect of interest is the cache update
    /// and the participation in `allSum` collectives that the forward
    /// pass triggers internally.
    private func handlePrefill(promptTokens: [Int]) {
        let length = promptTokens.count
        // Defensive: the coordinator side guarantees a non-empty
        // prompt, but the joiner cannot assume malformed traffic is
        // impossible. Skipping the forward pass on an empty prompt
        // avoids producing a zero-length token tensor that would
        // crash the reshape arithmetic.
        guard length > 0 else { return }
        let promptIds = MLXArray(promptTokens.map { Int32($0) })
            .reshaped(1, length)
        _ = forward(promptIds, cache)
    }

    /// Run a single decode-step forward against the most-recently-
    /// injected token. The pending token is cleared on consumption so
    /// a missing `injectToken` between two `decode` messages is
    /// observable as `pendingToken == nil` rather than silently
    /// re-feeding the previous step.
    private func handleDecode() {
        guard let token = pendingToken else {
            // Missing injectToken between sessionStart/prefill and
            // this decode, or between two decodes. The coordinator's
            // contract guarantees this does not happen; we surface
            // the divergence by simply skipping the forward rather
            // than feeding stale or zero data.
            return
        }
        let stepIds = MLXArray([Int32(token)]).reshaped(1, 1)
        _ = forward(stepIds, cache)
        pendingToken = nil
    }

    /// Store the coordinator's sampled token for the upcoming decode
    /// step. Overwrites any previously stored pending token; the
    /// coordinator's broadcast sequence interleaves one `injectToken`
    /// per `decode`, so a second `injectToken` without an intervening
    /// `decode` would itself be a contract violation handled by
    /// preferring the most recent value.
    private func handleInjectToken(sampledToken: Int) {
        pendingToken = sampledToken
    }

    /// Release everything tied to the active session. Called from
    /// `sessionEnd` and from the run-loop teardown when the channel's
    /// inbound stream terminates while a session is in flight.
    private func releaseSessionState() {
        sessionID = nil
        samplingParams = nil
        seed = nil
        cache = []
        pendingToken = nil
    }
}
