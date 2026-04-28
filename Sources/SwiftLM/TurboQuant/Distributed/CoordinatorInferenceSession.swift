// Coordinator-side actor that drives one user-visible inference
// request from prompt tokens through to a finished completion.
//
// Responsibilities:
//
//   - Allocate a fresh KV cache for the session.
//   - Broadcast the session-lifecycle control sequence to every
//     joiner (`sessionStart`, `prefill`, alternating
//     `decode` / `injectToken`, `sessionEnd`) so the joiners' KV
//     caches advance in lock step with the coordinator's.
//   - Run the local prefill and decode forward passes against the
//     supplied closure-injected model, sample on the coordinator
//     only, and yield each generated token to the streaming
//     continuation owned by the HTTP layer.
//   - End the session cleanly on natural stop, max-token termination,
//     and on cancellation. Cancellation must not leave joiners
//     stranded mid-session.
//
// Closure injection. The actor takes a `forward(tokenIds, cache)`
// function and a `makeCache()` function rather than a direct
// reference to `DistributedQwenModel`. This keeps the session loop
// unit-testable without standing up a real model instance and lets
// the runtime layer wrap any forward-pass-shaped object (the real
// distributed model, a single-rank oracle, or a stub) without
// changing this file. Production callers wrap the real model:
//
//     let session = CoordinatorInferenceSession(
//         forward: { ids, cache in model(ids, cache: cache) },
//         makeCache: { model.makeCache() },
//         joiners: channels)
//
// Broadcast semantics. The broadcast helper sends sequentially to
// every joiner. Sequential is intentional for v1 simplicity; the
// per-token control payload is small and the cluster sizes targeted
// in Phase 3 are modest. Parallel broadcast is a profile-driven
// optimisation later.
//
// Sampling. Sampling lives behind the `DistributedSampler` protocol.
// v1 ships only `GreedySampler`; stochastic samplers slot in without
// touching this actor.

import Foundation
import MLX
import MLXLMCommon

/// One distributed inference request, end to end. Construct one
/// instance per request; the actor owns the KV cache for the session
/// and is not designed to be reused across requests.
public actor CoordinatorInferenceSession {

    // MARK: - Closure-injected model boundary

    /// Forward-pass closure. Mirrors
    /// `DistributedQwenModel.callAsFunction(_:cache:)`: takes a
    /// `[batch, length]` integer-typed token-id tensor plus the
    /// per-block KV cache list and returns `[batch, length, vocabSize]`
    /// logits. Tests inject a stub closure that produces predictable
    /// logits without loading the real fixture.
    public typealias ForwardFn = (MLXArray, [KVCache]) -> MLXArray

    /// Cache-allocation closure. Mirrors
    /// `DistributedQwenModel.makeCache()`: produces a fresh
    /// per-block KV cache array sized to the model's layer count.
    public typealias MakeCacheFn = () -> [KVCache]

    // MARK: - Public types

    /// One inference request. The actor consumes this through `serve`
    /// and never mutates it; all session state lives in the actor.
    public struct InferenceRequest: Sendable {
        public let promptTokens: [Int]
        public let sampling: SamplingParams
        public let seed: UInt64

        public init(promptTokens: [Int], sampling: SamplingParams, seed: UInt64) {
            self.promptTokens = promptTokens
            self.sampling = sampling
            self.seed = seed
        }
    }

    /// One token-event yielded to the streaming continuation. The
    /// HTTP layer maps these onto SSE frames.
    public enum TokenEvent: Sendable {
        case delta(token: Int)
        case finish(reason: FinishReason)
    }

    /// Why a session ended. `error(_:)` carries a human-readable
    /// description so the HTTP layer can surface it without a typed
    /// error hierarchy at this layer.
    public enum FinishReason: Sendable {
        case stopToken
        case maxTokens
        case cancelled
        case error(String)
    }

    // MARK: - Stored state

    private let forward: ForwardFn
    private let makeCache: MakeCacheFn
    private let joiners: [InferenceControlChannel]
    private let sampler: DistributedSampler

    // MARK: - Init

    /// Construct a session. The closures plus channel list are the
    /// only collaborators; the actor owns no other state until
    /// `serve` is invoked.
    public init(
        forward: @escaping ForwardFn,
        makeCache: @escaping MakeCacheFn,
        joiners: [InferenceControlChannel],
        sampler: DistributedSampler = GreedySampler()
    ) {
        self.forward = forward
        self.makeCache = makeCache
        self.joiners = joiners
        self.sampler = sampler
    }

    // MARK: - Session loop

    /// Run one request to completion, yielding token deltas to
    /// `stream`. The continuation is `finish()`-ed before this method
    /// returns under all exit paths (natural stop, max-token, error,
    /// cancellation) so the caller's `for await` loop on the matching
    /// `AsyncStream` always terminates.
    ///
    /// Cancellation. If the surrounding `Task` is cancelled mid-decode,
    /// the actor catches `CancellationError`, broadcasts `sessionEnd`
    /// so joiners release per-session state, yields a `.cancelled`
    /// finish event, and rethrows.
    public func serve(
        _ req: InferenceRequest,
        stream: AsyncStream<TokenEvent>.Continuation
    ) async throws {
        // Track whether sessionEnd has been broadcast so the cleanup
        // path does not double-broadcast on the natural-exit branch
        // and so the cancellation / error branches do not skip it.
        var endBroadcast = false

        do {
            let cache = makeCache()
            let sessionID = UUID()

            try await broadcast(.sessionStart(
                sessionID: sessionID,
                sampling: req.sampling,
                seed: req.seed))
            try await broadcast(.prefill(promptTokens: req.promptTokens))

            // Local prefill. Token-id tensor is `[1, promptCount]`;
            // forward returns `[1, promptCount, vocabSize]`. We sample
            // from the final position only — earlier positions are
            // irrelevant for next-token generation.
            try Task.checkCancellation()
            let promptCount = req.promptTokens.count
            precondition(promptCount > 0,
                         "InferenceRequest.promptTokens must be non-empty")
            let promptIds = MLXArray(req.promptTokens.map { Int32($0) })
                .reshaped(1, promptCount)
            let prefillLogits = forward(promptIds, cache)

            // Slice `[1, L, V]` -> `[V]` for the final position.
            let firstLogits = lastPositionLogits(prefillLogits)
            var nextToken = sampler.sample(logits: firstLogits, sampling: req.sampling)

            try await broadcast(.injectToken(sampledToken: nextToken))
            stream.yield(.delta(token: nextToken))
            var emitted = 1

            // Natural-stop path: the freshly-sampled prefill token
            // hits a stop-token before any decode round runs.
            if req.sampling.stopTokens.contains(nextToken) {
                try await broadcast(.sessionEnd)
                endBroadcast = true
                stream.yield(.finish(reason: .stopToken))
                stream.finish()
                return
            }
            if emitted >= req.sampling.maxTokens {
                try await broadcast(.sessionEnd)
                endBroadcast = true
                stream.yield(.finish(reason: .maxTokens))
                stream.finish()
                return
            }

            // Decode loop. Each iteration: broadcast `decode`, run
            // one local decode-step forward, sample the next token,
            // broadcast `injectToken`, yield to the stream, and
            // re-evaluate stop conditions.
            while true {
                try Task.checkCancellation()
                try await broadcast(.decode)

                let stepIds = MLXArray([Int32(nextToken)]).reshaped(1, 1)
                let stepLogits = forward(stepIds, cache)
                let stepFinal = lastPositionLogits(stepLogits)
                nextToken = sampler.sample(logits: stepFinal, sampling: req.sampling)

                try await broadcast(.injectToken(sampledToken: nextToken))
                stream.yield(.delta(token: nextToken))
                emitted += 1

                if req.sampling.stopTokens.contains(nextToken) {
                    try await broadcast(.sessionEnd)
                    endBroadcast = true
                    stream.yield(.finish(reason: .stopToken))
                    stream.finish()
                    return
                }
                if emitted >= req.sampling.maxTokens {
                    try await broadcast(.sessionEnd)
                    endBroadcast = true
                    stream.yield(.finish(reason: .maxTokens))
                    stream.finish()
                    return
                }
            }
        } catch is CancellationError {
            // Ensure joiners are not left mid-session. Best-effort:
            // the broadcast itself may fail if the transport has
            // already torn down, in which case we surface the
            // cancellation regardless.
            if !endBroadcast {
                try? await broadcast(.sessionEnd)
            }
            stream.yield(.finish(reason: .cancelled))
            stream.finish()
            throw CancellationError()
        } catch {
            if !endBroadcast {
                try? await broadcast(.sessionEnd)
            }
            stream.yield(.finish(reason: .error(String(describing: error))))
            stream.finish()
            throw error
        }
    }

    // MARK: - Helpers

    /// Broadcast `message` to every joiner in registration order.
    /// Sequential by design; see file-level note on parallel
    /// broadcast as a deferred optimisation.
    private func broadcast(_ message: InferenceControlMessage) async throws {
        for joiner in joiners {
            try await joiner.send(message)
        }
    }

    /// Slice the final position out of a forward-pass output.
    /// Forward returns `[batch, length, vocabSize]`; the sampler
    /// expects a 1-D `[vocabSize]` row from batch=0, position=L-1.
    private func lastPositionLogits(_ logits: MLXArray) -> MLXArray {
        precondition(logits.ndim == 3,
                     "forward output must be [batch, length, vocabSize]; " +
                     "got shape \(logits.shape)")
        let length = logits.dim(1)
        precondition(length > 0,
                     "forward output length axis must be non-zero; " +
                     "got shape \(logits.shape)")
        // `[B, L, V][:, L-1, :]` -> `[B, V]`, then squeeze batch axis.
        return logits[0, length - 1]
    }
}
