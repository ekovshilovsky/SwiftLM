// Behavioural tests for `CoordinatorInferenceSession`. These tests
// exercise the actor's broadcast sequence, stop-token handling, and
// max-token handling against a closure-injected stub forward function
// — no real model fixture is loaded. The stub returns predictable
// logits whose argmax is controllable from the test, so the
// generated-token sequence is fully deterministic.
//
// Each test asserts both the streamed `TokenEvent` sequence and the
// recorded broadcast log on every joiner channel. The two views
// together prove that the actor maintains the wire-protocol
// invariant: every emitted token is preceded by an `injectToken`
// broadcast, every decode step is preceded by a `decode` broadcast,
// and every session terminates with `sessionEnd`.

import XCTest
import Foundation
import MLX
import MLXLMCommon
@testable import TurboQuantKit

final class CoordinatorInferenceSessionTests: XCTestCase {

    // MARK: - Fakes

    /// Test sampler that returns a caller-provided token id on every
    /// call, ignoring the logits and sampling parameters. Lets a test
    /// drive a deterministic generated-token sequence without having
    /// to construct logits whose argmax is precisely the desired
    /// token. The next-token closure receives the zero-based call
    /// index so a test can vary the result across decode rounds.
    private struct ScriptedSampler: DistributedSampler {
        // Class-backed counter so the struct stays Sendable while the
        // mutable state survives across `sample` calls invoked from
        // the actor's isolation domain.
        final class Counter: @unchecked Sendable {
            private let lock = NSLock()
            private var value: Int = 0
            func next() -> Int {
                lock.lock()
                defer { lock.unlock() }
                let current = value
                value += 1
                return current
            }
        }
        let counter = Counter()
        let nextToken: @Sendable (Int) -> Int

        func sample(logits: MLXArray, sampling: SamplingParams) -> Int {
            return nextToken(counter.next())
        }
    }

    // MARK: - Helpers

    /// Build a stub forward closure that returns a `[1, length, 1]`
    /// logits tensor of zeros. The actual sampled token is decided by
    /// `ScriptedSampler`; the stub only needs to produce a tensor of
    /// the right rank so the actor's reshape / slice arithmetic
    /// succeeds. A 1-element vocab keeps the allocation trivial.
    private func makeStubForward()
        -> (MLXArray, [KVCache]) -> MLXArray
    {
        return { tokenIds, _ in
            let length = tokenIds.dim(1)
            // Tensor shape `[1, length, 1]` with a single dummy logit
            // per position. The session actor only inspects the final
            // position, and the scripted sampler ignores the value
            // entirely.
            return MLXArray.zeros([1, length, 1], dtype: .float32)
        }
    }

    /// Allocate a per-block KV cache list for the stub model. The
    /// stub forward never consults the cache, but `serve` calls
    /// `makeCache` once at session start and the cache must outlive
    /// the call so an empty array is returned. Returning an empty
    /// list is intentional — the stub forward never indexes into it.
    private func makeStubCache() -> [KVCache] {
        return []
    }

    /// Drain every event the session yielded into a flat array.
    /// Bridges the AsyncStream pattern back to a synchronous
    /// assertion-friendly shape.
    private func collect(
        _ stream: AsyncStream<CoordinatorInferenceSession.TokenEvent>
    ) async -> [CoordinatorInferenceSession.TokenEvent] {
        var events: [CoordinatorInferenceSession.TokenEvent] = []
        for await event in stream {
            events.append(event)
        }
        return events
    }

    // MARK: - Test 1: broadcast sequence

    /// Verify the broadcast sequence on a max-tokens-bounded run with
    /// no stop-token match. The actor must emit, in order:
    ///
    ///   1. `sessionStart`
    ///   2. `prefill(promptTokens)`
    ///   3. `injectToken(t1)`              — token sampled from prefill
    ///   4. `decode`, `injectToken(t2)`    — first decode round
    ///   5. `decode`, `injectToken(t3)`    — second decode round
    ///   6. `sessionEnd`
    ///
    /// Three tokens land in the stream; three injectToken broadcasts
    /// land on the joiner. `decode` appears exactly twice — once per
    /// decode round following the prefill-derived first token.
    func testBroadcastSequenceMaxTokensBounded() async throws {
        let joiner = InMemoryInferenceControlChannel()
        let sampler = ScriptedSampler(nextToken: { _ in 42 })

        let session = CoordinatorInferenceSession(
            forward: makeStubForward(),
            makeCache: { self.makeStubCache() },
            joiners: [joiner],
            sampler: sampler
        )

        let (stream, continuation) = AsyncStream
            .makeStream(of: CoordinatorInferenceSession.TokenEvent.self)
        let request = CoordinatorInferenceSession.InferenceRequest(
            promptTokens: [10, 20, 30],
            sampling: SamplingParams(temperature: 0, maxTokens: 3),
            seed: 7
        )

        try await session.serve(request, stream: continuation)
        let events = await collect(stream)

        // Stream contract: three deltas + one `.finish(.maxTokens)`.
        XCTAssertEqual(events.count, 4,
                       "expected 3 deltas + 1 finish; got \(events.count) events")
        var deltas: [Int] = []
        var finishReason: CoordinatorInferenceSession.FinishReason?
        for event in events {
            switch event {
            case .delta(let token): deltas.append(token)
            case .finish(let reason): finishReason = reason
            }
        }
        XCTAssertEqual(deltas, [42, 42, 42],
                       "scripted sampler must produce 42 every step")
        guard case .maxTokens = finishReason else {
            return XCTFail("expected .maxTokens finish; got \(String(describing: finishReason))")
        }

        // Wire-protocol contract: exact broadcast sequence.
        let sent = joiner.sentMessages
        let expectedShape: [String] = [
            "sessionStart",
            "prefill",
            "injectToken(42)",
            "decode",
            "injectToken(42)",
            "decode",
            "injectToken(42)",
            "sessionEnd"
        ]
        let actualShape = sent.map { describe($0) }
        XCTAssertEqual(actualShape, expectedShape,
                       "broadcast sequence diverged from spec")
    }

    // MARK: - Test 2: stop-token termination

    /// When the sampler returns a stop-token id, the session ends as
    /// soon as that token is broadcast and yielded. Verify the
    /// session terminates after exactly two emitted tokens when the
    /// second sample is the stop token, and the finish reason is
    /// `.stopToken` rather than `.maxTokens` even though the
    /// `maxTokens` budget allows further generation.
    func testStopTokenEndsSessionImmediately() async throws {
        let joiner = InMemoryInferenceControlChannel()
        // First sample (prefill) -> 7. Second sample (decode 1) -> 42.
        let sampler = ScriptedSampler(nextToken: { idx in idx == 0 ? 7 : 42 })

        let session = CoordinatorInferenceSession(
            forward: makeStubForward(),
            makeCache: { self.makeStubCache() },
            joiners: [joiner],
            sampler: sampler
        )

        let (stream, continuation) = AsyncStream
            .makeStream(of: CoordinatorInferenceSession.TokenEvent.self)
        let request = CoordinatorInferenceSession.InferenceRequest(
            promptTokens: [1, 2],
            // Generous max so the stop-token path is the only thing
            // that can terminate the session within this run.
            sampling: SamplingParams(temperature: 0, maxTokens: 100, stopTokens: [42]),
            seed: 0
        )

        try await session.serve(request, stream: continuation)
        let events = await collect(stream)

        var deltas: [Int] = []
        var finishReason: CoordinatorInferenceSession.FinishReason?
        for event in events {
            switch event {
            case .delta(let token): deltas.append(token)
            case .finish(let reason): finishReason = reason
            }
        }
        XCTAssertEqual(deltas, [7, 42],
                       "expected first prefill-sampled token then the stop token")
        guard case .stopToken = finishReason else {
            return XCTFail("expected .stopToken finish; got \(String(describing: finishReason))")
        }

        // Only one decode round occurs before the stop fires, so the
        // recorded sequence is: start, prefill, inject(7),
        // decode, inject(42), end.
        let actualShape = joiner.sentMessages.map { describe($0) }
        XCTAssertEqual(actualShape, [
            "sessionStart",
            "prefill",
            "injectToken(7)",
            "decode",
            "injectToken(42)",
            "sessionEnd"
        ])
    }

    // MARK: - Test 3: max-tokens termination

    /// `maxTokens = 3` with a stop-free token stream should produce
    /// exactly three deltas and a `.maxTokens` finish. Cross-checks
    /// the same loop the broadcast-sequence test exercises but with
    /// a non-trivial stop-token list (whose entries never appear)
    /// to verify the stop-check does not over-fire on unrelated
    /// values.
    func testMaxTokensTerminates() async throws {
        let joiner = InMemoryInferenceControlChannel()
        let sampler = ScriptedSampler(nextToken: { _ in 9 })

        let session = CoordinatorInferenceSession(
            forward: makeStubForward(),
            makeCache: { self.makeStubCache() },
            joiners: [joiner],
            sampler: sampler
        )

        let (stream, continuation) = AsyncStream
            .makeStream(of: CoordinatorInferenceSession.TokenEvent.self)
        let request = CoordinatorInferenceSession.InferenceRequest(
            promptTokens: [100, 200],
            sampling: SamplingParams(
                temperature: 0,
                maxTokens: 3,
                stopTokens: [13, 999]),
            seed: 42
        )

        try await session.serve(request, stream: continuation)
        let events = await collect(stream)

        var deltas: [Int] = []
        var finishReason: CoordinatorInferenceSession.FinishReason?
        for event in events {
            switch event {
            case .delta(let token): deltas.append(token)
            case .finish(let reason): finishReason = reason
            }
        }
        XCTAssertEqual(deltas, [9, 9, 9])
        guard case .maxTokens = finishReason else {
            return XCTFail("expected .maxTokens finish; got \(String(describing: finishReason))")
        }
    }

    // MARK: - Test 4: multi-joiner broadcast

    /// Every joiner channel must receive the identical broadcast
    /// sequence. Verifies the broadcast helper iterates over all
    /// channels — a regression where the helper sent only to the
    /// first channel would leave subsequent channels' KV caches out
    /// of sync with the coordinator's.
    func testBroadcastReachesEveryJoiner() async throws {
        let joinerA = InMemoryInferenceControlChannel()
        let joinerB = InMemoryInferenceControlChannel()
        let sampler = ScriptedSampler(nextToken: { _ in 5 })

        let session = CoordinatorInferenceSession(
            forward: makeStubForward(),
            makeCache: { self.makeStubCache() },
            joiners: [joinerA, joinerB],
            sampler: sampler
        )

        let (stream, continuation) = AsyncStream
            .makeStream(of: CoordinatorInferenceSession.TokenEvent.self)
        let request = CoordinatorInferenceSession.InferenceRequest(
            promptTokens: [1],
            sampling: SamplingParams(temperature: 0, maxTokens: 2),
            seed: 0
        )

        try await session.serve(request, stream: continuation)
        _ = await collect(stream)

        XCTAssertEqual(joinerA.sentMessages.map { describe($0) },
                       joinerB.sentMessages.map { describe($0) },
                       "every joiner must observe an identical broadcast log")
        XCTAssertFalse(joinerA.sentMessages.isEmpty,
                       "broadcast sequence must not be empty")
    }

    // MARK: - Helpers: human-readable broadcast description

    /// Compact textual rendering of a control message for assertion
    /// diffs. The exact `sessionStart` and `prefill` payload values
    /// are not part of the contract these tests cover (they are
    /// verified by `InferenceControlProtocolTests`); collapsing them
    /// to their case name keeps the broadcast-shape comparison
    /// focused on ordering.
    private func describe(_ message: InferenceControlMessage) -> String {
        switch message {
        case .sessionStart: return "sessionStart"
        case .prefill: return "prefill"
        case .decode: return "decode"
        case .injectToken(let token): return "injectToken(\(token))"
        case .sessionEnd: return "sessionEnd"
        }
    }
}
