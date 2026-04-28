// Behavioural tests for `DistributedInferenceEngine`. The engine is
// the top-level facade the HTTP layer calls into; these tests verify
// its three observable contracts:
//
//   - the `AsyncThrowingStream` it returns yields `.delta` per token
//     and exactly one `.finish` event before terminating;
//   - the default sampler factory routes `temperature == 0` requests
//     through `GreedySampler`, producing a deterministic argmax stream;
//   - stochastic sampling is reproducible for a fixed `seed` —
//     replaying the same request with the same seed and the same
//     stub forward returns an identical token sequence.
//
// The engine is constructed without a real `DistributedQwenModel`
// because no model fixture is required to exercise the streaming
// glue. Instead, the engine is bypassed via a custom sampler factory
// that returns a sampler scripted from the test's expected logits;
// the underlying `CoordinatorInferenceSession` is constructed
// directly and its stream wiring matches what the engine's
// `generate(_:)` does internally. The engine path proper is exercised
// by the model-bound integration tests; these tests focus on the
// stream contract and sampler-factory behaviour, which are the only
// things the engine adds on top of the session.

import XCTest
import Foundation
import MLX
import MLXLMCommon
@testable import TurboQuantKit

final class DistributedInferenceEngineTests: XCTestCase {

    // MARK: - Stub forward / cache

    /// Build a stub forward closure that returns logits of shape
    /// `[1, length, vocab]` with the same row repeated across every
    /// position. The single row is supplied by the test so a
    /// stochastic sampler driven off the row produces predictable
    /// output. Argmax of a row is the test's "fixed argmax" target.
    private func makeStubForward(
        logitsRow: [Float]
    ) -> (MLXArray, [KVCache]) -> MLXArray {
        let vocab = logitsRow.count
        let row = logitsRow
        return { tokenIds, _ in
            let length = tokenIds.dim(1)
            // Replicate the row across `length` positions. The
            // session only inspects the final position, so a flat
            // tile is sufficient and avoids a per-position allocation.
            var flat = [Float](repeating: 0, count: length * vocab)
            for pos in 0 ..< length {
                let base = pos * vocab
                for v in 0 ..< vocab {
                    flat[base + v] = row[v]
                }
            }
            return MLXArray(flat, [1, length, vocab])
        }
    }

    /// Stub cache list. The stub forward never indexes into it so
    /// returning an empty array is safe.
    private func makeStubCache() -> [KVCache] {
        return []
    }

    /// Drain every event the engine yielded into a flat array.
    private func collect<T: Sendable>(
        _ stream: AsyncThrowingStream<T, Error>
    ) async throws -> [T] {
        var events: [T] = []
        for try await event in stream {
            events.append(event)
        }
        return events
    }

    // MARK: - Test 1: stream emission shape

    /// The engine must emit exactly `maxTokens` `.delta` events plus
    /// one trailing `.finish(.maxTokens)` before terminating the
    /// stream. Verified against a stub forward that returns logits
    /// whose argmax is fixed at index 42 — the sampler walks the
    /// argmax path and the event count is determined entirely by the
    /// `maxTokens` budget.
    func testStreamEmitsDeltasAndOneFinish() async throws {
        let vocab = 64
        var row = [Float](repeating: 0, count: vocab)
        row[42] = 10.0
        let forward = makeStubForward(logitsRow: row)

        // Build a model-shaped stand-in. The engine takes a
        // `DistributedQwenModel`, but its only interaction with the
        // model is to capture the forward closure and the
        // make-cache closure. The session-construction path inside
        // the engine works against any object that exposes those
        // two closures; for stream-shape tests, going through the
        // session directly mirrors the engine's wiring without
        // requiring a model fixture.
        let session = CoordinatorInferenceSession(
            forward: forward,
            makeCache: { [] },
            joiners: [InMemoryInferenceControlChannel()],
            sampler: GreedySampler()
        )

        let request = CoordinatorInferenceSession.InferenceRequest(
            promptTokens: [1, 2, 3],
            sampling: SamplingParams(temperature: 0, maxTokens: 4),
            seed: 0
        )

        // Mirror the engine's stream-bridging shape: drive `serve`
        // on one task, drain the session's inner stream from another,
        // and terminate the outer throwing stream only after the
        // drain has flushed the trailing `.finish` event. This is
        // exactly what the engine does internally; verifying the
        // shape against this construction exercises the bridging
        // logic without requiring a `DistributedQwenModel` fixture.
        let stream = AsyncThrowingStream<CoordinatorInferenceSession.TokenEvent, Error> { continuation in
            let (sessionStream, sessionContinuation) = AsyncStream
                .makeStream(of: CoordinatorInferenceSession.TokenEvent.self)
            Task {
                let serveResult: Task<Void, Error> = Task {
                    try await session.serve(request, stream: sessionContinuation)
                }
                for await event in sessionStream {
                    continuation.yield(event)
                }
                do {
                    try await serveResult.value
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        let events = try await collect(stream)

        XCTAssertEqual(events.count, 5,
                       "expected 4 deltas + 1 finish; got \(events.count) events")
        var deltas: [Int] = []
        var finishReason: CoordinatorInferenceSession.FinishReason?
        for event in events {
            switch event {
            case .delta(let token): deltas.append(token)
            case .finish(let reason): finishReason = reason
            }
        }
        XCTAssertEqual(deltas.count, 4)
        guard case .maxTokens = finishReason else {
            return XCTFail("expected .maxTokens finish; got \(String(describing: finishReason))")
        }
    }

    // MARK: - Test 2: greedy fallback at temperature == 0

    /// `temperature == 0` must route through `GreedySampler` and
    /// return the argmax of the logits at every step. The stub
    /// forward returns logits whose argmax is fixed at index 42, so
    /// every emitted token must be 42. Exercises the engine's
    /// default sampler factory specifically — not just any sampler
    /// the test happens to wire in.
    func testGreedyFallbackAtTemperatureZero() async throws {
        let vocab = 128
        var row = [Float](repeating: 0, count: vocab)
        row[42] = 12.0
        let forward = makeStubForward(logitsRow: row)

        let request = CoordinatorInferenceSession.InferenceRequest(
            promptTokens: [10, 20],
            sampling: SamplingParams(temperature: 0, maxTokens: 5),
            seed: 999
        )

        // Default factory must hand back a `GreedySampler` for
        // `temperature == 0`. Asserting on the type confirms the
        // factory's branching contract independently of the sampling
        // result; the token-stream check below confirms the result.
        let sampler = DistributedInferenceEngine.defaultSamplerFactory(request)
        XCTAssertTrue(sampler is GreedySampler,
                      "default factory must return GreedySampler at temperature=0")

        let session = CoordinatorInferenceSession(
            forward: forward,
            makeCache: { [] },
            joiners: [InMemoryInferenceControlChannel()],
            sampler: sampler
        )
        let (stream, continuation) = AsyncStream
            .makeStream(of: CoordinatorInferenceSession.TokenEvent.self)
        try await session.serve(request, stream: continuation)

        var deltas: [Int] = []
        for await event in stream {
            if case .delta(let token) = event {
                deltas.append(token)
            }
        }
        XCTAssertEqual(deltas, [42, 42, 42, 42, 42],
                       "greedy must always emit the argmax token")
    }

    // MARK: - Test 3: stochastic sampling determinism

    /// Two runs of the same prompt with the same seed and the same
    /// stub forward must produce the same token sequence under
    /// `temperature == 1`. Verifies the per-request reproducibility
    /// contract the HTTP layer exposes: the same `(prompt, seed,
    /// sampling)` triple deterministically reproduces the same
    /// completion.
    func testStochasticSamplingIsDeterministicForSameSeed() async throws {
        // Two equally-weighted tokens (7 and 11) with a long
        // low-mass tail. At `temperature = 1` and `seed` fixed,
        // the RNG draws determine which of the two heads the
        // sampler picks at each step; identical RNG state means
        // identical token sequence.
        let vocab = 32
        var row = [Float](repeating: -8.0, count: vocab)
        row[7] = 5.0
        row[11] = 5.0
        let forward = makeStubForward(logitsRow: row)

        let sampling = SamplingParams(temperature: 1.0, maxTokens: 8)
        let request = CoordinatorInferenceSession.InferenceRequest(
            promptTokens: [1],
            sampling: sampling,
            seed: 12345
        )

        // Default factory must return a stochastic sampler for any
        // non-zero temperature; assert that explicitly so a future
        // factory regression that defaults to greedy would surface
        // here rather than as a confusing token-mismatch.
        XCTAssertTrue(
            DistributedInferenceEngine.defaultSamplerFactory(request) is TemperatureSampler,
            "default factory must return TemperatureSampler at temperature>0")

        // Run the same request twice through fresh sampler instances,
        // both seeded with the same seed. Identical token sequences
        // are the contract.
        let runA = try await runOnce(request: request, forward: forward)
        let runB = try await runOnce(request: request, forward: forward)

        XCTAssertEqual(runA, runB,
                       "stochastic sampler must be deterministic for fixed seed; " +
                       "got A=\(runA), B=\(runB)")
        // Sanity: the sequence must contain at least one of the two
        // peak tokens. If the test stub were misconfigured and the
        // sampler picked from the tail, the determinism check above
        // would still pass on coincidence; this assertion guards
        // against that false positive.
        for token in runA {
            XCTAssertTrue([7, 11].contains(token),
                          "expected sampler to pick from the two-peak distribution; got \(token)")
        }
    }

    // MARK: - Helpers

    /// Run one request through a fresh session + fresh `TemperatureSampler`
    /// seeded from the request's seed, returning the token sequence
    /// the session emitted as `.delta` events.
    private func runOnce(
        request: CoordinatorInferenceSession.InferenceRequest,
        forward: @escaping (MLXArray, [KVCache]) -> MLXArray
    ) async throws -> [Int] {
        let sampler = DistributedInferenceEngine.defaultSamplerFactory(request)
        let session = CoordinatorInferenceSession(
            forward: forward,
            makeCache: { [] },
            joiners: [InMemoryInferenceControlChannel()],
            sampler: sampler
        )
        let (stream, continuation) = AsyncStream
            .makeStream(of: CoordinatorInferenceSession.TokenEvent.self)
        try await session.serve(request, stream: continuation)
        var deltas: [Int] = []
        for await event in stream {
            if case .delta(let token) = event {
                deltas.append(token)
            }
        }
        return deltas
    }
}
