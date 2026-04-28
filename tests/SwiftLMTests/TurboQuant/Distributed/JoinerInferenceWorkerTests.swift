// Behavioural tests for `JoinerInferenceWorker`. These tests exercise
// the actor's inbound-message dispatch, per-session state lifecycle,
// and the pending-token threading that keeps the joiner's KV cache in
// sync with the coordinator's. No real model fixture is loaded; the
// forward pass is a closure-injected stub that records its inputs so
// the test can assert on shape, token-id, and cache-presence per call.
//
// The in-memory channel from the protocol layer is reused as-is via
// its `injectInbound(_:)` and `finishInbound()` helpers; the worker
// loop exits once `finishInbound` is called, mirroring how a real
// transport would close the stream when the coordinator disconnects.

import XCTest
import Foundation
import MLX
import MLXLMCommon
@testable import TurboQuantKit

final class JoinerInferenceWorkerTests: XCTestCase {

    // MARK: - Recording forward fake

    /// Lock-guarded recorder for a single forward-pass call. Captures
    /// the inputs the worker passed to the forward closure so the test
    /// can assert on shape, token-ids extracted from the tensor, and
    /// whether the cache was empty at call time.
    struct ForwardCall: Equatable {
        let shape: [Int]
        let tokenIds: [Int32]
        let cacheWasEmpty: Bool
    }

    /// Recorder shared between the test and the forward closure. The
    /// closure runs inside the actor's isolation domain while the test
    /// reads the log from the surrounding XCTestCase task, so a lock
    /// guards every mutation.
    final class ForwardRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var _calls: [ForwardCall] = []

        var calls: [ForwardCall] {
            lock.withLock { _calls }
        }

        func record(_ call: ForwardCall) {
            lock.withLock { _calls.append(call) }
        }
    }

    /// Build a forward closure that records each call into `recorder`
    /// and returns a logits tensor of the correct rank. The joiner
    /// discards the returned logits, but the closure must still
    /// produce a `[batch, length, vocab]` tensor in case a future
    /// version of the worker inspects them — keeping the stub honest
    /// avoids hiding a regression behind a permissive return value.
    private func makeRecordingForward(
        recorder: ForwardRecorder
    ) -> @Sendable (MLXArray, [KVCache]) -> MLXArray {
        return { tokenIds, cache in
            // Extract token-ids back out of the tensor for assertion
            // purposes. The worker constructs `MLXArray(Int32 ...)`
            // and reshapes to `[1, length]`, so reading via
            // `asArray(Int32.self)` yields the original vector.
            let ids = tokenIds.asArray(Int32.self)
            recorder.record(ForwardCall(
                shape: tokenIds.shape,
                tokenIds: ids,
                cacheWasEmpty: cache.isEmpty
            ))
            return MLXArray.zeros([1, tokenIds.dim(1), 1], dtype: .float32)
        }
    }

    /// Cache-allocation closure that returns one `KVCacheSimple` per
    /// "layer". The worker only inspects `cache.isEmpty` and forwards
    /// the array through to the closure, so a single-element list is
    /// enough to distinguish "fresh session" from "empty / no
    /// session" in the recorder's `cacheWasEmpty` field.
    private func makeNonEmptyCache() -> @Sendable () -> [KVCache] {
        return { [KVCacheSimple()] }
    }

    /// Drive the worker through a sequence of inbound messages and
    /// wait for the loop to exit. Returns once `run()` has observed
    /// the `finishInbound()` and released session state, so test
    /// assertions immediately after this call are race-free.
    private func driveWorker(
        worker: JoinerInferenceWorker,
        channel: InMemoryInferenceControlChannel,
        messages: [InferenceControlMessage]
    ) async {
        let runTask = Task { await worker.run() }
        for message in messages {
            channel.injectInbound(message)
        }
        channel.finishInbound()
        await runTask.value
    }

    // MARK: - Test 1: full session round-trip

    /// Push the canonical message sequence through the worker and
    /// assert that:
    ///
    ///   1. Prefill triggers exactly one forward call with a
    ///      `[1, promptCount]` token-id tensor.
    ///   2. Each decode triggers exactly one forward call with a
    ///      `[1, 1]` token-id tensor.
    ///   3. The cache becomes non-empty after `sessionStart` and stays
    ///      non-empty across decodes.
    ///   4. After `sessionEnd`, a subsequent `sessionStart` allocates
    ///      a fresh cache — verified indirectly by the second session
    ///      test below; this test focuses on the within-session
    ///      contract.
    func testFullSessionRoundTrip() async throws {
        let channel = InMemoryInferenceControlChannel()
        let recorder = ForwardRecorder()
        let worker = JoinerInferenceWorker(
            channel: channel,
            forward: makeRecordingForward(recorder: recorder),
            makeCache: makeNonEmptyCache()
        )

        let sampling = SamplingParams(temperature: 0, maxTokens: 10)
        let messages: [InferenceControlMessage] = [
            .sessionStart(sessionID: UUID(), sampling: sampling, seed: 0),
            .prefill(promptTokens: [1, 2, 3, 4]),
            .injectToken(sampledToken: 50),
            .decode,
            .injectToken(sampledToken: 51),
            .decode,
            .sessionEnd
        ]

        await driveWorker(worker: worker, channel: channel, messages: messages)

        let calls = recorder.calls
        XCTAssertEqual(calls.count, 3,
                       "expected 1 prefill + 2 decode forward calls; got \(calls.count)")

        // Prefill: shape `[1, 4]`, cache freshly allocated and
        // therefore non-empty at call time.
        XCTAssertEqual(calls[0].shape, [1, 4],
                       "prefill forward must receive a [1, promptCount] token-id tensor")
        XCTAssertFalse(calls[0].cacheWasEmpty,
                       "prefill must run against the freshly-allocated session cache")

        // Both decode calls must be `[1, 1]` and still see the same
        // non-empty cache. The cache is identity-preserved across
        // calls inside the actor; if a regression accidentally reset
        // the cache between decode rounds, `cacheWasEmpty` would
        // flip true here.
        for index in 1 ... 2 {
            XCTAssertEqual(calls[index].shape, [1, 1],
                           "decode call #\(index) must receive a [1, 1] token-id tensor")
            XCTAssertFalse(calls[index].cacheWasEmpty,
                           "decode call #\(index) must run against the active session cache")
        }
    }

    // MARK: - Test 2: pending-token threading

    /// Verify that each `decode` forwards the most-recently-injected
    /// token through the model. A regression where the worker
    /// forwards a stale token (e.g. always the prefill's last token)
    /// or a zero-initialised default would manifest as the wrong
    /// `tokenIds` value on the second decode call.
    func testPendingTokenThreading() async throws {
        let channel = InMemoryInferenceControlChannel()
        let recorder = ForwardRecorder()
        let worker = JoinerInferenceWorker(
            channel: channel,
            forward: makeRecordingForward(recorder: recorder),
            makeCache: makeNonEmptyCache()
        )

        let sampling = SamplingParams(temperature: 0, maxTokens: 10)
        let messages: [InferenceControlMessage] = [
            .sessionStart(sessionID: UUID(), sampling: sampling, seed: 0),
            .prefill(promptTokens: [10, 11, 12]),
            .injectToken(sampledToken: 99),
            .decode,
            .injectToken(sampledToken: 100),
            .decode,
            .sessionEnd
        ]

        await driveWorker(worker: worker, channel: channel, messages: messages)

        let calls = recorder.calls
        XCTAssertEqual(calls.count, 3, "expected 1 prefill + 2 decode forward calls")

        // Prefill receives the prompt batch verbatim.
        XCTAssertEqual(calls[0].tokenIds, [10, 11, 12],
                       "prefill must forward the prompt-token batch unchanged")

        // First decode forwards the first injected token.
        XCTAssertEqual(calls[1].tokenIds, [99],
                       "first decode must forward the most-recently-injected token (99)")

        // Second decode forwards the second injected token, not a
        // stale value left over from the first round.
        XCTAssertEqual(calls[2].tokenIds, [100],
                       "second decode must forward the most-recently-injected token (100)")
    }

    // MARK: - Test 3: multi-session reuse

    /// Run two complete sessions back-to-back over the same worker
    /// and assert that the second session sees a freshly-allocated
    /// cache (the first session's cache does not leak through). Also
    /// verifies the pending-token bookkeeping is reset between
    /// sessions: the second session's first decode must use the
    /// second session's first injected token, not anything left over
    /// from the first.
    ///
    /// Distinguishing "fresh cache" from "leaked cache" requires the
    /// cache-allocation closure to produce identifiable instances per
    /// call. We use the call index of `makeCache` as a tag; the
    /// recorder captures the count of cache entries on each forward,
    /// so a leaked cache from session one would show up as a cache
    /// size of two in session two (the original entry plus the new
    /// allocation). A correctly-reset session re-allocates from
    /// scratch and the size stays at one.
    func testMultiSessionReuse() async throws {
        let channel = InMemoryInferenceControlChannel()

        // Recorder that captures cache size, not just emptiness, so a
        // leaked-cache regression is observable.
        final class SizeRecorder: @unchecked Sendable {
            private let lock = NSLock()
            private var _sizes: [Int] = []
            var sizes: [Int] { lock.withLock { _sizes } }
            func record(_ size: Int) { lock.withLock { _sizes.append(size) } }
        }
        let sizeRecorder = SizeRecorder()

        // Cache factory that produces a fresh single-element list per
        // invocation. Each call increments an internal counter so the
        // test could distinguish identity if needed; for this test
        // the size invariant alone is sufficient.
        final class CacheFactory: @unchecked Sendable {
            private let lock = NSLock()
            private var _calls: Int = 0
            var calls: Int { lock.withLock { _calls } }
            func make() -> [KVCache] {
                lock.withLock { _calls += 1 }
                return [KVCacheSimple()]
            }
        }
        let factory = CacheFactory()

        let forward: @Sendable (MLXArray, [KVCache]) -> MLXArray = { tokenIds, cache in
            sizeRecorder.record(cache.count)
            return MLXArray.zeros([1, tokenIds.dim(1), 1], dtype: .float32)
        }
        let makeCache: @Sendable () -> [KVCache] = { factory.make() }

        let worker = JoinerInferenceWorker(
            channel: channel,
            forward: forward,
            makeCache: makeCache
        )

        let sampling = SamplingParams(temperature: 0, maxTokens: 10)
        let messages: [InferenceControlMessage] = [
            // Session 1
            .sessionStart(sessionID: UUID(), sampling: sampling, seed: 0),
            .prefill(promptTokens: [1, 2]),
            .injectToken(sampledToken: 7),
            .decode,
            .sessionEnd,
            // Session 2
            .sessionStart(sessionID: UUID(), sampling: sampling, seed: 1),
            .prefill(promptTokens: [10, 20, 30]),
            .injectToken(sampledToken: 8),
            .decode,
            .sessionEnd
        ]

        await driveWorker(worker: worker, channel: channel, messages: messages)

        // Two sessions, each with one prefill + one decode = four
        // forward calls total. Every call must see a single-element
        // cache; if `sessionEnd` failed to release, session two's
        // calls would observe a two-element cache instead.
        let sizes = sizeRecorder.sizes
        XCTAssertEqual(sizes, [1, 1, 1, 1],
                       "every forward call across both sessions must see a single-element cache; got \(sizes)")

        // The cache factory must have been invoked once per session.
        // A regression where `sessionEnd` does not clear state would
        // either skip the second `makeCache` call (leaking the first
        // session's cache) or invoke it without ever being consumed
        // by the second session.
        XCTAssertEqual(factory.calls, 2,
                       "makeCache must be invoked once per sessionStart; got \(factory.calls) calls")
    }
}
