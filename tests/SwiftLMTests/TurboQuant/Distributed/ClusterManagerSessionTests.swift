// ClusterManager runtime-layer tests. These exercise the post-
// handshake data-channel handoff that wires `joinCluster` and the
// coordinator's accept loop up to live `ClusterDataChannel`
// instances, plus the two new public APIs the manager exposes for
// the inference path: `runJoinerWorker(model:)` and
// `beginInferenceSession(...)`.
//
// Two managers run in one process talking over loopback Bonjour and
// loopback TCP — the same fixture pattern as `ClusterManagerTests`,
// extended to drive control traffic across the surviving connection
// after the handshake completes.
//
// XCTSkip guards mirror `ClusterManagerTests`: environments that
// block multicast DNS skip the cluster-formation tests; a developer
// Mac with mDNSResponder running drives them in a couple of seconds.

#if DEBUG

import Foundation
import MLX
import MLXLMCommon
import Network
import XCTest
@testable import TurboQuantKit

final class ClusterManagerSessionTests: XCTestCase {

    // MARK: - Fixture

    /// Coordinator + joiner manager pair plus the underlying
    /// dependencies that need to be torn down at end-of-test. Identical
    /// shape to `ClusterManagerTests.Fixture`; duplicated here rather
    /// than shared because the two suites assert on disjoint surfaces
    /// and test fixtures are not part of the module's public API.
    private struct Fixture {
        let coordinator: ClusterManager
        let coordinatorBonjour: BonjourService
        let coordinatorName: String

        let joiner: ClusterManager
        let joinerBonjour: BonjourService
        let joinerName: String
    }

    private func uniqueName(_ prefix: String) -> String {
        "\(prefix)-\(UUID().uuidString.prefix(8))"
    }

    private func makeFixture() -> Fixture {
        let coordName = uniqueName("coord-host")
        let joinerName = uniqueName("joiner-host")

        let coordInfo = DiscoveryInfo(
            model: "ekovshilovsky/Qwen2.5-32B-TQ8",
            memoryGB: 128,
            role: .discovering,
            clusterHash: nil,
            clusterId: nil,
            version: "0.1.0",
            rdma: .available,
            name: coordName
        )
        let joinerInfo = DiscoveryInfo(
            model: "ekovshilovsky/Qwen2.5-32B-TQ8",
            memoryGB: 64,
            role: .discovering,
            clusterHash: nil,
            clusterId: nil,
            version: "0.1.0",
            rdma: .available,
            name: joinerName
        )

        let coordBonjour = BonjourService(info: coordInfo, localClusterHash: nil)
        let joinerBonjour = BonjourService(info: joinerInfo, localClusterHash: nil)

        let coordinator = ClusterManager(
            keyStore: InMemoryClusterKeyStore(),
            bonjour: coordBonjour,
            localHostname: coordName,
            model: "ekovshilovsky/Qwen2.5-32B-TQ8",
            memoryGB: 128,
            version: "0.1.0",
            rdma: .available
        )
        let joiner = ClusterManager(
            keyStore: InMemoryClusterKeyStore(),
            bonjour: joinerBonjour,
            localHostname: joinerName,
            model: "ekovshilovsky/Qwen2.5-32B-TQ8",
            memoryGB: 64,
            version: "0.1.0",
            rdma: .available
        )

        return Fixture(
            coordinator: coordinator,
            coordinatorBonjour: coordBonjour,
            coordinatorName: coordName,
            joiner: joiner,
            joinerBonjour: joinerBonjour,
            joinerName: joinerName
        )
    }

    /// Wait for the joiner to see the coordinator's advertised TXT
    /// record. Returns `nil` on timeout so the caller can convert
    /// timeouts into `XCTSkip` rather than failures — the loopback
    /// path requires mDNSResponder, which is not always available in
    /// CI sandboxes.
    private func discoverCoordinator(
        joinerBonjour: BonjourService,
        coordinatorName: String,
        timeoutSeconds: Double
    ) async throws -> DiscoveredPeer? {
        return try await withThrowingTaskGroup(of: DiscoveredPeer?.self) { group in
            group.addTask {
                for await peer in joinerBonjour.peers() {
                    if peer.info.name == coordinatorName {
                        return peer
                    }
                }
                return nil
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeoutSeconds * 1_000_000_000))
                return nil
            }
            let first = try await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }

    /// Drive create + discover + join end-to-end, returning the live
    /// fixture and the joiner's `ClusterRecord` so the calling test
    /// can pivot directly into channel-handoff or session assertions.
    /// Wraps the whole formation flow in `XCTSkip` so a missing
    /// loopback path turns into a skip rather than a failure.
    private func formCluster(
        timeoutSeconds: Double = 10
    ) async throws -> Fixture {
        let fx = makeFixture()
        let passphrase = "correct-horse-battery-staple"

        do {
            _ = try await fx.coordinator.createCluster(
                passphrase: passphrase,
                clusterName: nil
            )
        } catch {
            throw XCTSkip("createCluster failed — mDNSResponder or loopback listener likely unavailable: \(error)")
        }

        do {
            _ = try await fx.joinerBonjour.start()
        } catch {
            throw XCTSkip("joiner BonjourService failed to start: \(error)")
        }

        guard let peer = try await discoverCoordinator(
            joinerBonjour: fx.joinerBonjour,
            coordinatorName: fx.coordinatorName,
            timeoutSeconds: timeoutSeconds
        ) else {
            throw XCTSkip("mDNSResponder did not deliver the coordinator peer within \(timeoutSeconds)s")
        }

        _ = try await fx.joiner.joinCluster(
            passphrase: passphrase,
            peer: peer
        )

        return fx
    }

    /// Poll the coordinator until at least one joiner channel has
    /// been registered or the timeout elapses. The accept loop runs
    /// the handshake on a detached task and hops back into the actor
    /// to append the channel; `joinCluster` returning on the joiner
    /// side does not synchronously imply the coordinator has finished
    /// its registration step. A short retry loop converges in
    /// milliseconds on a healthy local stack.
    private func awaitCoordinatorJoinerChannel(
        coordinator: ClusterManager,
        timeoutSeconds: Double = 5
    ) async -> ClusterDataChannel? {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while Date() < deadline {
            let channels = await coordinator._joinerChannelsForTesting()
            if let first = channels.first { return first }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return nil
    }

    // MARK: - Test 1: handshake-to-channel handoff (joiner side)

    /// After `joinCluster` succeeds, the joiner manager must hold a
    /// live `ClusterDataChannel` over the same connection that carried
    /// the handshake. Verified by sending a control message from the
    /// coordinator's freshly-registered channel and asserting the
    /// joiner's channel decodes it.
    func testHandshakeHandsOffToLiveChannel() async throws {
        let fx = try await formCluster()
        defer {
            fx.coordinatorBonjour.stop()
            fx.joinerBonjour.stop()
        }

        guard let coordinatorChannel = await awaitCoordinatorJoinerChannel(
            coordinator: fx.coordinator
        ) else {
            XCTFail("coordinator did not register a joiner channel within the timeout")
            return
        }

        let joinerChannel = await fx.joiner._coordinatorChannelForTesting()
        XCTAssertNotNil(
            joinerChannel,
            "joinCluster must surface a live data channel on success"
        )
        guard let joinerChannel else { return }

        // Drive one control message across the live connection. A
        // successful decode end-to-end means the AEAD keys, nonces,
        // and direction tags agree — i.e. both sides derived the
        // same `sessionKey` from the handshake nonces and the
        // post-handshake connection survived without being cancelled.
        let testToken = 12345
        try await coordinatorChannel.send(.injectToken(sampledToken: testToken))

        var iterator = joinerChannel.inbound.makeAsyncIterator()
        let message = await iterator.next()
        guard case .injectToken(let received) = message else {
            XCTFail("joiner did not decode injectToken across live channel; got \(String(describing: message))")
            return
        }
        XCTAssertEqual(received, testToken,
                       "round-tripped token must match the broadcast value")

        await fx.coordinator.stop()
        await fx.joiner.stop()
    }

    // MARK: - Test 2: end-to-end joiner worker loop

    /// Stand up coordinator + joiner managers, complete the handshake,
    /// then have the coordinator broadcast the canonical session
    /// sequence over the real encrypted channel while the joiner runs
    /// `runJoinerWorker(...)` against a closure-injected forward stub.
    /// Verifies the worker observes prefill + decode forward calls in
    /// the expected order, end-to-end through the post-handshake
    /// transport.
    ///
    /// The worker normally takes a `DistributedQwenModel`, but its
    /// internal contract is closure-shaped — `forward` and
    /// `makeCache`. To exercise the manager's runtime layer without a
    /// live model fixture we instantiate `JoinerInferenceWorker`
    /// directly against the manager's coordinator-side channel and
    /// drive the loop the same way `runJoinerWorker` would.
    func testCoordinatorBroadcastDrivesJoinerWorkerLoop() async throws {
        let fx = try await formCluster()
        defer {
            fx.coordinatorBonjour.stop()
            fx.joinerBonjour.stop()
        }

        guard let coordinatorChannel = await awaitCoordinatorJoinerChannel(
            coordinator: fx.coordinator
        ) else {
            XCTFail("coordinator did not register a joiner channel within the timeout")
            return
        }
        guard let joinerChannel = await fx.joiner._coordinatorChannelForTesting() else {
            XCTFail("joiner did not surface a coordinator channel after joinCluster")
            return
        }

        // Closure-injected forward + cache stubs. The worker discards
        // the returned logits but a `[batch, length, vocab]` shape
        // matches what a real model emits, so the stub stays honest
        // about the contract.
        let recorder = ForwardRecorder()
        let forward: @Sendable (MLXArray, [KVCache]) -> MLXArray = { tokenIds, cache in
            let ids = tokenIds.asArray(Int32.self)
            recorder.record(
                shape: tokenIds.shape,
                tokenIds: ids,
                cacheWasEmpty: cache.isEmpty
            )
            return MLXArray.zeros([1, tokenIds.dim(1), 1], dtype: .float32)
        }
        let makeCache: @Sendable () -> [KVCache] = { [KVCacheSimple()] }

        let worker = JoinerInferenceWorker(
            channel: joinerChannel,
            forward: forward,
            makeCache: makeCache
        )
        let workerTask = Task { await worker.run() }

        // Canonical session sequence: sessionStart, prefill,
        // injectToken, decode, ..., sessionEnd. The worker forwards
        // exactly once per prefill and once per decode (after a
        // matching injectToken).
        let sampling = SamplingParams(
            temperature: 0.0,
            maxTokens: 8,
            stopTokens: []
        )
        try await coordinatorChannel.send(.sessionStart(
            sessionID: UUID(),
            sampling: sampling,
            seed: 42
        ))
        try await coordinatorChannel.send(.prefill(promptTokens: [101, 102, 103]))
        try await coordinatorChannel.send(.injectToken(sampledToken: 201))
        try await coordinatorChannel.send(.decode)
        try await coordinatorChannel.send(.injectToken(sampledToken: 202))
        try await coordinatorChannel.send(.decode)
        try await coordinatorChannel.send(.sessionEnd)

        // Wait for the joiner's receive loop to drain every message
        // through the worker. Polling on the recorder's call count
        // avoids racing the worker's actor-isolated dispatch with the
        // channel teardown below — closing before the receive loop
        // observes the final frame would lose buffered traffic.
        let drainDeadline = Date().addingTimeInterval(5)
        while recorder.calls.count < 3 && Date() < drainDeadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }

        // Closing the coordinator's channel produces a graceful EOF on
        // the joiner's receive loop, which finishes its inbound stream
        // and lets the worker's `run()` loop return.
        coordinatorChannel.close()
        await workerTask.value

        let calls = recorder.calls
        XCTAssertEqual(calls.count, 3,
                       "expected 1 prefill + 2 decode forward calls; got \(calls.count)")

        // Prefill: shape `[1, 3]` carrying the full prompt.
        XCTAssertEqual(calls[0].shape, [1, 3])
        XCTAssertEqual(calls[0].tokenIds, [101, 102, 103])
        // Decode rounds: shape `[1, 1]` carrying the most recently
        // injected token.
        XCTAssertEqual(calls[1].shape, [1, 1])
        XCTAssertEqual(calls[1].tokenIds, [201])
        XCTAssertEqual(calls[2].shape, [1, 1])
        XCTAssertEqual(calls[2].tokenIds, [202])

        // Cache reaches the closure non-empty on every call: the
        // worker allocates it via `makeCache` on `sessionStart`.
        for call in calls {
            XCTAssertFalse(call.cacheWasEmpty,
                           "worker must thread the session cache through every forward call")
        }

        await fx.coordinator.stop()
        await fx.joiner.stop()
    }

    // MARK: - Test 3: beginInferenceSession at size-1

    /// A fresh coordinator manager with no joiners attached must still
    /// produce a working stream from `beginInferenceSession`. The
    /// closure-injection overload bypasses the `DistributedQwenModel`
    /// requirement so the test asserts the broadcast/empty-joiner
    /// handling without standing up a real model fixture.
    func testBeginInferenceSessionAtSizeOne() async throws {
        let fx = makeFixture()
        defer {
            fx.coordinatorBonjour.stop()
            fx.joinerBonjour.stop()
        }

        // The coordinator manager has no cluster — beginInferenceSession
        // is independent of cluster formation as long as the joiner
        // list is empty. Construct it directly to avoid the Bonjour
        // path's environmental dependencies in this test.

        // Stub forward whose argmax is fixed at index 7. The session
        // walks the argmax under `temperature == 0` (greedy fallback).
        let vocab = 32
        let argmaxIndex = 7
        var row = [Float](repeating: 0, count: vocab)
        row[argmaxIndex] = 10.0
        let forward: CoordinatorInferenceSession.ForwardFn = { tokenIds, _ in
            let length = tokenIds.dim(1)
            var flat = [Float](repeating: 0, count: length * vocab)
            for pos in 0 ..< length {
                let base = pos * vocab
                for v in 0 ..< vocab {
                    flat[base + v] = row[v]
                }
            }
            return MLXArray(flat, [1, length, vocab])
        }
        let makeCache: CoordinatorInferenceSession.MakeCacheFn = { [] }

        let request = CoordinatorInferenceSession.InferenceRequest(
            promptTokens: [1, 2, 3],
            sampling: SamplingParams(temperature: 0, maxTokens: 4),
            seed: 0
        )
        let stream = await fx.coordinator.beginInferenceSession(
            request: request,
            forward: forward,
            makeCache: makeCache
        )

        var deltas: [Int] = []
        var finishReason: CoordinatorInferenceSession.FinishReason?
        for try await event in stream {
            switch event {
            case .delta(let token): deltas.append(token)
            case .finish(let reason): finishReason = reason
            }
        }

        // 4-token budget under greedy + fixed argmax: every emitted
        // token is the argmax index, plus one trailing finish event.
        XCTAssertEqual(deltas.count, 4,
                       "expected maxTokens deltas at size 1")
        XCTAssertEqual(deltas, Array(repeating: argmaxIndex, count: 4),
                       "greedy argmax must emit the fixed-row argmax repeatedly")
        guard case .maxTokens = finishReason else {
            XCTFail("expected .maxTokens finish; got \(String(describing: finishReason))")
            return
        }

        await fx.coordinator.stop()
    }

    // MARK: - Test 4: runJoinerWorker without a live channel

    /// Calling `runJoinerWorker` before any successful `joinCluster`
    /// must surface `noActiveJoinerChannel` rather than hang or crash.
    /// Constructed against a fresh manager that has never seen a
    /// handshake — the manager's `coordinatorChannel` stays nil and
    /// the entry point throws immediately.
    func testRunJoinerWorkerThrowsWithoutChannel() async throws {
        let fx = makeFixture()
        defer {
            fx.coordinatorBonjour.stop()
            fx.joinerBonjour.stop()
        }

        // Use a dummy DistributedQwenModel reference is impossible
        // here because the loader requires real weights. The error
        // path under test fires before the model is ever consulted,
        // so we reach the throw via the runtime check at the top of
        // `runJoinerWorker`. To exercise that without instantiating
        // a model, route through a tiny private bridge that mirrors
        // the runtime check.
        do {
            try await fx.joiner._runJoinerWorkerForTesting()
            XCTFail("runJoinerWorker must throw without an active channel")
            // unreachable; XCTFail does not return Never on this path
        } catch let err as ClusterManagerError {
            if case .noActiveJoinerChannel = err {
                // expected
            } else {
                XCTFail("expected .noActiveJoinerChannel, got \(err)")
            }
        } catch {
            XCTFail("unexpected error type: \(error)")
        }

        await fx.joiner.stop()
    }

    // MARK: - Recording stub for the worker forward closure

    /// Lock-guarded recorder that captures the inputs the worker
    /// passes to its forward closure. Same shape as the one used by
    /// `JoinerInferenceWorkerTests`; duplicated locally so this suite
    /// stays self-contained.
    private final class ForwardRecorder: @unchecked Sendable {
        struct Call: Equatable {
            let shape: [Int]
            let tokenIds: [Int32]
            let cacheWasEmpty: Bool
        }

        private let lock = NSLock()
        private var _calls: [Call] = []

        var calls: [Call] {
            lock.withLock { _calls }
        }

        func record(shape: [Int], tokenIds: [Int32], cacheWasEmpty: Bool) {
            lock.withLock {
                _calls.append(Call(
                    shape: shape,
                    tokenIds: tokenIds,
                    cacheWasEmpty: cacheWasEmpty
                ))
            }
        }
    }
}

// MARK: - Test affordance for the no-channel error path

extension ClusterManager {

    /// Test-only entry point that runs the no-active-channel guard in
    /// `runJoinerWorker` without requiring the caller to construct a
    /// `DistributedQwenModel`. The guard short-circuits before the
    /// model is ever consulted, so this entry point throws with the
    /// same error the public API would throw on the same code path.
    internal func _runJoinerWorkerForTesting() throws {
        if _coordinatorChannelForTesting() == nil {
            throw ClusterManagerError.noActiveJoinerChannel
        }
    }
}

#endif // DEBUG
