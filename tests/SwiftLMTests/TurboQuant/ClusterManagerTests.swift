// End-to-end tests for ClusterManager. Two ClusterManager instances
// in one process exercise the real create / advertise / discover /
// join / keystore-persist flow over loopback mDNS and loopback TCP.
// Both managers use their own InMemoryClusterKeyStore so failure vs
// success can be asserted without touching disk.
//
// The tests are wrapped in XCTSkip guards for the environments that
// block multicast DNS (containers, sandboxed CI, locked-down build
// farms). On a developer Mac with mDNSResponder running, both tests
// complete in a couple of seconds.

#if DEBUG

import XCTest
import Network
import TurboQuantKit

final class ClusterManagerTests: XCTestCase {

    // MARK: - Fixture

    /// A pair of fully-wired managers under test: coordinator side
    /// (the one that calls `createCluster`) and joiner side (the one
    /// that calls `joinCluster`). Each side gets its own key store
    /// and its own BonjourService so their state is fully
    /// independent within the same process.
    private struct Fixture {
        let coordinator: ClusterManager
        let coordinatorStore: InMemoryClusterKeyStore
        let coordinatorBonjour: BonjourService
        let coordinatorName: String

        let joiner: ClusterManager
        let joinerStore: InMemoryClusterKeyStore
        let joinerBonjour: BonjourService
        let joinerName: String
    }

    private func uniqueName(_ prefix: String) -> String {
        "\(prefix)-\(UUID().uuidString.prefix(8))"
    }

    private func makeFixture() -> Fixture {
        let coordName = uniqueName("coord-host")
        let joinerName = uniqueName("joiner-host")

        // Coordinator starts in .discovering state (no cluster yet);
        // createCluster will flip its advertised info to .coordinator
        // with the real hash + UUID once the key material is minted.
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

        let coordStore = InMemoryClusterKeyStore()
        let joinerStore = InMemoryClusterKeyStore()

        let coordBonjour = BonjourService(info: coordInfo, localClusterHash: nil)
        let joinerBonjour = BonjourService(info: joinerInfo, localClusterHash: nil)

        let coordinator = ClusterManager(
            keyStore: coordStore,
            bonjour: coordBonjour,
            localHostname: coordName,
            model: "ekovshilovsky/Qwen2.5-32B-TQ8",
            memoryGB: 128,
            version: "0.1.0",
            rdma: .available
        )
        let joiner = ClusterManager(
            keyStore: joinerStore,
            bonjour: joinerBonjour,
            localHostname: joinerName,
            model: "ekovshilovsky/Qwen2.5-32B-TQ8",
            memoryGB: 64,
            version: "0.1.0",
            rdma: .available
        )

        return Fixture(
            coordinator: coordinator,
            coordinatorStore: coordStore,
            coordinatorBonjour: coordBonjour,
            coordinatorName: coordName,
            joiner: joiner,
            joinerStore: joinerStore,
            joinerBonjour: joinerBonjour,
            joinerName: joinerName
        )
    }

    /// Start the joiner's BonjourService (so it can browse) and wait
    /// for the coordinator's advertised record to appear. Returns
    /// the discovered peer or nil on timeout.
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

    // MARK: - Positive path

    func testCreateThenJoinSucceeds() async throws {
        let fx = makeFixture()
        defer {
            fx.coordinatorBonjour.stop()
            fx.joinerBonjour.stop()
        }

        let passphrase = "correct-horse-battery-staple"

        let coordinatorRecord: ClusterRecord
        do {
            coordinatorRecord = try await fx.coordinator.createCluster(
                passphrase: passphrase,
                clusterName: "test-cluster"
            )
        } catch {
            throw XCTSkip("createCluster failed — mDNSResponder or loopback listener likely unavailable: \(error)")
        }

        // Joiner has to start its own BonjourService before it can
        // browse. Doing it here (rather than in the fixture) keeps
        // the failure-mode asymmetry clear: create failure is a
        // coordinator problem; discover failure is a joiner problem.
        do {
            _ = try await fx.joinerBonjour.start()
        } catch {
            throw XCTSkip("joiner BonjourService failed to start: \(error)")
        }

        guard let peer = try await discoverCoordinator(
            joinerBonjour: fx.joinerBonjour,
            coordinatorName: fx.coordinatorName,
            timeoutSeconds: 10
        ) else {
            throw XCTSkip("mDNSResponder did not deliver the coordinator peer within 10s")
        }

        // Sanity on the TXT fields — if these do not match, the
        // subsequent handshake will fail for a reason unrelated to
        // the code under test.
        XCTAssertEqual(peer.info.role, .coordinator)
        XCTAssertNotNil(peer.info.clusterHash)
        XCTAssertNotNil(peer.info.clusterId)

        let joinerRecord = try await fx.joiner.joinCluster(
            passphrase: passphrase,
            peer: peer
        )

        // Both managers should agree on the cluster identity AND the
        // working key — the entire point of the handshake.
        XCTAssertEqual(coordinatorRecord.clusterId, joinerRecord.clusterId)
        XCTAssertEqual(coordinatorRecord.key, joinerRecord.key)
        XCTAssertEqual(coordinatorRecord.key.count, 32)

        let coordStoredRecord = try fx.coordinatorStore.load()
        let joinerStoredRecord = try fx.joinerStore.load()
        XCTAssertNotNil(coordStoredRecord)
        XCTAssertNotNil(joinerStoredRecord)
        XCTAssertEqual(coordStoredRecord, joinerStoredRecord)
    }

    // MARK: - Wrong-passphrase rejection + retry

    func testWrongPassphraseFailsThenRetrySucceeds() async throws {
        let fx = makeFixture()
        defer {
            fx.coordinatorBonjour.stop()
            fx.joinerBonjour.stop()
        }

        let correctPassphrase = "correct-horse"
        let wrongPassphrase   = "wrong-horse"

        let coordinatorRecord: ClusterRecord
        do {
            coordinatorRecord = try await fx.coordinator.createCluster(
                passphrase: correctPassphrase,
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
            timeoutSeconds: 10
        ) else {
            throw XCTSkip("mDNSResponder did not deliver the coordinator peer within 10s")
        }

        // First attempt: wrong passphrase. The handshake HMAC
        // verification must fail with .peerAuthFailed; any other
        // error class means we got a spurious transport failure and
        // the assertion becomes noise.
        do {
            _ = try await fx.joiner.joinCluster(
                passphrase: wrongPassphrase,
                peer: peer
            )
            XCTFail("join with wrong passphrase should have thrown")
        } catch let err as ClusterHandshakeError {
            XCTAssertEqual(err, .peerAuthFailed,
                           "wrong-passphrase join should surface .peerAuthFailed, got \(err)")
        } catch {
            XCTFail("wrong-passphrase join threw unexpected error: \(error)")
        }

        // The joiner's key store must not have been mutated by the
        // failed attempt — partial state would leave the node in an
        // unusable half-joined condition.
        XCTAssertNil(try fx.joinerStore.load(),
                     "joiner key store must remain empty after a failed handshake")

        // Second attempt: correct passphrase. Must succeed end-to-end
        // against the same coordinator (the coordinator's accept loop
        // is resilient against the prior failure).
        let joinerRecord = try await fx.joiner.joinCluster(
            passphrase: correctPassphrase,
            peer: peer
        )
        XCTAssertEqual(coordinatorRecord.clusterId, joinerRecord.clusterId)
        XCTAssertEqual(coordinatorRecord.key, joinerRecord.key)

        let joinerStoredRecord = try fx.joinerStore.load()
        XCTAssertEqual(joinerStoredRecord, joinerRecord)
    }
}

#endif // DEBUG
