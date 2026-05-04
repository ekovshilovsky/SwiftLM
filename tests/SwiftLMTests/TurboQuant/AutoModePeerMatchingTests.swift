// Tests for the cryptographic peer-matching helper that drives `--auto`
// role resolution. The full auto-mode flow (browse + match + create-or-join)
// requires Bonjour and is exercised by the two-process integration test;
// the unit tests here lock in the pure matching logic so a regression in
// the per-peer derivation path surfaces immediately.

import XCTest
import Network
@testable import TurboQuantKit

final class AutoModePeerMatchingTests: XCTestCase {

    // MARK: - Helpers

    /// Build a coordinator peer whose advertised `clusterHash` is the
    /// real `deriveDiscoveryHash(deriveHandshakeKey(passphrase, clusterId))`
    /// value. This is the same derivation `ClusterManager.createCluster`
    /// does on the wire, so a peer constructed here matches what an
    /// auto-mode browser would actually see from a real coordinator.
    private func coordinatorPeer(
        passphrase: String,
        clusterId: UUID,
        model: String = "Qwen/Qwen2.5-Coder-3B",
        version: String = defaultSwiftlmClusterVersion,
        rdma: RdmaCapability = .unsupported,
        name: String = "coord-host"
    ) -> DiscoveredPeer {
        var bytes = clusterId.uuid
        let uuidBytes = withUnsafeBytes(of: &bytes) { Data($0) }
        let handshakeKey = ClusterAuth.deriveHandshakeKey(
            passphrase: passphrase, salt: uuidBytes
        )
        let discoveryHash = ClusterAuth.deriveDiscoveryHash(master: handshakeKey)
        let info = DiscoveryInfo(
            model: model,
            memoryGB: 128,
            role: .coordinator,
            clusterHash: discoveryHash,
            clusterId: clusterId.uuidString,
            version: version,
            rdma: rdma,
            name: name
        )
        return DiscoveredPeer(
            info: info,
            endpoint: .hostPort(host: "localhost", port: 7777)
        )
    }

    // MARK: - Match path

    /// A coordinator advertised with a hash derived from the local
    /// passphrase produces a positive match. The function returns the
    /// peer unchanged so the caller can plumb it into `joinCluster`
    /// without re-discovery.
    func testMatchSucceedsWhenPassphraseMatches() {
        let passphrase = "correct-horse-battery-staple"
        let peer = coordinatorPeer(
            passphrase: passphrase, clusterId: UUID()
        )
        let result = matchAutoModePeer(
            peer, passphrase: passphrase,
            modelId: "Qwen/Qwen2.5-Coder-3B",
            version: defaultSwiftlmClusterVersion
        )
        XCTAssertNotNil(result, "matching passphrase must produce a positive match")
        XCTAssertEqual(result?.info.clusterId, peer.info.clusterId)
    }

    // MARK: - Reject paths

    /// A coordinator advertising a hash derived from a different
    /// passphrase produces no match. This is the cryptographic core of
    /// auto-mode: only nodes holding the same passphrase as the
    /// coordinator can match it.
    func testMatchFailsWhenPassphraseDiffers() {
        let peer = coordinatorPeer(
            passphrase: "passphrase-A-correct", clusterId: UUID()
        )
        let result = matchAutoModePeer(
            peer, passphrase: "passphrase-B-different",
            modelId: "Qwen/Qwen2.5-Coder-3B",
            version: defaultSwiftlmClusterVersion
        )
        XCTAssertNil(result, "different passphrase must not produce a match")
    }

    /// Worker and discovering peers do not participate in auto-mode
    /// matching — only `.coordinator` peers do, since only coordinators
    /// have a stable cluster identity to join against.
    func testMatchFailsWhenPeerIsNotCoordinator() {
        let passphrase = "correct-horse-battery-staple"
        let coord = coordinatorPeer(passphrase: passphrase, clusterId: UUID())
        let workerInfo = DiscoveryInfo(
            model: coord.info.model,
            memoryGB: coord.info.memoryGB,
            role: .worker,
            clusterHash: coord.info.clusterHash,
            clusterId: coord.info.clusterId,
            version: coord.info.version,
            rdma: coord.info.rdma,
            name: "worker-host"
        )
        let workerPeer = DiscoveredPeer(
            info: workerInfo,
            endpoint: .hostPort(host: "localhost", port: 7778)
        )
        let result = matchAutoModePeer(
            workerPeer, passphrase: passphrase,
            modelId: coord.info.model,
            version: coord.info.version
        )
        XCTAssertNil(result, "non-coordinator peers must not participate in auto-match")
    }

    /// Different model IDs are different clusters even with the same
    /// passphrase. Cross-model joining would corrupt inference because
    /// the joiner's local weights would not match the coordinator's
    /// shard plan.
    func testMatchFailsWhenModelDiffers() {
        let passphrase = "correct-horse-battery-staple"
        let peer = coordinatorPeer(
            passphrase: passphrase, clusterId: UUID(),
            model: "Qwen/Qwen2.5-Coder-3B"
        )
        let result = matchAutoModePeer(
            peer, passphrase: passphrase,
            modelId: "Qwen/Qwen2.5-7B",
            version: defaultSwiftlmClusterVersion
        )
        XCTAssertNil(result, "differing model IDs must not match")
    }

    /// Different cluster wire-protocol versions must not cross-join.
    /// Version compatibility is enforced at advertise time via
    /// `DiscoveryInfo.isCompatible(with:version:)`, but the unit test
    /// proves the auto-mode helper threads the check through correctly.
    func testMatchFailsWhenVersionDiffers() {
        let passphrase = "correct-horse-battery-staple"
        let peer = coordinatorPeer(
            passphrase: passphrase, clusterId: UUID(),
            version: defaultSwiftlmClusterVersion
        )
        let result = matchAutoModePeer(
            peer, passphrase: passphrase,
            modelId: peer.info.model,
            version: "9.9.9"
        )
        XCTAssertNil(result, "differing wire-protocol versions must not match")
    }

    /// A coordinator without `clusterId` in its TXT record cannot be
    /// matched because the salt for the local handshake-key derivation
    /// is unrecoverable. Treat as "not our cluster" rather than failing.
    func testMatchFailsWhenClusterIdAbsent() {
        let passphrase = "correct-horse-battery-staple"
        let info = DiscoveryInfo(
            model: "Qwen/Qwen2.5-Coder-3B",
            memoryGB: 128,
            role: .coordinator,
            clusterHash: "deadbeef",
            clusterId: nil,
            version: defaultSwiftlmClusterVersion,
            rdma: .unsupported,
            name: "coord-host"
        )
        let peer = DiscoveredPeer(
            info: info, endpoint: .hostPort(host: "localhost", port: 7777)
        )
        let result = matchAutoModePeer(
            peer, passphrase: passphrase,
            modelId: info.model, version: info.version
        )
        XCTAssertNil(result, "missing clusterId must produce nil")
    }

    /// A malformed (non-UUID) `clusterId` is treated as no match
    /// rather than a hard error. A real cluster never advertises a
    /// non-UUID, so this is defensive against a future TXT-record
    /// schema bug or a malicious advertiser.
    func testMatchFailsWhenClusterIdNotUUID() {
        let passphrase = "correct-horse-battery-staple"
        let info = DiscoveryInfo(
            model: "Qwen/Qwen2.5-Coder-3B",
            memoryGB: 128,
            role: .coordinator,
            clusterHash: "deadbeef",
            clusterId: "not-a-uuid",
            version: defaultSwiftlmClusterVersion,
            rdma: .unsupported,
            name: "coord-host"
        )
        let peer = DiscoveredPeer(
            info: info, endpoint: .hostPort(host: "localhost", port: 7777)
        )
        let result = matchAutoModePeer(
            peer, passphrase: passphrase,
            modelId: info.model, version: info.version
        )
        XCTAssertNil(result, "malformed clusterId must produce nil")
    }
}
