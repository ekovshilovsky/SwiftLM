// BonjourDiscovery unit tests. Covers TXT record encode/decode, the
// model+version compatibility check, and the security-critical
// property that the `cluster` TXT field holds an opaque discovery
// hash rather than a user-chosen cluster name — so two clusters with
// different passphrases on the same network remain mutually invisible.

import XCTest
import TurboQuantKit

final class BonjourDiscoveryTests: XCTestCase {

    // MARK: - Helpers

    private func exampleInfo(
        clusterHash: String? = nil,
        clusterId: String? = nil,
        role: DiscoveryRole = .discovering
    ) -> DiscoveryInfo {
        DiscoveryInfo(
            model: "ekovshilovsky/Qwen2.5-32B-TQ8",
            memoryGB: 128,
            role: role,
            clusterHash: clusterHash,
            clusterId: clusterId,
            version: "0.1.0",
            rdma: .available,
            name: "mac-studio-1"
        )
    }

    // MARK: - TXT record encode/decode

    func testTxtRecordEncoding_coordinator() {
        let info = exampleInfo(
            clusterHash: "a7f3e9b1",
            clusterId: "7f3a8b91-4e2c-4c5d-9e6f-1a2b3c4d5e6f",
            role: .coordinator
        )
        let txt = info.toTxtRecord()
        XCTAssertEqual(txt["model"], "ekovshilovsky/Qwen2.5-32B-TQ8")
        XCTAssertEqual(txt["memory"], "128")
        XCTAssertEqual(txt["role"], "coordinator")
        XCTAssertEqual(txt["cluster"], "a7f3e9b1")
        XCTAssertEqual(txt["id"], "7f3a8b91-4e2c-4c5d-9e6f-1a2b3c4d5e6f")
        XCTAssertEqual(txt["rdma"], "available")
        XCTAssertEqual(txt["name"], "mac-studio-1")
        XCTAssertEqual(txt["version"], "0.1.0")
    }

    func testTxtRecordEncoding_discoveringUsesNoneSentinel() {
        // When no cluster has been joined, the cluster and id TXT fields
        // are emitted as the literal "none" so a node in discovering
        // state is distinguishable from a malformed record with missing
        // keys.
        let info = exampleInfo(role: .discovering)
        let txt = info.toTxtRecord()
        XCTAssertEqual(txt["cluster"], "none")
        XCTAssertEqual(txt["id"], "none")
    }

    func testTxtRecordDecoding_worker() {
        let txt: [String: String] = [
            "model": "ekovshilovsky/Qwen2.5-32B-TQ8",
            "memory": "64",
            "role": "worker",
            "cluster": "a7f3e9b1",
            "id": "7f3a8b91-4e2c-4c5d-9e6f-1a2b3c4d5e6f",
            "version": "0.1.0",
            "rdma": "disabled",
            "name": "macbook-pro",
        ]
        let info = DiscoveryInfo(fromTxtRecord: txt)
        XCTAssertNotNil(info)
        XCTAssertEqual(info?.memoryGB, 64)
        XCTAssertEqual(info?.role, .worker)
        XCTAssertEqual(info?.clusterHash, "a7f3e9b1")
        XCTAssertEqual(info?.clusterId, "7f3a8b91-4e2c-4c5d-9e6f-1a2b3c4d5e6f")
        XCTAssertEqual(info?.rdma, .disabled)
        XCTAssertEqual(info?.name, "macbook-pro")
    }

    func testTxtRecordDecoding_noneSentinelMapsToNil() {
        let txt: [String: String] = [
            "model": "ekovshilovsky/Qwen2.5-32B-TQ8",
            "memory": "32",
            "role": "discovering",
            "cluster": "none",
            "id": "none",
            "version": "0.1.0",
            "rdma": "unsupported",
            "name": "tiny-mac",
        ]
        let info = DiscoveryInfo(fromTxtRecord: txt)
        XCTAssertNotNil(info)
        XCTAssertNil(info?.clusterHash)
        XCTAssertNil(info?.clusterId)
        XCTAssertEqual(info?.rdma, .unsupported)
    }

    func testRoundTrip() {
        let original = exampleInfo(
            clusterHash: "deadbeef",
            clusterId: "7f3a8b91-4e2c-4c5d-9e6f-1a2b3c4d5e6f",
            role: .coordinator
        )
        let decoded = DiscoveryInfo(fromTxtRecord: original.toTxtRecord())
        XCTAssertEqual(decoded, original)
    }

    // MARK: - Compatibility

    func testModelMismatchFiltered() {
        let myModel = "ekovshilovsky/Qwen2.5-32B-TQ8"
        let peer = exampleInfo(clusterHash: "a7f3e9b1",
                               clusterId: "7f3a8b91",
                               role: .coordinator)
        // Same model, same version: compatible.
        XCTAssertTrue(peer.isCompatible(with: myModel, version: "0.1.0"))
        // Different model: incompatible.
        XCTAssertFalse(peer.isCompatible(with: "ekovshilovsky/Qwen3.5-27B-TQ8",
                                         version: "0.1.0"))
        // Different version: incompatible.
        XCTAssertFalse(peer.isCompatible(with: myModel, version: "0.2.0"))
    }

    // MARK: - Security: cluster field is opaque, not a user name

    func testClusterFieldDoesNotRevealUserName() {
        // Security property: the cluster TXT field is a hash derived
        // from the cluster master key, not a user-chosen name. A peer
        // that does not hold the passphrase must not be able to see
        // any user-friendly cluster identifier anywhere in the TXT
        // record — otherwise clusters would be discoverable by name on
        // a shared network even when authentication would fail.
        let info = exampleInfo(
            clusterHash: "a7f3e9b1",
            clusterId: "7f3a8b91-4e2c-4c5d-9e6f-1a2b3c4d5e6f",
            role: .coordinator
        )
        let txt = info.toTxtRecord()
        // The cluster value must be strictly hex, 8 chars.
        let hex = Set("0123456789abcdef")
        let cluster = txt["cluster"] ?? ""
        XCTAssertEqual(cluster.count, 8, "discovery hash must be 8 hex chars")
        XCTAssertTrue(cluster.allSatisfy { hex.contains($0) },
                      "discovery hash must be hex only")
        // No TXT field anywhere should contain a human word like "home-lab".
        for (_, v) in txt {
            XCTAssertFalse(v.lowercased().contains("home-lab"),
                           "TXT record must not leak user-chosen cluster name")
        }
    }

    // MARK: - Discovery hash derivation

    func testDiscoveryHashIsDeterministic() {
        // Same master key → same discovery hash, every time.
        let master = Data(repeating: 0x42, count: 32)
        let h1 = ClusterAuth.deriveDiscoveryHash(master: master)
        let h2 = ClusterAuth.deriveDiscoveryHash(master: master)
        XCTAssertEqual(h1, h2)
    }

    func testDiscoveryHashShapeIs8HexChars() {
        let master = Data(repeating: 0x42, count: 32)
        let hash = ClusterAuth.deriveDiscoveryHash(master: master)
        XCTAssertEqual(hash.count, 8)
        let hex = Set("0123456789abcdef")
        XCTAssertTrue(hash.allSatisfy { hex.contains($0) })
    }

    func testDiscoveryHashDiffersByMasterKey() {
        // Different master keys (different passphrases or different
        // cluster.id salts) produce different hashes — this is the
        // property that makes different clusters invisible to each other.
        let masterA = Data(repeating: 0x11, count: 32)
        let masterB = Data(repeating: 0x22, count: 32)
        XCTAssertNotEqual(
            ClusterAuth.deriveDiscoveryHash(master: masterA),
            ClusterAuth.deriveDiscoveryHash(master: masterB)
        )
    }

    func testDiscoveryHashFlowFromPassphraseAndClusterId() {
        // End-to-end flow: passphrase + cluster.id (UUID bytes used as
        // the Argon2id salt) → master_key → discovery_hash. Two nodes
        // doing the same derivation from the same inputs arrive at the
        // same hash — this is what makes two nodes recognize they
        // belong to the same cluster via the Bonjour TXT record
        // without any round-trip communication.
        let passphrase = "my-home-lab"
        let clusterUUID = UUID(uuidString: "7f3a8b91-4e2c-4c5d-9e6f-1a2b3c4d5e6f")!
        let salt = withUnsafeBytes(of: clusterUUID.uuid) { Data($0) }

        let masterA = ClusterAuth.deriveMasterKey(passphrase: passphrase, salt: salt)
        let masterB = ClusterAuth.deriveMasterKey(passphrase: passphrase, salt: salt)
        XCTAssertEqual(masterA, masterB)

        let hashA = ClusterAuth.deriveDiscoveryHash(master: masterA)
        let hashB = ClusterAuth.deriveDiscoveryHash(master: masterB)
        XCTAssertEqual(hashA, hashB)
        XCTAssertEqual(hashA.count, 8)
    }
}
