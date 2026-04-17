// KeychainClusterKeyStore tests. These exercise the real macOS data-
// protection Keychain via Security.framework. They run for real under
// an entitled test host (Xcode with a Developer ID signed test target)
// and skip cleanly under `swift test`, where the default xctest host
// has no `keychain-access-groups` entitlement and every call returns
// OSStatus -34018 (errSecMissingEntitlement).
//
// The skip is not a workaround for a bad test — it reflects a platform
// constraint: macOS checks Keychain access against the host process's
// code signature, which for `swift test` is Apple's ad-hoc-signed
// xctest binary that we cannot modify without breaking Xcode. Real
// coverage of this path is Level 2 (Xcode test host) work.
//
// Each run uses a unique service name so concurrent or re-run tests
// cannot pollute one another; `deleteAll()` in tearDown keeps the host
// Keychain clean regardless of individual test outcomes.

import XCTest
import TurboQuantKit

final class ClusterKeyStoreTests: XCTestCase {
    private var store: KeychainClusterKeyStore!

    /// OSStatus errSecMissingEntitlement. Hit by data-protection
    /// Keychain calls when the calling process lacks
    /// `keychain-access-groups` in its code signature.
    private static let errSecMissingEntitlement: OSStatus = -34018

    override func setUpWithError() throws {
        try super.setUpWithError()

        // Unique per-test-run service so parallel `swift test` invocations
        // and accidental leftovers from prior runs cannot cross-contaminate.
        // Access group matches the entitlements plist embedded into the
        // xctest binary by Package.swift linker flags (__TEXT __entitlements);
        // it only takes effect when the host process signature carries the
        // same group (i.e. Level 2 test runs).
        let suffix = UUID().uuidString.prefix(8)
        store = KeychainClusterKeyStore(
            service: "com.turboquant.cluster.test.\(suffix)",
            accessGroup: "com.turboquant.cluster.test")

        // Probe: if the host can't access the Keychain at all, skip the
        // whole test case rather than reporting false failures. One probe
        // covers every test method.
        do {
            _ = try store.load(cluster: "_probe_")
        } catch let ClusterKeyStoreError.unexpectedStatus(status)
            where status == Self.errSecMissingEntitlement {
            throw XCTSkip(
                "Data-protection Keychain is unavailable: host process lacks " +
                "the keychain-access-groups entitlement (OSStatus -34018). " +
                "Run these tests via Xcode with a signed test host to cover " +
                "this code path."
            )
        }
    }

    override func tearDown() {
        // Best-effort cleanup; a test may already have deleted everything,
        // and if the entitlement check failed in setUp the store may not
        // have been used at all.
        try? store?.deleteAll()
        super.tearDown()
    }

    func testStoreAndLoadRoundTrip() throws {
        let key = Data((0..<32).map { UInt8($0) })
        try store.store(key: key, for: "home-lab")

        let loaded = try store.load(cluster: "home-lab")
        XCTAssertEqual(loaded, key)
        XCTAssertEqual(loaded?.count, 32)
    }

    func testLoadReturnsNilForMissingCluster() throws {
        let loaded = try store.load(cluster: "no-such-cluster")
        XCTAssertNil(loaded)
    }

    func testStoreOverwritesPriorKey() throws {
        let first = Data(repeating: 0xAA, count: 32)
        let second = Data(repeating: 0xBB, count: 32)

        try store.store(key: first, for: "home-lab")
        try store.store(key: second, for: "home-lab")

        let loaded = try store.load(cluster: "home-lab")
        XCTAssertEqual(loaded, second)
    }

    func testDeleteRemovesKey() throws {
        let key = Data(repeating: 0xCC, count: 32)
        try store.store(key: key, for: "home-lab")
        XCTAssertEqual(try store.load(cluster: "home-lab"), key)

        try store.delete(cluster: "home-lab")
        XCTAssertNil(try store.load(cluster: "home-lab"))
    }

    func testDeleteMissingClusterIsNotAnError() throws {
        // Deleting a cluster that was never stored must succeed silently so
        // idempotent cleanup paths don't need to probe-then-delete.
        XCTAssertNoThrow(try store.delete(cluster: "never-existed"))
    }

    func testMultipleClustersAreIndependent() throws {
        let keyA = Data(repeating: 0x11, count: 32)
        let keyB = Data(repeating: 0x22, count: 32)

        try store.store(key: keyA, for: "cluster-a")
        try store.store(key: keyB, for: "cluster-b")

        XCTAssertEqual(try store.load(cluster: "cluster-a"), keyA)
        XCTAssertEqual(try store.load(cluster: "cluster-b"), keyB)

        // Deleting one must not affect the other.
        try store.delete(cluster: "cluster-a")
        XCTAssertNil(try store.load(cluster: "cluster-a"))
        XCTAssertEqual(try store.load(cluster: "cluster-b"), keyB)
    }

    // MARK: - End-to-end with the real derivation path

    // Full flow: derive master key via Argon2id + HKDF, persist it, load
    // it back, derive subkeys from the loaded master, confirm subkeys
    // produced from stored-and-loaded material match those from the
    // original in-memory material. This is the test that proves the
    // Keychain round-trip preserves key material byte-for-byte through a
    // real derivation use case.
    func testEndToEnd_deriveStoreLoadSubkeys() throws {
        let salt = Data(repeating: 0x5A, count: 16)
        let originalMaster = ClusterAuth.deriveMasterKey(
            passphrase: "correct horse battery staple", salt: salt)

        try store.store(key: originalMaster, for: "home-lab")

        let loadedMaster = try store.load(cluster: "home-lab")
        XCTAssertEqual(loadedMaster, originalMaster)

        let originalSubkey = ClusterAuth.deriveSubkey(
            master: originalMaster, info: "tq-handshake-auth")
        let roundTripSubkey = ClusterAuth.deriveSubkey(
            master: loadedMaster!, info: "tq-handshake-auth")

        XCTAssertEqual(originalSubkey.withUnsafeBytes { Data($0) },
                       roundTripSubkey.withUnsafeBytes { Data($0) })
    }

    func testDeleteAllRemovesEverythingUnderService() throws {
        try store.store(key: Data(repeating: 0x01, count: 32), for: "a")
        try store.store(key: Data(repeating: 0x02, count: 32), for: "b")
        try store.store(key: Data(repeating: 0x03, count: 32), for: "c")

        try store.deleteAll()

        XCTAssertNil(try store.load(cluster: "a"))
        XCTAssertNil(try store.load(cluster: "b"))
        XCTAssertNil(try store.load(cluster: "c"))
    }
}
