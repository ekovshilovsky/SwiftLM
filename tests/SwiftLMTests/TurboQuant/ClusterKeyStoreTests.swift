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
// xctest binary that cannot be modified without breaking Xcode. Real
// coverage of this path requires an entitled Level 2 test host.
//
// Each run uses a unique service name so concurrent or re-run tests
// cannot pollute one another; `delete()` in tearDown keeps the host
// Keychain clean regardless of individual test outcomes.

import XCTest
import TurboQuantKit

final class ClusterKeyStoreTests: XCTestCase {
    private var store: KeychainClusterKeyStore!

    /// OSStatus errSecMissingEntitlement. Returned by data-protection
    /// Keychain calls when the calling process lacks
    /// `keychain-access-groups` in its code signature.
    private static let errSecMissingEntitlement: OSStatus = -34018

    override func setUpWithError() throws {
        try super.setUpWithError()

        // Unique per-test-run service. Access group matches the
        // entitlements plist embedded into the xctest bundle via
        // Package.swift linker flags (__TEXT __entitlements); the
        // group only takes effect when the host process signature
        // carries the same group (Level 2 test runs).
        let suffix = UUID().uuidString.prefix(8)
        store = KeychainClusterKeyStore(
            service: "com.turboquant.cluster.test.\(suffix)",
            accessGroup: "com.turboquant.cluster.test")

        // Probe once: if the host can't talk to the data-protection
        // Keychain, skip the entire test case rather than surface every
        // test as a failure. One probe covers all test methods.
        do {
            _ = try store.load()
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
        // Best-effort cleanup; if the entitlement check failed in setUp
        // the store may not have been used.
        try? store?.delete()
        super.tearDown()
    }

    private func sampleRecord(
        idByte: UInt8 = 0xA1, keyByte: UInt8 = 0xB2
    ) -> ClusterRecord {
        ClusterRecord(
            clusterId: Data(repeating: idByte, count: 16),
            key: Data(repeating: keyByte, count: 32)
        )
    }

    // MARK: - Round-trip

    func testSaveAndLoadRoundTrip() throws {
        let record = sampleRecord()
        try store.save(record)

        let loaded = try store.load()
        XCTAssertEqual(loaded, record)
        XCTAssertEqual(loaded?.clusterId.count, 16)
        XCTAssertEqual(loaded?.key.count, 32)
    }

    func testLoadReturnsNilWhenNoRecordStored() throws {
        let loaded = try store.load()
        XCTAssertNil(loaded)
    }

    func testSaveOverwritesPriorRecord() throws {
        try store.save(sampleRecord(idByte: 0x11, keyByte: 0xAA))
        try store.save(sampleRecord(idByte: 0x22, keyByte: 0xBB))

        let loaded = try store.load()
        XCTAssertEqual(loaded?.clusterId, Data(repeating: 0x22, count: 16))
        XCTAssertEqual(loaded?.key, Data(repeating: 0xBB, count: 32))
    }

    // MARK: - Delete

    func testDeleteRemovesRecord() throws {
        try store.save(sampleRecord())
        XCTAssertNotNil(try store.load())

        try store.delete()
        XCTAssertNil(try store.load())
    }

    func testDeleteWhenNoRecordIsNotAnError() throws {
        // Idempotent cleanup path: calling delete on an empty store
        // must not throw. Callers rely on this for "ensure absent"
        // semantics without a prior probe.
        XCTAssertNoThrow(try store.delete())
    }
}
