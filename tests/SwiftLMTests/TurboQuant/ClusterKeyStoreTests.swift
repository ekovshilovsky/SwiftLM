// KeychainClusterKeyStore tests. The store talks to the macOS
// data-protection Keychain via Security.framework in production;
// making those calls under plain `swift test` returns
// errSecMissingEntitlement (-34018) because the xctest host cannot
// carry the `keychain-access-groups` entitlement that macOS's
// restricted-entitlement enforcement requires.
//
// These tests substitute a recording fake for the `SecItemClient`
// dependency, which lets us exercise every branch of the save /
// load / delete logic without any Keychain access at all. That
// covers the code we wrote; what the live Keychain would contribute
// on top of the fake ("does Apple's SecItem* behave as documented")
// is Apple's test, not ours.
//
// If you want to verify against the real Keychain on a development
// machine, see tests/SwiftLMTests/TurboQuant/README.md for the
// codesign recipe.

#if DEBUG

import XCTest
import Security
@testable import TurboQuantKit

// MARK: - Recording fake

/// In-memory `SecItemClient` that mirrors the real SecItem semantics
/// the store relies on: uniqueness keyed on `(service, account)`,
/// `errSecItemNotFound` when the key is absent, `errSecDuplicateItem`
/// when adding over an existing row, `errSecSuccess` otherwise. Every
/// call is recorded so tests can assert on the query attributes the
/// store emitted (access group presence, data-protection flag, etc.)
/// rather than only on end-to-end behavior.
private final class RecordingSecItemClient: SecItemClient, @unchecked Sendable {
    private let lock = NSLock()

    struct Entry: Hashable {
        let service: String
        let account: String
    }

    private(set) var storage: [Entry: Data] = [:]
    private(set) var copyMatchingCalls: [[String: Any]] = []
    private(set) var addCalls: [[String: Any]] = []
    private(set) var deleteCalls: [[String: Any]] = []

    func copyMatchingData(_ query: [String: Any]) -> (OSStatus, Data?) {
        lock.lock(); defer { lock.unlock() }
        copyMatchingCalls.append(query)
        guard let key = Self.key(for: query) else {
            return (errSecParam, nil)
        }
        if let data = storage[key] {
            return (errSecSuccess, data)
        }
        return (errSecItemNotFound, nil)
    }

    func add(_ attributes: [String: Any]) -> OSStatus {
        lock.lock(); defer { lock.unlock() }
        addCalls.append(attributes)
        guard let key = Self.key(for: attributes),
              let data = attributes[kSecValueData as String] as? Data else {
            return errSecParam
        }
        if storage[key] != nil {
            return errSecDuplicateItem
        }
        storage[key] = data
        return errSecSuccess
    }

    func delete(_ query: [String: Any]) -> OSStatus {
        lock.lock(); defer { lock.unlock() }
        deleteCalls.append(query)
        guard let key = Self.key(for: query) else {
            return errSecParam
        }
        if storage.removeValue(forKey: key) != nil {
            return errSecSuccess
        }
        return errSecItemNotFound
    }

    /// Inject a specific entry without going through the public API.
    /// Used to simulate partial / corrupt states.
    func seed(service: String, account: String, data: Data) {
        lock.lock(); defer { lock.unlock() }
        storage[Entry(service: service, account: account)] = data
    }

    private static func key(for query: [String: Any]) -> Entry? {
        guard let service = query[kSecAttrService as String] as? String,
              let account = query[kSecAttrAccount as String] as? String else {
            return nil
        }
        return Entry(service: service, account: account)
    }
}

// MARK: - Tests

final class ClusterKeyStoreTests: XCTestCase {
    private var secItem: RecordingSecItemClient!
    private var store: KeychainClusterKeyStore!
    private let service = "com.turboquant.cluster.test"
    private let accessGroup = "com.turboquant.cluster.test"

    override func setUp() {
        super.setUp()
        secItem = RecordingSecItemClient()
        store = KeychainClusterKeyStore(
            service: service,
            accessGroup: accessGroup,
            secItemClient: secItem
        )
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

    func testDeleteWhenNoRecordIsNotAnError() {
        // Idempotent cleanup path: calling delete on an empty store
        // must not throw. Callers rely on this for "ensure absent"
        // semantics without a prior probe.
        XCTAssertNoThrow(try store.delete())
    }

    // MARK: - Query attributes the store emits

    func testQueryCarriesServiceAndAccountNames() throws {
        try store.save(sampleRecord())

        // Save issues a delete-then-add per account; we get two
        // delete calls (id + key) and two add calls.
        XCTAssertEqual(secItem.deleteCalls.count, 2)
        XCTAssertEqual(secItem.addCalls.count, 2)

        let services = secItem.addCalls.compactMap {
            $0[kSecAttrService as String] as? String
        }
        let accounts = secItem.addCalls.compactMap {
            $0[kSecAttrAccount as String] as? String
        }
        XCTAssertEqual(Set(services), [service],
                       "every add must carry the configured service name")
        XCTAssertEqual(Set(accounts), ["cluster.id", "cluster.key"],
                       "store must use exactly two account names")
    }

    func testAccessGroupIncludedInQueriesWhenConfigured() throws {
        try store.save(sampleRecord())

        for attributes in secItem.addCalls {
            XCTAssertEqual(
                attributes[kSecAttrAccessGroup as String] as? String,
                accessGroup,
                "add query must carry the configured access group"
            )
        }
    }

    func testAccessGroupOmittedWhenNil() throws {
        let clientWithoutGroup = RecordingSecItemClient()
        let storeWithoutGroup = KeychainClusterKeyStore(
            service: service,
            accessGroup: nil,
            secItemClient: clientWithoutGroup
        )
        try storeWithoutGroup.save(sampleRecord())

        for attributes in clientWithoutGroup.addCalls {
            XCTAssertNil(
                attributes[kSecAttrAccessGroup as String],
                "when accessGroup is nil, the query must not carry kSecAttrAccessGroup"
            )
        }
    }

    func testQueryRequestsDataProtectionKeychain() throws {
        try store.save(sampleRecord())
        _ = try store.load()

        // Every query the store builds must set
        // kSecUseDataProtectionKeychain=true so it targets the
        // per-app data-protection Keychain rather than the
        // user-global login keychain.
        for attributes in secItem.addCalls {
            XCTAssertEqual(
                attributes[kSecUseDataProtectionKeychain as String] as? Bool, true,
                "add must opt into the data-protection Keychain"
            )
        }
        for query in secItem.copyMatchingCalls {
            XCTAssertEqual(
                query[kSecUseDataProtectionKeychain as String] as? Bool, true,
                "copyMatching must opt into the data-protection Keychain"
            )
        }
    }

    func testAddSetsAccessibleAfterFirstUnlockThisDeviceOnly() throws {
        try store.save(sampleRecord())

        for attributes in secItem.addCalls {
            XCTAssertEqual(
                attributes[kSecAttrAccessible as String] as? String,
                kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly as String,
                "the working key must be device-bound and unlocked-only"
            )
        }
    }

    // MARK: - Partial / corrupt state

    func testPartialRecordIsTreatedAsAbsent() throws {
        // Seed only the cluster id — simulate a crash between the
        // two SecItemAdd calls a save issues. Load must return nil
        // rather than return a half-populated record that would
        // produce an Argon2 salt mismatch downstream.
        secItem.seed(
            service: service,
            account: "cluster.id",
            data: Data(repeating: 0xAA, count: 16)
        )

        XCTAssertNil(try store.load())
    }

    // MARK: - Error surfacing

    func testUnexpectedStatusOnLoadThrows() {
        // Stand up a client that always returns an unexpected status
        // (here: errSecAuthFailed). The store should wrap it as
        // ClusterKeyStoreError.unexpectedStatus rather than swallow.
        final class FailingClient: SecItemClient, @unchecked Sendable {
            func copyMatchingData(_ query: [String: Any]) -> (OSStatus, Data?) {
                (errSecAuthFailed, nil)
            }
            func add(_ attributes: [String: Any]) -> OSStatus { errSecSuccess }
            func delete(_ query: [String: Any]) -> OSStatus { errSecSuccess }
        }

        let failingStore = KeychainClusterKeyStore(
            service: service,
            accessGroup: accessGroup,
            secItemClient: FailingClient()
        )

        XCTAssertThrowsError(try failingStore.load()) { error in
            guard case ClusterKeyStoreError.unexpectedStatus(let status) = error else {
                XCTFail("expected ClusterKeyStoreError.unexpectedStatus, got \(error)")
                return
            }
            XCTAssertEqual(status, errSecAuthFailed)
        }
    }
}

#endif // DEBUG
