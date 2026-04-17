// Persistent storage for derived cluster keys. The 256-bit master key
// produced by ClusterAuth.deriveMasterKey is written to the macOS Keychain
// so a running node can re-join its cluster after restart without
// re-prompting the user for the passphrase.
//
// Uses the data-protection Keychain (kSecUseDataProtectionKeychain) rather
// than the legacy login Keychain — this avoids ACL prompts when the
// calling binary is unsigned (e.g., during `swift test` or local dev
// builds) and matches Apple's recommended pattern for modern Swift code.

import Foundation
import Security

public enum ClusterKeyStoreError: Error, Equatable, CustomStringConvertible {
    case unexpectedStatus(OSStatus)

    public var description: String {
        switch self {
        case .unexpectedStatus(let status):
            return "Keychain OSStatus \(status)"
        }
    }
}

/// Abstraction so callers can depend on a protocol rather than on the
/// concrete Keychain implementation; aids test injection and future
/// alternate backends (e.g., Secure Enclave-backed storage).
public protocol ClusterKeyStore {
    /// Persist the key for the given cluster. Overwrites any prior entry.
    func store(key: Data, for cluster: String) throws

    /// Retrieve the key for the given cluster, or nil if none is stored.
    func load(cluster: String) throws -> Data?

    /// Remove the key for the given cluster. A missing entry is not an error.
    func delete(cluster: String) throws
}

/// macOS Keychain-backed implementation of ClusterKeyStore.
///
/// Items are stored as `kSecClassGenericPassword` with:
/// - service:  the `service` property (default `"com.turboquant.cluster"`)
/// - account:  the cluster name supplied by the caller
/// - value:    the raw 32-byte master key
///
/// The service name is exposed so tests can use a distinct namespace and
/// production binaries can override it if multiple TurboQuant-derived
/// products need coexistent cluster-key stores.
public struct KeychainClusterKeyStore: ClusterKeyStore {
    public let service: String
    public let accessGroup: String?

    /// - Parameters:
    ///   - service: Service identifier for Keychain items (kSecAttrService).
    ///   - accessGroup: Value for kSecAttrAccessGroup. Must appear in the
    ///     binary's `keychain-access-groups` entitlement for the
    ///     data-protection Keychain to accept reads/writes. Pass nil to
    ///     omit the attribute entirely. Without a matching entitlement the
    ///     store returns OSStatus -34018 and callers should fall back to
    ///     FileClusterKeyStore.
    ///
    /// The SwiftLMTests.entitlements plist carries the bare access group
    /// `com.turboquant.cluster.test` which the test suite uses. The
    /// default value `com.turboquant.cluster` is a placeholder name for
    /// callers who embed their own matching entitlement; this open-source
    /// repo does not ship a signed production binary, so downstream
    /// packagers who codesign are responsible for choosing the group
    /// value (typically prefixed with their Apple Developer Team ID) and
    /// authoring the matching entitlements plist themselves.
    public init(service: String = "com.turboquant.cluster",
                accessGroup: String? = "com.turboquant.cluster") {
        self.service = service
        self.accessGroup = accessGroup
    }

    private func baseQuery(cluster: String) -> [String: Any] {
        var query: [String: Any] = [
            kSecClass as String:                 kSecClassGenericPassword,
            kSecAttrService as String:           service,
            kSecAttrAccount as String:           cluster,
            kSecUseDataProtectionKeychain as String: true,
        ]
        if let accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }
        return query
    }

    public func store(key: Data, for cluster: String) throws {
        // Remove any prior entry first. A missing item is fine; any other
        // failure is surfaced so callers see actual Keychain problems
        // rather than silently shadowed prior data.
        let deleteStatus = SecItemDelete(baseQuery(cluster: cluster) as CFDictionary)
        guard deleteStatus == errSecSuccess || deleteStatus == errSecItemNotFound else {
            throw ClusterKeyStoreError.unexpectedStatus(deleteStatus)
        }

        var addQuery = baseQuery(cluster: cluster)
        addQuery[kSecValueData as String] = key
        addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly

        let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            throw ClusterKeyStoreError.unexpectedStatus(addStatus)
        }
    }

    public func load(cluster: String) throws -> Data? {
        var query = baseQuery(cluster: cluster)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        switch status {
        case errSecSuccess:
            return result as? Data
        case errSecItemNotFound:
            return nil
        default:
            throw ClusterKeyStoreError.unexpectedStatus(status)
        }
    }

    public func delete(cluster: String) throws {
        let status = SecItemDelete(baseQuery(cluster: cluster) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw ClusterKeyStoreError.unexpectedStatus(status)
        }
    }

    /// Delete every entry under this store's service. Intended for test
    /// teardown; production code should use delete(cluster:) targeted at
    /// the specific cluster being removed.
    public func deleteAll() throws {
        var query: [String: Any] = [
            kSecClass as String:                 kSecClassGenericPassword,
            kSecAttrService as String:           service,
            kSecUseDataProtectionKeychain as String: true,
        ]
        if let accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw ClusterKeyStoreError.unexpectedStatus(status)
        }
    }
}
