// Persistent storage for the cluster record a node belongs to. A node is
// in at most one cluster at a time, so the storage abstraction holds
// exactly one ClusterRecord — not a dictionary keyed by cluster name.
//
// The record bundles the cluster UUID (public, shared across all nodes
// in the same cluster, also used as the Argon2id salt) with the 32-byte
// master key (secret). The two values are always saved and loaded
// together: the key alone cannot be re-derived or verified without the
// salt, and losing the id after storing the key would leave the node
// unable to recognize the cluster it belongs to.

import Foundation
import Security

public enum ClusterKeyStoreError: Error, Equatable, CustomStringConvertible {
    case unexpectedStatus(OSStatus)
    case ioError(String)

    public var description: String {
        switch self {
        case .unexpectedStatus(let status):
            return "Keychain OSStatus \(status)"
        case .ioError(let msg):
            return "Cluster key store I/O error: \(msg)"
        }
    }
}

/// A node's cluster membership on disk. Two values bundled as one
/// because a node cannot participate in the cluster with either half
/// missing.
public struct ClusterRecord: Sendable, Equatable {
    /// 16-byte cluster UUID. Generated exactly once, by the coordinator,
    /// at cluster creation time. Every node that subsequently joins the
    /// same cluster receives and stores the SAME UUID — that is how all
    /// nodes in a cluster derive an identical master key from their
    /// shared passphrase: Argon2id(passphrase, sharedUUID) is the same
    /// 32 bytes on every node. The UUID is public (broadcast in Bonjour
    /// TXT so joining nodes can read it before the first handshake) and
    /// stable across key rotation, so it serves as the cluster's
    /// durable identifier.
    public let clusterId: Data
    /// 32-byte master key. Secret. Currently Argon2id(passphrase,
    /// clusterId); will become the coordinator-generated working key
    /// once cluster-level key rotation is wired up.
    public let key: Data

    public init(clusterId: Data, key: Data) {
        self.clusterId = clusterId
        self.key = key
    }
}

/// Abstraction so callers depend on a protocol rather than a concrete
/// backend. Implementations:
///
/// - `FileClusterKeyStore` (default, v1): writes to
///   `~/Library/Application Support/SwiftLM/cluster/`. Works for
///   unsigned binaries; no entitlements required.
/// - `InMemoryClusterKeyStore`: volatile, for tests. Wrapped in
///   `#if DEBUG` so it is absent from release binaries.
/// - `KeychainClusterKeyStore`: retained for signed distribution
///   builds that can carry the `keychain-access-groups` entitlement.
///   Not the default because `swift test` and open-source unsigned
///   distribution cannot use it without further signing work.
public protocol ClusterKeyStore: Sendable {
    /// Return the stored cluster record, or nil if this node has not
    /// joined a cluster (or has been removed from one).
    func load() throws -> ClusterRecord?

    /// Persist a cluster record, replacing any prior record. Callers
    /// typically invoke save exactly twice per node lifetime: once on
    /// first join, and again if the cluster key is rotated.
    func save(_ record: ClusterRecord) throws

    /// Remove any stored cluster record. A missing record is not an
    /// error so callers can use this as idempotent cleanup.
    func delete() throws
}

/// Thin abstraction over Security.framework's `SecItem*` C API. The
/// production implementation forwards directly to the system calls;
/// tests substitute an in-memory recording fake so the save / load /
/// delete logic in `KeychainClusterKeyStore` is exercisable under
/// plain `swift test` without the `keychain-access-groups`
/// entitlement (which macOS's data-protection Keychain requires and
/// which ad-hoc-signed test binaries cannot carry).
public protocol SecItemClient: Sendable {
    /// Mirrors `SecItemCopyMatching` for the single specific shape
    /// `KeychainClusterKeyStore` uses: `kSecReturnData = true`,
    /// `kSecMatchLimit = kSecMatchLimitOne`. Returns the status and,
    /// on `errSecSuccess`, the stored payload.
    func copyMatchingData(_ query: [String: Any]) -> (OSStatus, Data?)

    /// Mirrors `SecItemAdd`; the `kSecValueData` attribute in the
    /// dictionary supplies the payload.
    func add(_ attributes: [String: Any]) -> OSStatus

    /// Mirrors `SecItemDelete`.
    func delete(_ query: [String: Any]) -> OSStatus
}

/// Production `SecItemClient` that forwards to the live macOS
/// Security framework. Used by default when constructing a
/// `KeychainClusterKeyStore` without a test-supplied client.
public struct SystemSecItemClient: SecItemClient {
    public init() {}

    public func copyMatchingData(_ query: [String: Any]) -> (OSStatus, Data?) {
        var finalQuery = query
        finalQuery[kSecReturnData as String] = true
        finalQuery[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: AnyObject?
        let status = SecItemCopyMatching(finalQuery as CFDictionary, &result)
        return (status, result as? Data)
    }

    public func add(_ attributes: [String: Any]) -> OSStatus {
        return SecItemAdd(attributes as CFDictionary, nil)
    }

    public func delete(_ query: [String: Any]) -> OSStatus {
        return SecItemDelete(query as CFDictionary)
    }
}

/// macOS Keychain-backed implementation using the data-protection
/// Keychain. Requires the calling binary to carry a matching
/// `keychain-access-groups` entitlement; without one, every call
/// returns `errSecMissingEntitlement` (OSStatus -34018). Under plain
/// `swift test` the entitlement is unavailable (ad-hoc signing
/// cannot carry restricted entitlements on modern macOS), so
/// production callers should fall back to `FileClusterKeyStore` in
/// unentitled environments.
///
/// Two Keychain items per cluster record: one account for the cluster
/// id, one for the working key, both under a shared service. A signed
/// distribution build would override the access group with whatever
/// team-prefixed value matches its own entitlement.
///
/// The `SecItemClient` dependency is injected so tests can substitute
/// a recording fake and verify the query attributes without requiring
/// any entitlement.
public struct KeychainClusterKeyStore: ClusterKeyStore {
    public let service: String
    public let accessGroup: String?
    private let secItem: SecItemClient

    private static let idAccount = "cluster.id"
    private static let keyAccount = "cluster.key"

    public init(service: String = "com.turboquant.cluster",
                accessGroup: String? = "com.turboquant.cluster",
                secItemClient: SecItemClient = SystemSecItemClient()) {
        self.service = service
        self.accessGroup = accessGroup
        self.secItem = secItemClient
    }

    private func baseQuery(account: String) -> [String: Any] {
        var query: [String: Any] = [
            kSecClass as String:                 kSecClassGenericPassword,
            kSecAttrService as String:           service,
            kSecAttrAccount as String:           account,
            kSecUseDataProtectionKeychain as String: true,
        ]
        if let accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }
        return query
    }

    public func load() throws -> ClusterRecord? {
        guard let id = try loadItem(account: Self.idAccount) else { return nil }
        guard let key = try loadItem(account: Self.keyAccount) else {
            // Partial state: id present but key missing. Treat as no
            // record so the caller re-runs the join flow rather than
            // trying to use an unusable half-record.
            return nil
        }
        return ClusterRecord(clusterId: id, key: key)
    }

    public func save(_ record: ClusterRecord) throws {
        try saveItem(account: Self.idAccount, data: record.clusterId)
        try saveItem(account: Self.keyAccount, data: record.key)
    }

    public func delete() throws {
        try deleteItem(account: Self.idAccount)
        try deleteItem(account: Self.keyAccount)
    }

    private func loadItem(account: String) throws -> Data? {
        let (status, data) = secItem.copyMatchingData(baseQuery(account: account))
        switch status {
        case errSecSuccess:
            return data
        case errSecItemNotFound:
            return nil
        default:
            throw ClusterKeyStoreError.unexpectedStatus(status)
        }
    }

    private func saveItem(account: String, data: Data) throws {
        // delete-then-add is clearer than add-with-fallback-to-update,
        // and saves happen at most once per cluster join — not a hot
        // path.
        let deleteStatus = secItem.delete(baseQuery(account: account))
        guard deleteStatus == errSecSuccess || deleteStatus == errSecItemNotFound else {
            throw ClusterKeyStoreError.unexpectedStatus(deleteStatus)
        }

        var addQuery = baseQuery(account: account)
        addQuery[kSecValueData as String] = data
        addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly

        let addStatus = secItem.add(addQuery)
        guard addStatus == errSecSuccess else {
            throw ClusterKeyStoreError.unexpectedStatus(addStatus)
        }
    }

    private func deleteItem(account: String) throws {
        let status = secItem.delete(baseQuery(account: account))
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw ClusterKeyStoreError.unexpectedStatus(status)
        }
    }
}
