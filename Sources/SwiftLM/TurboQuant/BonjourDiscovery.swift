// Bonjour discovery metadata. Every TurboQuant node advertises a DiscoveryInfo
// record over _turboquant._tcp so peers can, before any TCP connection is
// opened:
//
//   - identify which cluster a peer belongs to via an 8-hex discovery hash
//     derived from the cluster master key. The hash is opaque: peers who
//     do not hold the passphrase see an unintelligible hex value rather
//     than a user-chosen cluster name, so two clusters with different
//     passphrases are mutually invisible even on the same network.
//   - obtain the cluster's 16-byte UUID. The UUID doubles as the Argon2id
//     salt, so a joining node can derive the same master key from its
//     passphrase before any handshake messages are exchanged.
//   - filter incompatible models and runtime versions.
//   - estimate sharding capacity from advertised memory + RDMA state.
//
// Swift property names describe the stored value rather than the TXT
// key (e.g. `clusterHash` for the TXT `cluster` field) so callers are
// not misled into thinking `cluster` holds a user-facing name.

import Foundation

/// Role advertised in the Bonjour TXT record. Every node reports exactly
/// one of these values so peers can distinguish coordinators, workers
/// already committed to a cluster, and nodes still looking for one.
public enum DiscoveryRole: String, Sendable {
    case discovering
    case coordinator
    case worker
}

/// RDMA capability reported by each node. Three-state rather than boolean
/// because the triage value matters: `disabled` means the hardware is
/// present but the user has not run `rdma_ctl enable` in recovery mode
/// (user-fixable); `unsupported` means the hardware cannot do RDMA at all
/// (not fixable). Collapsing these to a single "no RDMA" state would hide
/// the fix path from the user.
public enum RdmaCapability: String, Sendable {
    case available
    case disabled
    case unsupported
}

/// Metadata advertised and discovered via Bonjour TXT records. Instances
/// are value-typed and immutable; construct a new record when advertised
/// state changes rather than mutating in place.
public struct DiscoveryInfo: Sendable, Equatable {
    public let model: String
    public let memoryGB: Int
    public let role: DiscoveryRole
    /// 8-hex-character discovery hash derived from the cluster master
    /// key, or nil when no cluster has been joined. Peers with
    /// mismatching hashes are different clusters and ignore each other.
    /// Encoded as the literal "none" in the TXT record when nil so the
    /// absence of a cluster is advertised positively rather than by key
    /// omission (which would be ambiguous with a malformed record).
    public let clusterHash: String?
    /// Cluster UUID. Doubles as the Argon2id salt so a joining node can
    /// derive the cluster master key from its passphrase before the first
    /// handshake. Survives key rotation; the stable identifier of the
    /// cluster. Nil when no cluster has been joined.
    public let clusterId: String?
    public let version: String
    public let rdma: RdmaCapability
    /// Human-readable hostname for UI display. Never used for routing or
    /// authentication; only for user-facing cluster status output.
    public let name: String

    public init(model: String,
                memoryGB: Int,
                role: DiscoveryRole,
                clusterHash: String?,
                clusterId: String?,
                version: String,
                rdma: RdmaCapability,
                name: String) {
        self.model = model
        self.memoryGB = memoryGB
        self.role = role
        self.clusterHash = clusterHash
        self.clusterId = clusterId
        self.version = version
        self.rdma = rdma
        self.name = name
    }

    /// Decode a TXT record dictionary into a DiscoveryInfo. Returns nil
    /// when any required key is missing or malformed. The `cluster` /
    /// `id` fields are permitted to be absent or set to the literal
    /// "none" because nodes in the discovering role have not yet joined
    /// a cluster.
    public init?(fromTxtRecord txt: [String: String]) {
        guard let model = txt["model"],
              let memStr = txt["memory"], let mem = Int(memStr),
              let roleStr = txt["role"], let role = DiscoveryRole(rawValue: roleStr),
              let version = txt["version"],
              let rdmaStr = txt["rdma"], let rdma = RdmaCapability(rawValue: rdmaStr),
              let name = txt["name"]
        else { return nil }

        self.model = model
        self.memoryGB = mem
        self.role = role
        self.clusterHash = Self.nilIfAbsentOrNone(txt["cluster"])
        self.clusterId = Self.nilIfAbsentOrNone(txt["id"])
        self.version = version
        self.rdma = rdma
        self.name = name
    }

    /// Encode this DiscoveryInfo for advertisement as a Bonjour TXT record.
    /// The `cluster` and `id` keys are emitted as the literal "none" when
    /// no cluster is joined so a node in discovering state is distinguishable
    /// from a malformed record with missing keys.
    public func toTxtRecord() -> [String: String] {
        return [
            "model": model,
            "memory": "\(memoryGB)",
            "role": role.rawValue,
            "cluster": clusterHash ?? "none",
            "id": clusterId ?? "none",
            "version": version,
            "rdma": rdma.rawValue,
            "name": name,
        ]
    }

    /// Check whether a discovered peer can participate in the same
    /// cluster as the caller. Model and version must match; the cluster
    /// hash match is a separate check performed by the discovery client
    /// against its own locally-derived hash.
    public func isCompatible(with model: String, version: String) -> Bool {
        return self.model == model && self.version == version
    }

    private static func nilIfAbsentOrNone(_ value: String?) -> String? {
        guard let value, value != "none" else { return nil }
        return value
    }
}
