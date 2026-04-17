// Bonjour discovery metadata. Every TurboQuant node advertises a DiscoveryInfo
// record over _turboquant._tcp so peers can filter incompatible models, match
// cluster names, and estimate sharding capacity before any TCP connection is
// opened. The network transport layer is scheduled for later tasks; this file
// currently owns only the TXT-record data model and compatibility rules.

import Foundation

/// Status advertised in the Bonjour TXT record. Every node in the cluster
/// reports exactly one of these values so peers can distinguish coordinators,
/// workers already committed to a cluster, and nodes still looking for one.
public enum DiscoveryStatus: String, Sendable {
    case discovering
    case coordinator
    case worker
}

/// RDMA capability reported by each node. Consumed by the shard planner to
/// prefer low-latency transports when available; nodes without RDMA still
/// participate over TCP.
public enum RdmaCapability: String, Sendable {
    case available
    case disabled
    case unsupported
}

/// Metadata advertised and discovered via Bonjour TXT records. Instances are
/// value-typed and immutable; construct a new record when advertised state
/// changes rather than mutating in place.
public struct DiscoveryInfo: Sendable, Equatable {
    public let model: String
    public let memoryGB: Int
    public let status: DiscoveryStatus
    public let cluster: String?
    public let version: String
    public let rdma: RdmaCapability

    public init(model: String,
                memoryGB: Int,
                status: DiscoveryStatus,
                cluster: String?,
                version: String,
                rdma: RdmaCapability) {
        self.model = model
        self.memoryGB = memoryGB
        self.status = status
        self.cluster = cluster
        self.version = version
        self.rdma = rdma
    }

    /// Decode a TXT record dictionary into a DiscoveryInfo. Returns nil when
    /// any required key is missing or malformed; the optional cluster key is
    /// permitted to be absent because nodes in the `discovering` state have
    /// not yet joined a cluster.
    public init?(fromTxtRecord txt: [String: String]) {
        guard let model = txt["model"],
              let memStr = txt["memory_gb"], let mem = Int(memStr),
              let statusStr = txt["status"], let status = DiscoveryStatus(rawValue: statusStr),
              let version = txt["version"],
              let rdmaStr = txt["rdma"], let rdma = RdmaCapability(rawValue: rdmaStr)
        else { return nil }

        self.model = model
        self.memoryGB = mem
        self.status = status
        self.cluster = txt["cluster"]
        self.version = version
        self.rdma = rdma
    }

    /// Encode this DiscoveryInfo for advertisement as a Bonjour TXT record.
    /// The cluster key is omitted when no cluster has been joined so the
    /// absence in the record is meaningful to peers during discovery.
    public func toTxtRecord() -> [String: String] {
        var txt: [String: String] = [
            "model": model,
            "memory_gb": "\(memoryGB)",
            "status": status.rawValue,
            "version": version,
            "rdma": rdma.rawValue,
        ]
        if let cluster {
            txt["cluster"] = cluster
        }
        return txt
    }

    /// Check whether a discovered peer can participate in the same cluster
    /// as the caller. Peers must run the same model identifier and the same
    /// runtime version; mismatches are silently filtered during cluster
    /// formation so that a Qwen node never joins a Llama cluster.
    public func isCompatible(with model: String, version: String) -> Bool {
        return self.model == model && self.version == version
    }
}
