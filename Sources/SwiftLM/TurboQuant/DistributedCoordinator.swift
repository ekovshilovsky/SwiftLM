import Foundation

#if canImport(TurboQuantC)
import TurboQuantC
#endif

/// Swift wrapper for TQDistributedCoordinator C API.
/// Manages multi-Mac cluster topology, shard planning, and distributed forward pass.
///
/// Uses conditional compilation so SwiftLM builds cleanly whether or not
/// the TurboQuantC library is linked. When unavailable, all initializers
/// return nil and properties return single-node defaults.
public final class DistributedCoordinator {

    /// Distributed backend options.
    public enum Backend: String {
        case jaccl
        case ring
        case auto
    }

    /// Shard strategy options.
    public enum ShardStrategy: String {
        case pipeline
        case tensor
        case auto
    }

    #if canImport(TurboQuantC)
    private let handle: tq_coordinator_t

    private init(handle: tq_coordinator_t) {
        self.handle = handle
    }

    deinit {
        tq_distributed_free(handle)
    }

    /// Initialize distributed coordinator from a hostfile.
    /// Returns nil if initialization fails (e.g., nodes unreachable).
    public static func initialize(
        hostfile: String,
        backend: Backend = .auto
    ) -> DistributedCoordinator? {
        guard let h = tq_distributed_init(hostfile, backend.rawValue) else {
            return nil
        }
        return DistributedCoordinator(handle: h)
    }

    /// Initialize for single-node inference (no hostfile).
    public static func initializeLocal() -> DistributedCoordinator? {
        guard let h = tq_distributed_init_local() else {
            return nil
        }
        return DistributedCoordinator(handle: h)
    }

    /// This node's rank in the cluster.
    public var rank: Int {
        return Int(tq_distributed_rank(handle))
    }

    /// Total number of nodes in the cluster.
    public var worldSize: Int {
        return Int(tq_distributed_world_size(handle))
    }
    #else
    /// Initialize distributed coordinator from a hostfile.
    /// Returns nil when TurboQuantC is not available.
    public static func initialize(
        hostfile: String,
        backend: Backend = .auto
    ) -> DistributedCoordinator? {
        return nil
    }

    /// Initialize for single-node inference (no hostfile).
    /// Returns nil when TurboQuantC is not available.
    public static func initializeLocal() -> DistributedCoordinator? {
        return nil
    }

    /// This node's rank in the cluster.
    public var rank: Int { return 0 }

    /// Total number of nodes in the cluster.
    public var worldSize: Int { return 1 }
    #endif

    /// Whether this node is the coordinator (rank 0).
    public var isCoordinator: Bool {
        return rank == 0
    }
}
