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

/// Builder over the TurboQuantC cluster C API for memory-aware
/// pipeline-parallel layer assignment. Typical use: create with the
/// model's dimensions, add each discovered peer's memory, call plan(),
/// then read the layer range assigned to each rank.
///
/// The builder is single-shot: once plan() succeeds, the assignment is
/// fixed and no further nodes can be added. This mirrors the C API
/// contract, which guards against inconsistent state between the
/// registered node list and the computed ShardPlan.
public final class ClusterPlanBuilder {

    #if canImport(TurboQuantC)
    private let handle: tq_cluster_t
    private var planned = false

    public init?(numLayers: Int, numHeads: Int, headDim: Int) {
        guard let h = tq_cluster_create(Int32(numLayers), Int32(numHeads), Int32(headDim)) else {
            return nil
        }
        self.handle = h
    }

    deinit {
        tq_cluster_free(handle)
    }

    /// Register a discovered peer. Returns false if the hostname is empty,
    /// memory is non-positive, or plan() has already been called.
    public func addNode(hostname: String, memoryGB: Double) -> Bool {
        return hostname.withCString { cstr in
            tq_cluster_add_node(handle, cstr, memoryGB) == 0
        }
    }

    /// Compute the memory-aware pipeline-parallel assignment. Must be
    /// called exactly once; further calls return false.
    public func plan() -> Bool {
        let rc = tq_cluster_plan(handle)
        if rc == 0 { planned = true }
        return rc == 0
    }

    /// Number of nodes currently registered with the builder.
    public var nodeCount: Int {
        return Int(tq_cluster_node_count(handle))
    }

    /// Inclusive-exclusive layer range for the given rank, or nil if
    /// plan() has not been called or rank is out of range.
    public func layerRange(forRank rank: Int) -> Range<Int>? {
        guard planned else { return nil }
        let start = tq_cluster_get_layer_start(handle, Int32(rank))
        let end = tq_cluster_get_layer_end(handle, Int32(rank))
        guard start >= 0, end >= 0 else { return nil }
        return Int(start)..<Int(end)
    }
    #else
    /// Initializer returns nil when TurboQuantC is not available so callers
    /// detect the missing backend rather than operating on a stub handle.
    public init?(numLayers: Int, numHeads: Int, headDim: Int) {
        return nil
    }

    public func addNode(hostname: String, memoryGB: Double) -> Bool { return false }
    public func plan() -> Bool { return false }
    public var nodeCount: Int { return 0 }
    public func layerRange(forRank rank: Int) -> Range<Int>? { return nil }
    #endif
}
