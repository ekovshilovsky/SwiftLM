// Marker protocol for Distributed-aware TurboQuant linear layers.
// Types conforming to this protocol expose the rank-local output/input
// dim and the DistributedGroup they collectively participate in —
// giving tests and introspection a uniform view over both the
// column-parallel (TurboQuantAllToShardedLinear) and row-parallel
// (TurboQuantShardedToAllLinear) variants.

import MLX
import MLXNN

public protocol TurboQuantShardedLayer: UnaryLayer {
    /// This rank's slice of the output dimension. For column-parallel
    /// layers this is fullOutputDim / N; for row-parallel layers this
    /// is the full output dim (output is replicated after allSum).
    var rankOutputDim: Int { get }

    /// This rank's slice of the input dimension. For row-parallel
    /// layers this is fullInputDim / N; for column-parallel layers
    /// this is the full input dim (input is replicated).
    var rankInputDim: Int { get }

    /// The DistributedGroup across which the shard is split. At
    /// size-1 groups every collective is a no-op, making in-process
    /// testing of the math straightforward; at size-N groups the
    /// row-parallel variant invokes group.allSum on its partial
    /// output.
    var group: DistributedGroup { get }
}
