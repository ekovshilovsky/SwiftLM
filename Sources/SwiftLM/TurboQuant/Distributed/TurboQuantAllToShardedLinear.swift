// TurboQuant column-parallel linear layer. Holds a rank-local slice of
// the output dimension of a TQ linear layer's weight. callAsFunction
// performs matmul against the full-dim input and produces a rank-local
// sized output — no collective operation, the output stays sharded for
// downstream consumers (row-parallel layers will allSum at their
// boundary).
//
// Phase 3 Task 7 stores the rank-local weight as a plain MLXArray and
// uses MLX.matmul for the forward pass. Phase 3 Task 9 swaps the
// matmul path for TurboQuant's kernel (which understands TQ codebook
// + indices + rotation packing) so the layer operates on the actual
// compressed weight format rather than requiring an fp16 dequant.

import Foundation
import MLX
import MLXNN

public final class TurboQuantAllToShardedLinear: Module, TurboQuantShardedLayer {
    public let rankOutputDim: Int
    public let rankInputDim: Int
    public let group: DistributedGroup

    // Rank-local weight. In Phase 3 this is fp16 (matmul-ready) until
    // Task 9 wires the TQ kernel; after Task 9 this storage moves to
    // codebook + indices + rotation held by the layer.
    private let rankWeight: MLXArray

    public init(
        rankWeight: MLXArray,
        rankOutputDim: Int,
        rankInputDim: Int,
        group: DistributedGroup
    ) {
        self.rankWeight = rankWeight
        self.rankOutputDim = rankOutputDim
        self.rankInputDim = rankInputDim
        self.group = group
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Column-parallel: x @ W^T where W is (rankOutputDim, rankInputDim).
        // Result is (..., rankOutputDim) sharded along output dim. No
        // collective — downstream layers use the shard directly.
        return MLX.matmul(x, rankWeight.transposed(-1, -2))
    }
}
