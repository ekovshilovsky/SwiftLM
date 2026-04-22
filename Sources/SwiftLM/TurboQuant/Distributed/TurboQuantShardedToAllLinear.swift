// TurboQuant row-parallel linear layer. Holds a rank-local slice of
// the input dimension of a TQ linear layer's weight. Input to
// callAsFunction must already be sharded along the input dim (produced
// by an upstream column-parallel layer). Per-rank matmul produces a
// partial output of the full output shape; group.allSum accumulates
// partials across ranks to yield the replicated full output.
//
// At a size-1 group the allSum is a no-op and the partial IS the full
// answer. In-process testing exploits this: compute two partials on a
// size-1 group with the two rank-slice weights, manually sum the
// partials, and compare to a single-rank full matmul. This mirrors
// what a size-2 group's allSum would produce.
//
// Phase 3 Task 8 uses a stub MLX.matmul forward path; Phase 3 Task 9
// swaps in TurboQuant's kernel so the layer operates on the compressed
// TQ representation (codebook + indices + rotation) rather than an
// fp16 dequantized weight.

import Foundation
import MLX
import MLXNN

public final class TurboQuantShardedToAllLinear: Module, TurboQuantShardedLayer {
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
        // Row-parallel: x is (..., rankInputDim). Per-rank matmul
        // against W^T where W is (rankOutputDim, rankInputDim) produces
        // a partial (..., rankOutputDim). allSum accumulates partials
        // across ranks to yield the full replicated output. On a
        // size-1 group allSum is an identity op.
        let partial = MLX.matmul(x, rankWeight.transposed(-1, -2))
        return group.allSum(partial)
    }
}
