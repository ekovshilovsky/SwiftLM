// TurboQuant column-parallel linear layer. Holds the rank-local slice
// of a TurboQuant linear layer's compressed payload — packed primary
// indices, packed residual indices, per-row norms, the two Lloyd-Max
// codebooks, and the rotation seeds — covering this rank's slice of
// the output dimension. `callAsFunction` forwards the full-dim input
// through the shard-aware TurboQuant kernel and produces a rank-local
// sized activation. No collective is issued: the output stays sharded
// along the output dim for downstream consumers (the row-parallel
// layer at the next boundary allSums its own partial).
//
// Task 9a.4 wires this layer to the real TQ kernel via the
// `TurboQuantShardedLinear` Swift bridge (Task 9a.3), replacing the
// Task 7 MLX.matmul stub that operated on a pre-dequantized fp16
// weight. Column-parallel semantics mean `localInFeatures` equals
// `fullInFeatures` (the input is replicated across ranks); the C
// layer uses `fullInFeatures` to compute the correct combined_scale
// so per-shard rescale does not drift.
//
// Payload ownership: this layer owns the internal
// `TurboQuantShardedLinear` which in turn retains the MLXArray tensor
// payloads (packed indices, norms, codebooks) as stored properties so
// MLX keeps the backing buffers alive for the lifetime of the C++
// handle.

import Foundation
import MLX
import MLXNN

public final class TurboQuantAllToShardedLinear: Module, TurboQuantShardedLayer {
    public let rankOutputDim: Int
    public let rankInputDim: Int
    public let group: DistributedGroup

    // Rank-local TQ kernel. Owns the opaque tq_linear_t handle and
    // retains the tensor payloads for the kernel's lifetime.
    private let kernel: TurboQuantShardedLinear

    /// Build a column-parallel TQ linear layer from the loader's
    /// rank-local tensor payloads. The tensors must already be sliced
    /// along the output-dim axis for this rank; `fullInFeatures` is
    /// used by the C kernel to reproduce the whole-weight
    /// combined_scale (so numerical correctness does not depend on the
    /// rank count).
    ///
    /// Shape contract (per `turboquant_c.h`, column-parallel case):
    /// - `packedPrimary`  / `packedResidual`: uint8 packed along a
    ///   row-major `[rankOutFeatures, fullInFeatures * bits / 8]`
    ///   layout (residual may be empty when `residualBits == 0`).
    /// - `norms`:           `[rankOutFeatures]` fp16/fp32.
    /// - `primaryCodebook`  / `residualCodebook`: shared across the
    ///   layer (not sharded) — the loader passes the full codebook to
    ///   every rank.
    public init(
        fullInFeatures: Int,
        rankOutFeatures: Int,
        primaryBits: Int,
        residualBits: Int,
        packedPrimary: MLXArray,
        packedResidual: MLXArray,
        norms: MLXArray,
        primaryCodebook: MLXArray,
        residualCodebook: MLXArray,
        seedPrimary: UInt32,
        seedResidual: UInt32,
        blockSize: Int,
        group: DistributedGroup
    ) throws {
        // Column-parallel: the input dim is fully replicated across
        // ranks, so local_in_features == full_in_features. The C
        // kernel still needs full_in_features explicitly to derive
        // combined_scale correctly.
        self.kernel = try TurboQuantShardedLinear(
            fullInFeatures: fullInFeatures,
            localInFeatures: fullInFeatures,
            rankOutFeatures: rankOutFeatures,
            primaryBits: primaryBits,
            residualBits: residualBits,
            packedPrimary: packedPrimary,
            packedResidual: packedResidual,
            norms: norms,
            primaryCodebook: primaryCodebook,
            residualCodebook: residualCodebook,
            seedPrimary: seedPrimary,
            seedResidual: seedResidual,
            blockSize: blockSize
        )
        self.rankOutputDim = rankOutFeatures
        self.rankInputDim = fullInFeatures
        self.group = group
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Column-parallel: x is the full-dim input, shape
        // (..., fullInFeatures). The TQ kernel produces
        // (..., rankOutFeatures) along this rank's output-dim slice.
        // No collective — downstream layers consume the shard directly.
        return kernel.callAsFunction(x)
    }
}
