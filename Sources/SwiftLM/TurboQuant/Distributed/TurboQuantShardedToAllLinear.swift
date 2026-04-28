// TurboQuant row-parallel linear layer. Holds the rank-local slice
// of a TurboQuant linear layer's compressed payload — packed primary
// indices and packed residual indices sliced along the input-dim
// axis, plus the per-row norms and Lloyd-Max codebooks (broadcast
// unchanged across ranks). `callAsFunction` forwards a rank-local
// input slice through the shard-aware TurboQuant kernel to produce a
// partial output of the full output shape, then `group.allSum`
// accumulates partials across ranks to yield the replicated full
// output.
//
// The layer wraps a `TurboQuantShardedLinear` Swift bridge, which
// dispatches into the shard-aware TQ kernel on the rank-local payload.
// Row-parallel semantics: the input dim is sliced across ranks
// (`localInFeatures == fullInFeatures / worldSize`) while the output
// dim is produced in full on every rank and summed by the collective.
// The C kernel still needs `fullInFeatures` explicitly to derive the
// correct `combined_scale`: the WHT factorisation requires the full
// pre-shard input dimension so per-shard rescale does not drift.
//
// Row-parallel sharding is only valid when the full input dim aligns
// on a group-sized block boundary — `fullInFeatures % (worldSize *
// blockSize) == 0`. This is the only configuration where rotation and
// per-block norm semantics compose cleanly across shards; misaligned
// splits produce silently wrong combined_scale values. The
// precondition in `init` enforces this invariant at construction time.
//
// At a size-1 group the `allSum` is a no-op and the single-rank
// partial IS the full answer.
//
// Payload ownership: this layer owns the internal
// `TurboQuantShardedLinear` which in turn retains the MLXArray tensor
// payloads (packed indices, norms, codebooks) as stored properties so
// MLX keeps the backing buffers alive for the lifetime of the C++
// handle.

import Foundation
import MLX
import MLXNN

public final class TurboQuantShardedToAllLinear: Module, TurboQuantShardedLayer {
    public let rankOutputDim: Int
    public let rankInputDim: Int
    public let group: DistributedGroup

    // Rank-local TQ kernel. Owns the opaque tq_linear_t handle and
    // retains the tensor payloads for the kernel's lifetime.
    private let kernel: TurboQuantShardedLinear

    /// Build a row-parallel TQ linear layer from the loader's
    /// rank-local tensor payloads. The packed-index tensors must
    /// already be sliced along the input-dim axis for this rank;
    /// norms and codebooks are broadcast unchanged (rotation is
    /// block-local and per-row state is identical across ranks — only
    /// the input-block bytes differ).
    ///
    /// Shape contract (per `turboquant_c.h`, row-parallel case):
    /// - `packedPrimary` / `packedResidual`: uint8 packed along a
    ///   row-major `[rankOutFeatures, localInFeatures * bits / 8]`
    ///   layout (residual may be empty when `residualBits == 0`).
    /// - `norms`: `[rankOutFeatures]` fp16/fp32, shared across ranks.
    /// - `primaryCodebook` / `residualCodebook`: shared across the
    ///   layer (not sharded) — the loader passes the full codebook to
    ///   every rank.
    ///
    /// Preconditions:
    /// - `fullInFeatures` must be divisible by `worldSize * blockSize`
    ///   so rotation / per-block norm semantics factorise cleanly
    ///   across shards.
    /// - `localInFeatures * worldSize == fullInFeatures`.
    ///
    /// `worldSize` defaults to `group.size`. Callers that drive shard
    /// selection from an explicit logical-rank override (e.g.
    /// in-process construction of multiple rank-local model instances
    /// against a singleton group) pass the logical world size so the
    /// preconditions validate against the layout being constructed
    /// rather than the physical group identity.
    public init(
        fullInFeatures: Int,
        rankOutFeatures: Int,
        localInFeatures: Int,
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
        group: DistributedGroup,
        worldSize: Int? = nil
    ) throws {
        let resolvedWorldSize = worldSize ?? group.size
        precondition(
            fullInFeatures % (resolvedWorldSize * blockSize) == 0,
            "row-parallel TQ sharding requires fullInFeatures divisible " +
            "by worldSize * blockSize; got " +
            "fullInFeatures=\(fullInFeatures), worldSize=\(resolvedWorldSize), " +
            "blockSize=\(blockSize)"
        )
        precondition(
            localInFeatures * resolvedWorldSize == fullInFeatures,
            "row-parallel localInFeatures must equal " +
            "fullInFeatures / worldSize; got localInFeatures=" +
            "\(localInFeatures), worldSize=\(resolvedWorldSize), " +
            "fullInFeatures=\(fullInFeatures)"
        )

        self.kernel = try TurboQuantShardedLinear(
            fullInFeatures: fullInFeatures,
            localInFeatures: localInFeatures,
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
        self.rankInputDim = localInFeatures
        self.group = group
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Row-parallel: x is (..., localInFeatures) — this rank has
        // already received its input-dim slice. The TQ kernel produces
        // a partial (..., rankOutFeatures) along the full output dim;
        // group.allSum accumulates partials across ranks to yield the
        // replicated full output. On a size-1 group allSum is an
        // identity op and the partial IS the full answer.
        let partial = kernel.callAsFunction(x)
        return group.allSum(partial)
    }
}
