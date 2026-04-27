import Foundation
import MLX

#if canImport(TurboQuantC)
import TurboQuantC
#endif

#if canImport(Cmlx)
import Cmlx
#endif

/// Swift bridge over the shard-aware TurboQuant C API (`tq_linear_*`).
///
/// Wraps an opaque `tq_linear_t` handle constructed from rank-local compressed
/// weight tensors: quantization indices (primary + optional residual),
/// per-row norms, and the two Lloyd-Max codebooks. The loader side
/// (`ShardAwareSafetensorsReader`) produces already-sliced byte
/// ranges for the rank; this wrapper is responsible only for handing those
/// ranges to the fused kernel via the C boundary and lifting the returned
/// activation back into Swift as an MLXArray.
///
/// Lifetime: the layer owns the C handle and releases it via
/// `tq_linear_free` on deinit. The tensor MLXArrays passed at construction
/// time are referenced by pointer by the underlying C++ layer — their
/// backing storage must remain live for as long as the layer is in use.
/// This class retains the MLXArray values as stored properties so the
/// MLX runtime keeps their buffers alive as long as `self` does.
///
/// When the TurboQuantC module is not linked (dylib absent at build time),
/// initialization fails with `Error.moduleUnavailable` so the rest of
/// SwiftLM continues to compile and run the non-TQ code paths.
public final class TurboQuantShardedLinear {

    // MARK: - Public shape metadata

    /// The layer's original (pre-shard) input-feature count. Matches
    /// `local_in_features` for column-parallel shards and
    /// `local_in_features * N_ranks` for row-parallel shards.
    public let fullInFeatures: Int

    /// This rank's input-feature count. For column-parallel layers this
    /// equals `fullInFeatures`; for row-parallel layers it is
    /// `fullInFeatures / N_ranks`.
    public let localInFeatures: Int

    /// This rank's output-feature count. For row-parallel layers this is
    /// the full output dim (each rank computes a partial along the input
    /// dim); for column-parallel layers it is `out_features / N_ranks`.
    public let rankOutFeatures: Int

    /// Quantization group size. Must match what the offline quantizer used
    /// and what the fused kernel assumes (typically 32 or 64).
    public let blockSize: Int

    // MARK: - Error surface

    public enum Error: Swift.Error {
        /// `tq_linear_create_shard` returned NULL. Likely causes: invalid
        /// dimensions, unsupported `primaryBits` value, or an allocation
        /// failure inside the C++ layer.
        case createFailed

        /// `tq_linear_forward` returned NULL at runtime. Surfaces as a
        /// fatal error from `callAsFunction` since the forward pass has
        /// no recoverable fallback — the inference graph cannot continue.
        case forwardReturnedNull

        /// The TurboQuantC module was not available at build time. Build
        /// the core dylib (`cmake --build build`) and re-link before
        /// constructing shard-aware layers.
        case moduleUnavailable
    }

    // MARK: - Stored tensor references

    // Retained so MLX keeps the underlying buffers alive while the C
    // layer holds raw-pointer views into them. These are never read
    // directly by Swift after init — their sole purpose is ownership.
    private let packedPrimaryRef: MLXArray
    private let packedResidualRef: MLXArray
    private let normsRef: MLXArray
    private let primaryCodebookRef: MLXArray
    private let residualCodebookRef: MLXArray

    #if canImport(TurboQuantC)
    private let handle: tq_linear_t
    #endif

    // MARK: - Initialization

    /// Build a rank-local TurboQuantLinear from the shard-aware loader's
    /// tensor slices. See `turboquant_c.h` for the exact shape contract
    /// expected of each tensor under column-parallel vs row-parallel
    /// split. This wrapper does not re-validate shapes; the C layer
    /// returns NULL on mismatch and we surface that as `createFailed`.
    public init(
        fullInFeatures: Int,
        localInFeatures: Int,
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
        blockSize: Int
    ) throws {
        self.fullInFeatures = fullInFeatures
        self.localInFeatures = localInFeatures
        self.rankOutFeatures = rankOutFeatures
        self.blockSize = blockSize

        self.packedPrimaryRef = packedPrimary
        self.packedResidualRef = packedResidual
        self.normsRef = norms
        self.primaryCodebookRef = primaryCodebook
        self.residualCodebookRef = residualCodebook

        #if canImport(TurboQuantC)
        // `tq_linear_create_shard` takes raw data pointers
        // (`const uint8_t*` for packed indices, `const float*` for
        // norms and codebooks) and copies them into MLX-owned
        // storage on construction — the caller's buffers can be
        // released as soon as the factory returns. MLXArray's
        // backing buffer is not directly addressable as host-side
        // bytes, so we materialize a contiguous fp32/u8 copy of each
        // tensor via `asData(access: .copy)` and hand the resulting
        // `Data` buffers to the C API within `withUnsafeBytes`
        // closures. This avoids the prior bug where the Swift side
        // passed the inner `mlx::core::array*` pointer and the C
        // impl dereferenced its metadata header as raw index bytes.
        //
        // `asData(access: .copy)` materializes the backing (triggers
        // graph evaluation) before copying so the returned bytes are
        // valid even when the MLXArray is a lazy slice of a parent.
        let primaryBytes = packedPrimary.asType(.uint8).asData(access: .copy).data
        let residualBytes = packedResidual.asType(.uint8).asData(access: .copy).data
        let normsBytes = norms.asType(.float32).asData(access: .copy).data
        let primaryCbBytes = primaryCodebook.asType(.float32).asData(access: .copy).data
        let residualCbBytes = residualCodebook.asType(.float32).asData(access: .copy).data

        let created: tq_linear_t? = primaryBytes.withUnsafeBytes { primaryPtr in
            residualBytes.withUnsafeBytes { residualPtr in
                normsBytes.withUnsafeBytes { normsPtr in
                    primaryCbBytes.withUnsafeBytes { primaryCbPtr in
                        residualCbBytes.withUnsafeBytes { residualCbPtr in
                            tq_linear_create_shard(
                                Int32(fullInFeatures),
                                Int32(localInFeatures),
                                Int32(rankOutFeatures),
                                Int32(primaryBits),
                                Int32(residualBits),
                                primaryPtr.baseAddress,
                                residualPtr.baseAddress,
                                normsPtr.baseAddress,
                                primaryCbPtr.baseAddress,
                                residualCbPtr.baseAddress,
                                seedPrimary,
                                seedResidual,
                                Int32(blockSize)
                            )
                        }
                    }
                }
            }
        }
        guard let h = created else {
            throw Error.createFailed
        }
        self.handle = h
        #else
        throw Error.moduleUnavailable
        #endif
    }

    deinit {
        #if canImport(TurboQuantC)
        tq_linear_free(handle)
        #endif
    }

    // MARK: - Forward

    /// Forward an activation through the rank-local TQ kernel.
    /// `input` must have shape `[batch, localInFeatures]` and any
    /// floating dtype (the C layer casts to float16 internally).
    /// Returns a new MLXArray of shape `[batch, rankOutFeatures]`
    /// owned by Swift — its deinit releases the underlying C++ array.
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        #if canImport(TurboQuantC)
        // Force the input graph to materialize before handing its
        // backing `mlx::core::array*` to the C kernel. The C++ layer
        // accesses the array's data directly via `data<T>()`, which
        // requires the graph to be evaluated; an unevaluated input
        // hands the kernel an empty buffer and silently produces
        // zeros (the kernel reads `input.ctx` directly via `data<T>()`).
        input.eval()
        let outputPtr = tq_linear_forward(
            handle,
            UnsafeRawPointer(input.ctx.ctx)
        )
        guard let outputPtr else {
            // Forward failure is unrecoverable mid-graph; match the
            // crash semantics of MLX ops that raise from the C++ layer.
            fatalError("tq_linear_forward returned null — kernel dispatch failed")
        }
        // `tq_linear_forward` documents its return value as a newly
        // allocated `mlx::core::array*`; wrapping it via
        // `mlx_array(ctx:)` and then `MLXArray.init(_:)` transfers
        // ownership to MLXArray, whose deinit calls
        // `mlx_array_free` (which in turn deletes the C++ array).
        // No `tq_array_free` is needed after this point.
        let wrapped = mlx_array(ctx: UnsafeMutableRawPointer(mutating: outputPtr))
        return MLXArray(wrapped)
        #else
        _ = input
        fatalError("TurboQuantC module unavailable — cannot forward through TurboQuantShardedLinear")
        #endif
    }
}
