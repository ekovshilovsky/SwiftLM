// Shared TurboQuant layer loader. Decodes a TQ-packed linear layer's
// companion safetensors tensors (packed primary indices, packed
// residual indices, per-row norms, seeds + block size, and the
// codebooks shipped alongside it) and returns them as MLXArrays
// dimensioned for either the whole-weight reference path or a
// rank-local sharded path.
//
// Background. The shard-metadata sidecar (`tq_shard_metadata.json`)
// only enumerates the public `<layer>.weight` entries. The TQ kernel
// also needs the per-layer companion tensors that the offline
// quantizer wrote into the same safetensors file. This loader parses
// the safetensors header itself to locate those companion tensors
// and produces an `MLXArray`-typed payload that
// `TurboQuantShardedLinear`, `TurboQuantAllToShardedLinear`, and
// `TurboQuantShardedToAllLinear` all consume.
//
// Two entry points:
//
//   - `loadFullTQLayerPayload(...)` — returns the full, unsliced
//     payload. Used by tests that simulate sharding in-process by
//     manually slicing the result, and by debug paths that want the
//     whole-weight oracle.
//
//   - `loadShardedTQLayerPayload(...)` — applies rank-local slicing
//     based on a `Role` (column-parallel slices output rows of the
//     packed indices and the per-row norms; row-parallel slices the
//     packed-index axis-1 byte range and leaves norms and codebooks
//     replicated). Returns a payload sized for the constructor of
//     either sharded layer wrapper.
//
// Why a shared loader. Before this file landed, the safetensors
// header parser, MLXArray-typed companion-tensor readers, and the
// TQLayerPayload struct all lived inside the
// `TurboQuantShardedLinearEndToEndTests` test file. `DistributedQwenModel`
// needs the same loading machinery for every TQ layer in the model,
// so duplicating the code into a non-test target would have produced
// two divergent code paths to maintain. Lifting it here gives both
// the tests and the production model loader a single owning module.

import Foundation
import MLX

// MARK: - Safetensors header

/// Per-tensor entry parsed out of a safetensors file's JSON header.
/// The header records each tensor's element type, shape, and the
/// half-open `[data_offsets[0], data_offsets[1])` byte range relative
/// to the start of the file's data section. This struct exposes the
/// resolved absolute byte offset in the file for direct reads.
public struct SafetensorsHeaderEntry: Sendable {
    public let dtype: String
    public let shape: [Int]
    /// Absolute offset from the start of the file (header length + 8
    /// + relative data offset).
    public let absoluteOffset: Int
    public let length: Int

    public init(dtype: String, shape: [Int], absoluteOffset: Int, length: Int) {
        self.dtype = dtype
        self.shape = shape
        self.absoluteOffset = absoluteOffset
        self.length = length
    }
}

public enum TQLayerLoaderError: Error, CustomStringConvertible {
    case headerLengthTruncated
    case headerBodyTruncated
    case headerNotJSONObject
    case missingTensor(String)
    case unexpectedDType(name: String, expected: String, got: String)
    case seedsTooSmall(name: String, count: Int)
    case sidecarMissingLayerWeight(String)
    case nonDivisibleShardDim(name: String, dim: Int, worldSize: Int)
    case unsupportedShardRole(String)

    public var description: String {
        switch self {
        case .headerLengthTruncated:
            return "safetensors header length field truncated"
        case .headerBodyTruncated:
            return "safetensors header body truncated"
        case .headerNotJSONObject:
            return "safetensors header is not a JSON object"
        case .missingTensor(let n):
            return "safetensors file missing tensor '\(n)'"
        case .unexpectedDType(let n, let e, let g):
            return "tensor '\(n)' expected dtype \(e), got \(g)"
        case .seedsTooSmall(let n, let c):
            return "tensor '\(n)' must have at least 3 elements (got \(c))"
        case .sidecarMissingLayerWeight(let n):
            return "shard metadata missing weight entry for '\(n)'"
        case .nonDivisibleShardDim(let n, let d, let w):
            return "tensor '\(n)' shard dimension \(d) is not divisible by world size \(w)"
        case .unsupportedShardRole(let r):
            return "unsupported shard role '\(r)' for TQ layer loading"
        }
    }
}

/// Parse the safetensors header of `fileURL` and return a map from
/// tensor name to its header entry. The 8-byte little-endian header
/// length prefix is followed by a UTF-8 JSON object; this routine
/// reads both and resolves each tensor's relative `data_offsets`
/// pair into an absolute byte offset in the file.
public func parseSafetensorsHeader(
    fileURL: URL
) throws -> [String: SafetensorsHeaderEntry] {
    let handle = try FileHandle(forReadingFrom: fileURL)
    defer { try? handle.close() }

    guard let lenBytes = try handle.read(upToCount: 8),
          lenBytes.count == 8
    else {
        throw TQLayerLoaderError.headerLengthTruncated
    }
    let headerLen = lenBytes.withUnsafeBytes { ptr -> UInt64 in
        ptr.load(as: UInt64.self)
    }
    guard let headerData = try handle.read(upToCount: Int(headerLen)),
          headerData.count == Int(headerLen)
    else {
        throw TQLayerLoaderError.headerBodyTruncated
    }
    let dataStart = 8 + Int(headerLen)

    let json = try JSONSerialization.jsonObject(with: headerData, options: [])
    guard let dict = json as? [String: Any] else {
        throw TQLayerLoaderError.headerNotJSONObject
    }

    var out: [String: SafetensorsHeaderEntry] = [:]
    out.reserveCapacity(dict.count)
    for (name, value) in dict {
        if name == "__metadata__" { continue }
        guard let entry = value as? [String: Any],
              let dtype = entry["dtype"] as? String,
              let shapeAny = entry["shape"] as? [Any],
              let offsets = entry["data_offsets"] as? [Any],
              offsets.count == 2
        else { continue }
        let shape = shapeAny.compactMap { ($0 as? NSNumber)?.intValue }
        guard shape.count == shapeAny.count else { continue }
        let startRel = (offsets[0] as? NSNumber)?.intValue ?? -1
        let endRel = (offsets[1] as? NSNumber)?.intValue ?? -1
        guard startRel >= 0, endRel >= startRel else { continue }
        out[name] = SafetensorsHeaderEntry(
            dtype: dtype,
            shape: shape,
            absoluteOffset: dataStart + startRel,
            length: endRel - startRel
        )
    }
    return out
}

// MARK: - Companion-tensor readers

/// Read a U8 companion tensor (packed primary / packed residual) and
/// wrap it in an MLXArray with the recorded shape.
public func loadUInt8MLXArray(
    fileURL: URL, entry: SafetensorsHeaderEntry, name: String
) throws -> MLXArray {
    guard entry.dtype == "U8" else {
        throw TQLayerLoaderError.unexpectedDType(
            name: name, expected: "U8", got: entry.dtype)
    }
    let reader = ShardAwareSafetensorsReader(fileURL: fileURL)
    let range = ShardOffsetCalculator.SliceRange(
        offset: entry.absoluteOffset,
        length: entry.length,
        strategy: .replicated,
        rankOutputDim: nil,
        rankInputDim: nil
    )
    let data = try reader.readContiguousRange(range)
    return MLXArray(data, entry.shape, type: UInt8.self)
}

/// Read an F32 companion tensor (norms / codebook) and wrap it in an
/// MLXArray with the recorded shape.
public func loadFloat32MLXArray(
    fileURL: URL, entry: SafetensorsHeaderEntry, name: String
) throws -> MLXArray {
    guard entry.dtype == "F32" else {
        throw TQLayerLoaderError.unexpectedDType(
            name: name, expected: "F32", got: entry.dtype)
    }
    let reader = ShardAwareSafetensorsReader(fileURL: fileURL)
    let range = ShardOffsetCalculator.SliceRange(
        offset: entry.absoluteOffset,
        length: entry.length,
        strategy: .replicated,
        rankOutputDim: nil,
        rankInputDim: nil
    )
    let data = try reader.readContiguousRange(range)
    return MLXArray(data, entry.shape, dtype: DType.float32)
}

/// Read a U32 seeds tensor and return the raw element array.
public func loadUInt32Array(
    fileURL: URL, entry: SafetensorsHeaderEntry, name: String
) throws -> [UInt32] {
    guard entry.dtype == "U32" else {
        throw TQLayerLoaderError.unexpectedDType(
            name: name, expected: "U32", got: entry.dtype)
    }
    let reader = ShardAwareSafetensorsReader(fileURL: fileURL)
    let range = ShardOffsetCalculator.SliceRange(
        offset: entry.absoluteOffset,
        length: entry.length,
        strategy: .replicated,
        rankOutputDim: nil,
        rankInputDim: nil
    )
    let data = try reader.readContiguousRange(range)
    let n = data.count / MemoryLayout<UInt32>.size
    return data.withUnsafeBytes { raw -> [UInt32] in
        let bound = raw.bindMemory(to: UInt32.self)
        return Array(bound.prefix(n))
    }
}

// MARK: - TQ layer payload

/// Composite payload for a single TurboQuant linear layer: the
/// packed-index tensors, per-row norms, the two Lloyd-Max codebooks,
/// the rotation seeds, and the inferred input/output dimensions.
/// Sized either as the whole layer (`loadFullTQLayerPayload`) or
/// pre-sliced for a rank (`loadShardedTQLayerPayload`).
///
/// Not `Sendable` because `MLXArray` is not Sendable. The payload is
/// passed straight into a layer constructor on the loading actor and
/// is not intended to cross isolation boundaries.
public struct TQLayerPayload {
    /// `[outFeatures, inFeatures * primaryBits / 8]` U8.
    public let packedPrimary: MLXArray
    /// `[outFeatures, inFeatures * residualBits / 8]` U8 (may be empty if residualBits == 0).
    public let packedResidual: MLXArray
    /// `[outFeatures]` F32 — per-output-row norm correction.
    public let norms: MLXArray
    /// `[2^primaryBits]` F32 — Lloyd-Max codebook. Replicated across ranks.
    public let primaryCodebook: MLXArray
    /// `[2^residualBits]` F32 — Lloyd-Max codebook. Replicated across ranks.
    public let residualCodebook: MLXArray

    public let seedPrimary: UInt32
    public let seedResidual: UInt32
    public let blockSize: Int

    /// Layer output dim (full or rank-local depending on caller).
    public let outFeatures: Int
    /// Layer input dim (full or rank-local depending on caller).
    public let inFeatures: Int

    public init(
        packedPrimary: MLXArray,
        packedResidual: MLXArray,
        norms: MLXArray,
        primaryCodebook: MLXArray,
        residualCodebook: MLXArray,
        seedPrimary: UInt32,
        seedResidual: UInt32,
        blockSize: Int,
        outFeatures: Int,
        inFeatures: Int
    ) {
        self.packedPrimary = packedPrimary
        self.packedResidual = packedResidual
        self.norms = norms
        self.primaryCodebook = primaryCodebook
        self.residualCodebook = residualCodebook
        self.seedPrimary = seedPrimary
        self.seedResidual = seedResidual
        self.blockSize = blockSize
        self.outFeatures = outFeatures
        self.inFeatures = inFeatures
    }
}

/// Force MLX to materialise the lazy graph backing each provided
/// array so subsequent reads see contiguous device storage. Wraps
/// `MLX.eval(_:)` to keep call sites declarative.
@inline(__always)
private func materialize(_ arrays: MLXArray...) {
    MLX.eval(arrays)
}

/// Load the full, unsliced TurboQuant payload for `layerName` from
/// the model directory. The sidecar resolves which safetensors file
/// holds the layer; this routine parses that file's header to find
/// the four companion tensors plus the codebooks.
///
/// Codebooks are stored once per safetensors file (the offline
/// quantizer duplicates them across shards). A future per-layer
/// codebook key (`<layerName>.codebook_primary`) takes precedence
/// when present so layer-local codebook configurations remain
/// forward-compatible.
public func loadFullTQLayerPayload(
    modelDir: URL,
    metadata: ShardMetadata,
    layerName: String,
    primaryBits: Int,
    residualBits: Int
) throws -> TQLayerPayload {
    guard let weightEntry = metadata.tensors["\(layerName).weight"] else {
        throw TQLayerLoaderError.sidecarMissingLayerWeight(layerName)
    }
    // Weight shape is [outFeatures, inFeatures] (HuggingFace convention).
    let outFeatures = weightEntry.shape[0]
    let inFeatures = weightEntry.shape[1]

    let safetensorsURL = modelDir.appendingPathComponent(weightEntry.file)
    let header = try parseSafetensorsHeader(fileURL: safetensorsURL)

    func required(_ name: String) throws -> SafetensorsHeaderEntry {
        guard let e = header[name] else {
            throw TQLayerLoaderError.missingTensor(name)
        }
        return e
    }

    let packedPrimaryEntry = try required("\(layerName).packed_primary")
    let packedResidualEntry = try required("\(layerName).packed_residual")
    let normsEntry = try required("\(layerName).norms")
    let seedsEntry = try required("\(layerName).seeds")

    // Prefer a layer-local codebook if present; fall back to the
    // model-wide shared codebook.
    let primaryCodebookKey: String
    let primaryCodebookEntry: SafetensorsHeaderEntry
    if let entry = header["\(layerName).codebook_primary"] {
        primaryCodebookEntry = entry
        primaryCodebookKey = "\(layerName).codebook_primary"
    } else {
        primaryCodebookEntry = try required("tq_codebook_primary")
        primaryCodebookKey = "tq_codebook_primary"
    }
    let residualCodebookKey: String
    let residualCodebookEntry: SafetensorsHeaderEntry
    if let entry = header["\(layerName).codebook_residual"] {
        residualCodebookEntry = entry
        residualCodebookKey = "\(layerName).codebook_residual"
    } else {
        residualCodebookEntry = try required("tq_codebook_residual")
        residualCodebookKey = "tq_codebook_residual"
    }

    let packedPrimary = try loadUInt8MLXArray(
        fileURL: safetensorsURL, entry: packedPrimaryEntry,
        name: "\(layerName).packed_primary")
    let packedResidual = try loadUInt8MLXArray(
        fileURL: safetensorsURL, entry: packedResidualEntry,
        name: "\(layerName).packed_residual")
    let norms = try loadFloat32MLXArray(
        fileURL: safetensorsURL, entry: normsEntry,
        name: "\(layerName).norms")
    let primaryCodebook = try loadFloat32MLXArray(
        fileURL: safetensorsURL, entry: primaryCodebookEntry,
        name: primaryCodebookKey)
    let residualCodebook = try loadFloat32MLXArray(
        fileURL: safetensorsURL, entry: residualCodebookEntry,
        name: residualCodebookKey)

    let seeds = try loadUInt32Array(
        fileURL: safetensorsURL, entry: seedsEntry, name: "\(layerName).seeds")
    guard seeds.count >= 3 else {
        throw TQLayerLoaderError.seedsTooSmall(name: layerName, count: seeds.count)
    }

    // Sanity-check the bit packing implied by the inferred dims so
    // misconfigured `(primaryBits, residualBits)` combinations are
    // caught at load time rather than inside the C kernel.
    let expectedPrimaryRowBytes = inFeatures * primaryBits / 8
    let expectedResidualRowBytes = inFeatures * residualBits / 8
    precondition(
        packedPrimary.shape == [outFeatures, expectedPrimaryRowBytes],
        "packed_primary shape \(packedPrimary.shape) does not match expected " +
        "[\(outFeatures), \(expectedPrimaryRowBytes)] for \(layerName) " +
        "with primaryBits=\(primaryBits)"
    )
    precondition(
        packedResidual.shape == [outFeatures, expectedResidualRowBytes],
        "packed_residual shape \(packedResidual.shape) does not match expected " +
        "[\(outFeatures), \(expectedResidualRowBytes)] for \(layerName) " +
        "with residualBits=\(residualBits)"
    )

    return TQLayerPayload(
        packedPrimary: packedPrimary,
        packedResidual: packedResidual,
        norms: norms,
        primaryCodebook: primaryCodebook,
        residualCodebook: residualCodebook,
        seedPrimary: seeds[0],
        seedResidual: seeds[1],
        blockSize: Int(seeds[2]),
        outFeatures: outFeatures,
        inFeatures: inFeatures
    )
}

// MARK: - Sharded loading

/// Sharding role for a TQ linear layer. Determines which axes of the
/// packed-index tensors and norms vector get sliced for a rank.
public enum TQShardRole: Sendable, Equatable {
    /// Output-dim sharding. The full input stays replicated; rank
    /// `r` keeps output rows `[r * out/N, (r+1) * out/N)`.
    case columnParallel
    /// Input-dim sharding. Rank `r` keeps input columns
    /// `[r * in/N, (r+1) * in/N)` (in elements; the packed-index
    /// tensors slice along axis 1 in bytes). Norms and codebooks
    /// remain replicated since the per-row correction was calibrated
    /// against the full layer.
    case rowParallel
}

/// Load a TurboQuant layer payload and slice it for the calling
/// rank. The returned payload carries `outFeatures` /
/// `inFeatures` reflecting the rank-local dimensions:
///
///   - column-parallel: `outFeatures = full / worldSize`, `inFeatures = full`
///   - row-parallel:    `outFeatures = full`, `inFeatures = full / worldSize`
///
/// At `worldSize == 1` both roles return the full layer unchanged
/// (the start/end range covers the full axis; `.contiguous()` still
/// runs but produces a packed buffer matching the input shape).
public func loadShardedTQLayerPayload(
    modelDir: URL,
    metadata: ShardMetadata,
    layerName: String,
    primaryBits: Int,
    residualBits: Int,
    role: TQShardRole,
    rank: Int,
    worldSize: Int
) throws -> TQLayerPayload {
    let full = try loadFullTQLayerPayload(
        modelDir: modelDir,
        metadata: metadata,
        layerName: layerName,
        primaryBits: primaryBits,
        residualBits: residualBits
    )

    switch role {
    case .columnParallel:
        guard full.outFeatures % worldSize == 0 else {
            throw TQLayerLoaderError.nonDivisibleShardDim(
                name: layerName, dim: full.outFeatures, worldSize: worldSize)
        }
        let rankOut = full.outFeatures / worldSize
        let start = rank * rankOut
        let end = start + rankOut

        // Axis-0 slicing of [out, packed_in_bytes] tensors and the
        // [out] norms vector. Materialise the views so the C kernel
        // reads a packed buffer rather than a strided view.
        let packedPrimary = full.packedPrimary[start..<end].contiguous()
        let packedResidual = full.packedResidual[start..<end].contiguous()
        let norms = full.norms[start..<end].contiguous()
        materialize(packedPrimary, packedResidual, norms)

        return TQLayerPayload(
            packedPrimary: packedPrimary,
            packedResidual: packedResidual,
            norms: norms,
            primaryCodebook: full.primaryCodebook,
            residualCodebook: full.residualCodebook,
            seedPrimary: full.seedPrimary,
            seedResidual: full.seedResidual,
            blockSize: full.blockSize,
            outFeatures: rankOut,
            inFeatures: full.inFeatures
        )

    case .rowParallel:
        guard full.inFeatures % worldSize == 0 else {
            throw TQLayerLoaderError.nonDivisibleShardDim(
                name: layerName, dim: full.inFeatures, worldSize: worldSize)
        }
        let rankIn = full.inFeatures / worldSize

        // Axis-1 byte-range slicing. The packed layout encodes
        // `inFeatures * bits / 8` bytes per output row, so each
        // rank's input slice translates to a `rankIn * bits / 8`
        // wide slab.
        let primaryColsPerRank = rankIn * primaryBits / 8
        let residualColsPerRank = rankIn * residualBits / 8
        let primaryStart = rank * primaryColsPerRank
        let primaryEnd = primaryStart + primaryColsPerRank
        let residualStart = rank * residualColsPerRank
        let residualEnd = residualStart + residualColsPerRank

        let packedPrimary = full.packedPrimary[0..., primaryStart..<primaryEnd].contiguous()
        let packedResidual: MLXArray
        if residualBits > 0 {
            packedResidual = full.packedResidual[0..., residualStart..<residualEnd].contiguous()
        } else {
            // No residual stage configured — pass the (empty) tensor
            // through unchanged. The kernel ignores it when
            // residualBits == 0.
            packedResidual = full.packedResidual
        }
        materialize(packedPrimary, packedResidual)

        return TQLayerPayload(
            packedPrimary: packedPrimary,
            packedResidual: packedResidual,
            // Norms and codebooks are replicated under row-parallel
            // sharding — the per-row correction was calibrated on
            // the whole layer and the codebooks are model-wide.
            norms: full.norms,
            primaryCodebook: full.primaryCodebook,
            residualCodebook: full.residualCodebook,
            seedPrimary: full.seedPrimary,
            seedResidual: full.seedResidual,
            blockSize: full.blockSize,
            outFeatures: full.outFeatures,
            inFeatures: rankIn
        )
    }
}
