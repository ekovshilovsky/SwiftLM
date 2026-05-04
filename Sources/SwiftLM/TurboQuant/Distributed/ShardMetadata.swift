// Decoded `tq_shard_metadata.json` document. Emitted by tq-convert and
// tq-emit-sidecar in the TurboQuant core repo; consumed here by the
// shard-aware weight loader. The sidecar format is versioned; this
// decoder reads format_version 1 (the original layout) and 2, which
// adds a top-level `max_supported_world_size` field surfacing the
// largest tensor-parallel world size the snapshot was converted for.
// Format and semantics are documented in
// specs/2026-04-21-distributed-inference-impl-design.md §6 and in
// turboquant-mlx-core/docs/conversion.md.

import Foundation

/// Classification of a tensor's sharding strategy across cluster ranks.
/// Emitted by the TurboQuant convert tools as part of the shard
/// metadata sidecar; consumed by the shard-aware loader to decide how
/// each rank reads its slice of a tensor from disk.
public enum ShardStrategy: String, Codable, Sendable {
    case columnParallel = "column_parallel"
    case rowParallel = "row_parallel"
    case replicated = "replicated"
    case expertParallel = "expert_parallel"
}

/// Per-tensor entry in the shard metadata manifest. Captures the
/// tensor's shape, dtype, containing safetensors file, absolute byte
/// offset + length within that file, shard strategy, and (for
/// TQ-packed weights) companion keys for the codebook and rotation
/// tensors that accompany it.
public struct ShardTensorEntry: Sendable {
    public let shape: [Int]
    public let dtype: String
    public let file: String
    public let byteOffset: Int
    public let byteLength: Int
    public let shardAxis: Int?
    public let shardStrategy: ShardStrategy
    public let codebookKey: String?
    public let rotationKey: String?
    public let expertIndex: Int?
}

/// Top-level shard metadata document. Load with
/// `ShardMetadata(jsonData: ...)`; downstream components walk the
/// `tensors` dictionary to build per-rank shard plans and compute
/// byte ranges for partial-read loading.
public struct ShardMetadata: Sendable {
    public let formatVersion: Int
    public let modelArchitecture: String
    public let hiddenSize: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int?
    public let numExperts: Int?
    public let topK: Int?
    /// Largest tensor-parallel world size the snapshot was converted for.
    /// `nil` for format_version 1 sidecars (pre-dates the field); for
    /// format_version 2 sidecars it reflects the `--target-world-size`
    /// the operator passed to tq-convert. Cluster bring-up surfaces a
    /// clean error when a runtime cluster's world size exceeds this.
    public let maxSupportedWorldSize: Int?
    public let tensors: [String: ShardTensorEntry]

    public init(jsonData: Data) throws {
        let decoder = JSONDecoder()
        let raw = try decoder.decode(RawShardMetadata.self, from: jsonData)
        formatVersion = raw.formatVersion
        modelArchitecture = raw.modelArchitecture
        hiddenSize = raw.hiddenSize
        numAttentionHeads = raw.numAttentionHeads
        intermediateSize = raw.intermediateSize
        numExperts = raw.numExperts
        topK = raw.topK
        maxSupportedWorldSize = raw.maxSupportedWorldSize
        tensors = try raw.tensors.mapValues { try $0.intoEntry() }
    }
}

public enum ShardMetadataError: Error, Equatable {
    case unknownShardStrategy(String)
}

// MARK: - Internal decoding shapes
//
// The JSON wire format uses snake_case per the convert-tool output;
// we keep the Swift public API in idiomatic camelCase and translate
// via a private `Raw` intermediary. Keeping the private types close
// to the public ones makes the mapping obvious to future readers.

private struct RawShardMetadata: Decodable {
    let formatVersion: Int
    let modelArchitecture: String
    let hiddenSize: Int
    let numAttentionHeads: Int
    let intermediateSize: Int?
    let numExperts: Int?
    let topK: Int?
    let maxSupportedWorldSize: Int?
    let tensors: [String: RawTensorEntry]

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case modelArchitecture = "model_architecture"
        case hiddenSize = "hidden_size"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case numExperts = "num_experts"
        case topK = "top_k"
        case maxSupportedWorldSize = "max_supported_world_size"
        case tensors
    }
}

private struct RawTensorEntry: Decodable {
    let shape: [Int]
    let dtype: String
    let file: String
    let byteOffset: Int
    let byteLength: Int
    let shardAxis: Int?
    let shardStrategy: String
    let codebookKey: String?
    let rotationKey: String?
    let expertIndex: Int?

    enum CodingKeys: String, CodingKey {
        case shape, dtype, file
        case byteOffset = "byte_offset"
        case byteLength = "byte_length"
        case shardAxis = "shard_axis"
        case shardStrategy = "shard_strategy"
        case codebookKey = "codebook_key"
        case rotationKey = "rotation_key"
        case expertIndex = "expert_index"
    }

    func intoEntry() throws -> ShardTensorEntry {
        guard let strategy = ShardStrategy(rawValue: shardStrategy) else {
            throw ShardMetadataError.unknownShardStrategy(shardStrategy)
        }
        return ShardTensorEntry(
            shape: shape,
            dtype: dtype,
            file: file,
            byteOffset: byteOffset,
            byteLength: byteLength,
            shardAxis: shardAxis,
            shardStrategy: strategy,
            codebookKey: codebookKey,
            rotationKey: rotationKey,
            expertIndex: expertIndex
        )
    }
}
