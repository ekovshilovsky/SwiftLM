// Tensor-parallel Qwen2/Qwen3-family decoder-only model. Constructs
// a rank-local transformer stack from a `ShardMetadata` sidecar plus
// the safetensors files in the model directory.
//
// Layer-to-sharding mapping for each transformer block:
//
//   - `input_layernorm` (RMSNorm, pre-attention) — replicated. Stored
//     as a raw bfloat16 weight tensor and applied via
//     `MLXFast.rmsNorm` so the loaded scale survives without the dtype
//     conversion that `MLXNN.RMSNorm` would force at construction.
//   - `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj` —
//     column-parallel, each a separate `TurboQuantAllToShardedLinear`.
//     The fixture ships split QKV (no fused tensor); each rank owns
//     its slice of the output rows of the three projections. Each
//     projection has a replicated bfloat16 bias tensor added to the
//     output before the per-head reshape.
//   - `self_attn.o_proj` — row-parallel
//     (`TurboQuantShardedToAllLinear`). Consumes the rank-local
//     attention output and `allSum`s the partial back to the
//     replicated hidden state.
//   - `post_attention_layernorm` (RMSNorm, pre-MLP) — replicated, raw
//     bfloat16 weight as above.
//   - `mlp.gate_proj`, `mlp.up_proj` — column-parallel, each a
//     separate `TurboQuantAllToShardedLinear`. Split MLP layout, no
//     fused gate-up tensor in the fixture.
//   - `mlp.down_proj` — row-parallel (`TurboQuantShardedToAllLinear`).
//
// Top-level:
//
//   - `embed_tokens` — `ReplicatedEmbedding`. The current implementation
//     ships only the replicated strategy; vocab-parallel can swap in
//     later via the `ShardedEmbeddingLayer` protocol. The
//     row table is materialised from the TQ-packed `model.embed_tokens`
//     payload via the shared `materialiseEmbeddingTable` helper —
//     same dequant path the single-rank oracle uses, so size-1
//     equivalence is preserved at the construction boundary.
//   - `layers` — array of `DistributedQwenTransformerBlock` sized to
//     `config.numHiddenLayers`.
//   - final RMSNorm scale over the hidden dim, stored as a raw
//     bfloat16 weight and applied via `MLXFast.rmsNorm`.
//   - `lm_head` — `ReplicatedLMHead`. When `tieWordEmbeddings` is
//     true the lm_head shares the same `MLXArray` instance as
//     `embed_tokens`.
//
// Forward pass. Implements a standard Qwen2 decoder: token embedding
// lookup, N transformer blocks (RMSNorm -> GQA self-attention with
// RoPE -> residual -> RMSNorm -> SwiGLU MLP -> residual), final
// RMSNorm, and the lm_head matmul against the materialised embedding
// table.
//
// The forward pass supports both prefill (no cache argument) and
// incremental decode (caller supplies one `KVCache` per transformer
// block via `cache:`). When a cache is supplied, RoPE is applied at
// the cache's current offset so position encodings advance correctly
// across decode steps, the cache is updated with the rotated
// per-block keys and values, and the SDPA mask is sourced from
// `KVCache.makeMask` rather than the prefill `.causal` shorthand.
// At `worldSize=1` each rank holds the full KV-head set, so the
// cache is sized for `numKeyValueHeads * headDim`. Multi-rank with
// vocab-/head-parallel KV would size the cache for the rank's slice
// instead; the present configuration exercises the single-rank
// equivalent only.
//
// Compressed KV via the `TurboQuantKVCache` Swift bridge in
// `TurboQuantBridge.swift` would require Data round-tripping per
// decode step and is intentionally deferred. `KVCacheSimple` from
// MLXLMCommon is the canonical interface in this codebase.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Architecture configuration for a Qwen2/Qwen3-family decoder.
/// Decoded from the standard HuggingFace `config.json` next to the
/// model's safetensors files. Only the fields the rank-local model
/// construction actually needs are listed; missing-but-needed fields
/// will surface as decode failures with a clear key name.
public struct DistributedQwenConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool

    public init(
        hiddenSize: Int,
        numHiddenLayers: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        rmsNormEps: Float,
        vocabSize: Int,
        ropeTheta: Float,
        tieWordEmbeddings: Bool
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.rmsNormEps = rmsNormEps
        self.vocabSize = vocabSize
        self.ropeTheta = ropeTheta
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        self.rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        self.vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        // Default to 1e6 (Qwen2's documented value) if the field is
        // absent so older configs round-trip.
        self.ropeTheta =
            try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }

    /// Decode a configuration from a `config.json` URL.
    public static func load(from configURL: URL) throws -> DistributedQwenConfiguration {
        let data = try Data(contentsOf: configURL)
        return try JSONDecoder().decode(DistributedQwenConfiguration.self, from: data)
    }
}

// MARK: - Per-block submodules

/// Sharded self-attention sub-module. Holds the four projection
/// layers that the transformer block dispatches into, plus the
/// replicated bfloat16 q/k/v bias tensors added to the projection
/// outputs prior to the per-head reshape. `o_proj` has no bias in the
/// Qwen2/2.5 architecture.
public final class DistributedQwenAttention: Module {
    public let q_proj: TurboQuantAllToShardedLinear
    public let k_proj: TurboQuantAllToShardedLinear
    public let v_proj: TurboQuantAllToShardedLinear
    public let o_proj: TurboQuantShardedToAllLinear

    /// Replicated bfloat16 bias for the column-parallel q/k/v
    /// projections. Applied to the kernel's fp16 output via a
    /// runtime-side dtype cast that matches the activation dtype. The
    /// bias must be added before the projection result is reshaped
    /// into per-head slices — adding after the reshape would
    /// redistribute bias entries across heads and break equivalence
    /// with the standard Qwen2 implementation.
    public let qBias: MLXArray
    public let kBias: MLXArray
    public let vBias: MLXArray

    public init(
        q_proj: TurboQuantAllToShardedLinear,
        k_proj: TurboQuantAllToShardedLinear,
        v_proj: TurboQuantAllToShardedLinear,
        o_proj: TurboQuantShardedToAllLinear,
        qBias: MLXArray,
        kBias: MLXArray,
        vBias: MLXArray
    ) {
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.qBias = qBias
        self.kBias = kBias
        self.vBias = vBias
        super.init()
    }
}

/// Sharded MLP sub-module (split gate / up / down). The fixture has
/// no biases on any of the three MLP projections.
public final class DistributedQwenMLP: Module {
    public let gate_proj: TurboQuantAllToShardedLinear
    public let up_proj: TurboQuantAllToShardedLinear
    public let down_proj: TurboQuantShardedToAllLinear

    public init(
        gate_proj: TurboQuantAllToShardedLinear,
        up_proj: TurboQuantAllToShardedLinear,
        down_proj: TurboQuantShardedToAllLinear
    ) {
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        super.init()
    }
}

/// One transformer block: pre-attention RMSNorm, sharded self-
/// attention, residual, pre-MLP RMSNorm, sharded MLP, residual.
///
/// Both norms are stored as raw bfloat16 weight tensors rather than
/// `MLXNN.RMSNorm` instances so the bfloat16 scale loaded from the
/// passthrough safetensors round-trips without the implicit dtype
/// promotion that `RMSNorm`'s `let weight` would force. The forward
/// path applies them via `MLXFast.rmsNorm(_:weight:eps:)`.
public final class DistributedQwenTransformerBlock: Module {
    public let inputLayernormWeight: MLXArray
    public let self_attn: DistributedQwenAttention
    public let postAttentionLayernormWeight: MLXArray
    public let mlp: DistributedQwenMLP

    public init(
        inputLayernormWeight: MLXArray,
        self_attn: DistributedQwenAttention,
        postAttentionLayernormWeight: MLXArray,
        mlp: DistributedQwenMLP
    ) {
        self.inputLayernormWeight = inputLayernormWeight
        self.self_attn = self_attn
        self.postAttentionLayernormWeight = postAttentionLayernormWeight
        self.mlp = mlp
        super.init()
    }
}

// MARK: - Top-level model

/// Construction-time errors raised by `DistributedQwenModel.init`.
/// Thrown rather than fatal so callers (including the HTTP layer and
/// the test harness) can recover from misconfiguration cleanly.
public enum DistributedQwenModelError: Error, CustomStringConvertible {
    /// `config.tieWordEmbeddings` is false but the loader has no
    /// implementation for an independently-trained `lm_head` weight.
    /// Qwen2.5-Coder-3B fixtures all ship tied embeddings, so the
    /// untied branch surfaces here until a dedicated load path lands.
    case untiedLMHeadNotSupported
    /// A configuration field that must be evenly divisible by the
    /// world size is not. Carries the offending field name, value,
    /// and world size so the caller can format an actionable message
    /// (e.g. "select a different rank count or fixture").
    case worldSizeNotDivisor(field: String, value: Int, worldSize: Int)
    /// The caller supplied an externally-materialised embedding table
    /// whose shape or dtype does not match the model configuration.
    case embeddingTableMismatch(expectedShape: [Int], gotShape: [Int],
                                expectedDType: DType, gotDType: DType)

    public var description: String {
        switch self {
        case .untiedLMHeadNotSupported:
            return "untied lm_head is not supported by DistributedQwenModel; " +
                "config.tieWordEmbeddings=false but no separate load path is wired"
        case .worldSizeNotDivisor(let field, let value, let worldSize):
            return "DistributedQwenModel requires \(field) (=\(value)) to be " +
                "evenly divisible by worldSize=\(worldSize)"
        case .embeddingTableMismatch(let es, let gs, let ed, let gd):
            return "supplied embedding table has shape \(gs) dtype \(gd); " +
                "expected shape \(es) dtype \(ed)"
        }
    }
}

/// Tensor-parallel Qwen decoder model. Construction loads every
/// sharded layer from the model directory and assembles the block
/// stack. The forward pass mirrors the single-rank reference oracle's
/// choreography exactly so size-1 distributed runs match the oracle
/// within fp16 noise.
public final class DistributedQwenModel: Module {
    public let config: DistributedQwenConfiguration
    public let group: DistributedGroup
    public let rank: Int
    public let worldSize: Int

    /// Number of attention heads owned by this rank under
    /// column-parallel QKV sharding. Equal to
    /// `config.numAttentionHeads / worldSize`. Cached so the forward
    /// pass does not have to recompute it per block.
    public let rankNumQHeads: Int
    /// Number of key-value heads owned by this rank.
    public let rankNumKVHeads: Int
    /// Per-rank intermediate (MLP hidden) dim under column-parallel
    /// gate / up sharding.
    public let rankIntermediate: Int
    /// Q-side flattened width for this rank: `rankNumQHeads * headDim`.
    public let rankQDim: Int
    /// K/V-side flattened width for this rank: `rankNumKVHeads * headDim`.
    public let rankKVDim: Int
    /// Standard Qwen2 head dim: `hiddenSize / numAttentionHeads`.
    public let headDim: Int

    public let embed_tokens: ReplicatedEmbedding
    public let layers: [DistributedQwenTransformerBlock]
    /// Final RMSNorm scale, replicated and stored in the safetensors
    /// dtype (bfloat16 in the Qwen2.5-Coder-3B fixture). Applied via
    /// `MLXFast.rmsNorm` so the loaded scale survives without an
    /// implicit dtype cast.
    public let finalNormWeight: MLXArray
    public let lm_head: ReplicatedLMHead

    /// Build a rank-local distributed Qwen model. Convenience entry
    /// point that materialises the embedding table internally.
    ///
    /// - Parameters:
    ///   - metadata: Decoded `tq_shard_metadata.json` sidecar.
    ///   - modelDir: Directory holding the safetensors files plus
    ///     the standard HuggingFace `config.json`.
    ///   - group: MLX distributed group; `group.rank` and
    ///     `group.size` drive shard selection by default. The
    ///     explicit `rank` and `worldSize` parameters override the
    ///     group's identity for in-process multi-rank tests.
    ///   - primaryBits / residualBits: TurboQuant bit widths. Default
    ///     to the 4+4 layout shipped by tq-convert; override only for
    ///     fixtures produced with a non-standard quantizer
    ///     configuration.
    ///   - rank: Override for the rank used to slice this model's
    ///     shards. Defaults to `group.rank`. The override exists so
    ///     in-process tests can construct multiple rank-local models
    ///     against a singleton `DistributedGroup` without standing up
    ///     a real multi-process backend (multi-process numerical
    ///     equivalence is exercised separately).
    ///   - worldSize: Override for the world size. Defaults to
    ///     `group.size`. Tests pair this with `rank` to sweep through
    ///     every shard of an N-way layout in a single process.
    public convenience init(
        metadata: ShardMetadata,
        modelDir: URL,
        group: DistributedGroup,
        primaryBits: Int = 4,
        residualBits: Int = 4,
        rank: Int? = nil,
        worldSize: Int? = nil
    ) throws {
        // Materialise the embedding table once and forward to the
        // designated init. Callers that want to share an
        // already-materialised table across multiple model instances
        // should construct via the `embeddingTable:`-accepting init
        // directly so the ~600 MB allocation is not duplicated.
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try DistributedQwenConfiguration.load(from: configURL)
        let embeddingTable = try materialiseEmbeddingTable(
            modelDir: modelDir,
            metadata: metadata,
            embeddingLayerName: "model.embed_tokens",
            hiddenSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            primaryBits: primaryBits,
            residualBits: residualBits
        )
        try self.init(
            metadata: metadata,
            modelDir: modelDir,
            group: group,
            embeddingTable: embeddingTable,
            primaryBits: primaryBits,
            residualBits: residualBits,
            rank: rank,
            worldSize: worldSize
        )
    }

    /// Designated initialiser. Accepts an externally-materialised
    /// embedding table so multiple model instances can share the same
    /// ~600 MB `[vocabSize, hiddenSize]` allocation. The Metal
    /// allocator's wired-memory ceiling on a single process is the
    /// reason this matters in practice: hosting more than one model
    /// per process (the oracle plus the rank-local distributed model
    /// in equivalence tests, or a coordinator and joiner in the same
    /// process) requires the embedding to be a single allocation
    /// rather than two.
    ///
    /// - Parameters:
    ///   - metadata: Decoded `tq_shard_metadata.json` sidecar.
    ///   - modelDir: Directory holding the safetensors files plus the
    ///     standard HuggingFace `config.json`.
    ///   - group: MLX distributed group used to resolve `allSum`
    ///     collectives at forward time.
    ///   - embeddingTable: Pre-materialised `[vocabSize, hiddenSize]`
    ///     float16 row table. Validated against the configuration.
    ///     The same MLXArray instance is reused for both
    ///     `embed_tokens` and (when tied) `lm_head`.
    ///   - primaryBits / residualBits: TurboQuant bit widths.
    ///   - rank: Override for the rank used to slice this model's
    ///     shards. See the convenience init for rationale.
    ///   - worldSize: Override for the world size.
    public init(
        metadata: ShardMetadata,
        modelDir: URL,
        group: DistributedGroup,
        embeddingTable: MLXArray,
        primaryBits: Int = 4,
        residualBits: Int = 4,
        rank: Int? = nil,
        worldSize: Int? = nil
    ) throws {
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try DistributedQwenConfiguration.load(from: configURL)
        self.config = config
        self.group = group
        let resolvedRank = rank ?? group.rank
        let resolvedWorldSize = worldSize ?? group.size
        self.rank = resolvedRank
        self.worldSize = resolvedWorldSize

        // Tensor-parallel sharding splits q/k heads, KV heads, and the
        // MLP intermediate dim evenly across ranks. The fixture's
        // GQA layout fixes `numKeyValueHeads = 2` for Qwen2.5-Coder-3B
        // — only `worldSize ∈ {1, 2}` works for that fixture. Surface
        // the constraint as a thrown error so a misconfigured
        // worldSize fails at construction with a clear field name
        // rather than as a downstream shape mismatch.
        if config.numAttentionHeads % resolvedWorldSize != 0 {
            throw DistributedQwenModelError.worldSizeNotDivisor(
                field: "numAttentionHeads",
                value: config.numAttentionHeads,
                worldSize: resolvedWorldSize)
        }
        if config.numKeyValueHeads % resolvedWorldSize != 0 {
            throw DistributedQwenModelError.worldSizeNotDivisor(
                field: "numKeyValueHeads",
                value: config.numKeyValueHeads,
                worldSize: resolvedWorldSize)
        }
        if config.intermediateSize % resolvedWorldSize != 0 {
            throw DistributedQwenModelError.worldSizeNotDivisor(
                field: "intermediateSize",
                value: config.intermediateSize,
                worldSize: resolvedWorldSize)
        }

        let headDim = config.hiddenSize / config.numAttentionHeads
        let rankNumQHeads = config.numAttentionHeads / resolvedWorldSize
        let rankNumKVHeads = config.numKeyValueHeads / resolvedWorldSize
        let rankIntermediate = config.intermediateSize / resolvedWorldSize
        let rankQDim = rankNumQHeads * headDim
        let rankKVDim = rankNumKVHeads * headDim
        self.headDim = headDim
        self.rankNumQHeads = rankNumQHeads
        self.rankNumKVHeads = rankNumKVHeads
        self.rankIntermediate = rankIntermediate
        self.rankQDim = rankQDim
        self.rankKVDim = rankKVDim

        // Validate the supplied embedding table against the config.
        // Catching shape/dtype drift here avoids opaque downstream
        // failures inside `MLX.take` or the lm_head matmul.
        let expectedShape = [config.vocabSize, config.hiddenSize]
        let expectedDType: DType = .float16
        if embeddingTable.shape != expectedShape || embeddingTable.dtype != expectedDType {
            throw DistributedQwenModelError.embeddingTableMismatch(
                expectedShape: expectedShape,
                gotShape: embeddingTable.shape,
                expectedDType: expectedDType,
                gotDType: embeddingTable.dtype)
        }

        // Both `embed_tokens` and (when tied) `lm_head` reuse the same
        // MLXArray instance so identity checks on the tied path hold.
        // The current implementation ships only the
        // replicated-embedding strategy. A future vocab-parallel
        // implementation would materialise just this rank's slice, but
        // every rank still owns the full table here.
        self.embed_tokens = ReplicatedEmbedding.make(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            weight: embeddingTable,
            strategy: .replicated,
            group: group
        )

        // When `tie_word_embeddings` is true the lm_head reuses the
        // exact embedding instance — the structural test asserts
        // pointer identity (===), so allocating a fresh table for the
        // tied branch would silently regress that contract. Untied
        // models would load `lm_head.weight` from a separate TQ
        // payload; Qwen2.5-Coder-3B fixtures all ship tied embeddings.
        let lmHeadWeight: MLXArray
        if config.tieWordEmbeddings {
            lmHeadWeight = embeddingTable
        } else {
            throw DistributedQwenModelError.untiedLMHeadNotSupported
        }
        self.lm_head = ReplicatedLMHead.make(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            weight: lmHeadWeight,
            strategy: .replicated,
            group: group
        )

        // Final norm scale. Qwen2.5-Coder-3B stores it as bfloat16 in
        // the passthrough file; load it as-is so `MLXFast.rmsNorm`
        // operates without a dtype cast on every forward pass.
        self.finalNormWeight = try loadReplicatedTensor(
            modelDir: modelDir,
            metadata: metadata,
            tensorName: "model.norm.weight",
            expectedDType: "BF16"
        )

        // Per-layer block construction. Every block loads its six TQ
        // linear layers (q/k/v + o, gate/up + down) from the
        // safetensors file the sidecar points at, applies the
        // appropriate rank slicing, and wraps the rank-local payloads
        // in the matching sharded-layer types.
        var blocks: [DistributedQwenTransformerBlock] = []
        blocks.reserveCapacity(config.numHiddenLayers)

        // Sidecar lookup helper. Reads the full input dimension for a
        // tensor directly off the metadata document so row-parallel
        // construction does not depend on the loader's
        // already-halved `inFeatures` field. The sidecar stores
        // `weight.shape = [outFeatures, fullInFeatures]` per
        // HuggingFace convention.
        func sidecarFullInFeatures(_ layerName: String) throws -> Int {
            guard let entry = metadata.tensors["\(layerName).weight"] else {
                throw TQLayerLoaderError.sidecarMissingLayerWeight(layerName)
            }
            return entry.shape[1]
        }

        for layerIdx in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(layerIdx)"

            // Replicated norm scales. All bfloat16 in the
            // Qwen2.5-Coder-3B fixture; loaded as-is so the forward
            // path does not pay an extra dtype cast on every call.
            let inputNormWeight = try loadReplicatedTensor(
                modelDir: modelDir,
                metadata: metadata,
                tensorName: "\(prefix).input_layernorm.weight",
                expectedDType: "BF16"
            )
            let postNormWeight = try loadReplicatedTensor(
                modelDir: modelDir,
                metadata: metadata,
                tensorName: "\(prefix).post_attention_layernorm.weight",
                expectedDType: "BF16"
            )

            // q/k/v biases. The convert tool ships the full-width
            // bias because the underlying tensor has no shard
            // strategy. Each rank holds only the slice that aligns
            // with its column-parallel projection's output. Slicing
            // here keeps the bias-add in the forward pass dimensionally
            // consistent with the projection output. `.contiguous()`
            // materialises a fresh dense buffer because the bias-add
            // expects contiguous storage.
            let qBiasFull = try loadReplicatedTensor(
                modelDir: modelDir,
                metadata: metadata,
                tensorName: "\(prefix).self_attn.q_proj.bias",
                expectedDType: "BF16"
            )
            let kBiasFull = try loadReplicatedTensor(
                modelDir: modelDir,
                metadata: metadata,
                tensorName: "\(prefix).self_attn.k_proj.bias",
                expectedDType: "BF16"
            )
            let vBiasFull = try loadReplicatedTensor(
                modelDir: modelDir,
                metadata: metadata,
                tensorName: "\(prefix).self_attn.v_proj.bias",
                expectedDType: "BF16"
            )
            let qBiasStart = resolvedRank * rankQDim
            let qBiasEnd = qBiasStart + rankQDim
            let kBiasStart = resolvedRank * rankKVDim
            let kBiasEnd = kBiasStart + rankKVDim
            let vBiasStart = resolvedRank * rankKVDim
            let vBiasEnd = vBiasStart + rankKVDim
            let qBias = qBiasFull[qBiasStart ..< qBiasEnd].contiguous()
            let kBias = kBiasFull[kBiasStart ..< kBiasEnd].contiguous()
            let vBias = vBiasFull[vBiasStart ..< vBiasEnd].contiguous()

            let qPayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).self_attn.q_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .columnParallel, rank: resolvedRank, worldSize: resolvedWorldSize)
            let kPayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).self_attn.k_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .columnParallel, rank: resolvedRank, worldSize: resolvedWorldSize)
            let vPayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).self_attn.v_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .columnParallel, rank: resolvedRank, worldSize: resolvedWorldSize)
            let oPayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).self_attn.o_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .rowParallel, rank: resolvedRank, worldSize: resolvedWorldSize)

            let gatePayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).mlp.gate_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .columnParallel, rank: resolvedRank, worldSize: resolvedWorldSize)
            let upPayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).mlp.up_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .columnParallel, rank: resolvedRank, worldSize: resolvedWorldSize)
            let downPayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).mlp.down_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .rowParallel, rank: resolvedRank, worldSize: resolvedWorldSize)

            let qLayer = try TurboQuantAllToShardedLinear(
                fullInFeatures: qPayload.inFeatures,
                rankOutFeatures: qPayload.outFeatures,
                primaryBits: primaryBits, residualBits: residualBits,
                packedPrimary: qPayload.packedPrimary,
                packedResidual: qPayload.packedResidual,
                norms: qPayload.norms,
                primaryCodebook: qPayload.primaryCodebook,
                residualCodebook: qPayload.residualCodebook,
                seedPrimary: qPayload.seedPrimary,
                seedResidual: qPayload.seedResidual,
                blockSize: qPayload.blockSize,
                group: group)
            let kLayer = try TurboQuantAllToShardedLinear(
                fullInFeatures: kPayload.inFeatures,
                rankOutFeatures: kPayload.outFeatures,
                primaryBits: primaryBits, residualBits: residualBits,
                packedPrimary: kPayload.packedPrimary,
                packedResidual: kPayload.packedResidual,
                norms: kPayload.norms,
                primaryCodebook: kPayload.primaryCodebook,
                residualCodebook: kPayload.residualCodebook,
                seedPrimary: kPayload.seedPrimary,
                seedResidual: kPayload.seedResidual,
                blockSize: kPayload.blockSize,
                group: group)
            let vLayer = try TurboQuantAllToShardedLinear(
                fullInFeatures: vPayload.inFeatures,
                rankOutFeatures: vPayload.outFeatures,
                primaryBits: primaryBits, residualBits: residualBits,
                packedPrimary: vPayload.packedPrimary,
                packedResidual: vPayload.packedResidual,
                norms: vPayload.norms,
                primaryCodebook: vPayload.primaryCodebook,
                residualCodebook: vPayload.residualCodebook,
                seedPrimary: vPayload.seedPrimary,
                seedResidual: vPayload.seedResidual,
                blockSize: vPayload.blockSize,
                group: group)
            // Row-parallel `o_proj`: `localInFeatures` is the per-rank
            // input slice; `fullInFeatures` comes from the sidecar
            // entry's full weight shape so the kernel's combined_scale
            // derivation does not depend on the loader's halved
            // `inFeatures` value.
            let oFullIn = try sidecarFullInFeatures("\(prefix).self_attn.o_proj")
            let oLayer = try TurboQuantShardedToAllLinear(
                fullInFeatures: oFullIn,
                rankOutFeatures: oPayload.outFeatures,
                localInFeatures: oPayload.inFeatures,
                primaryBits: primaryBits, residualBits: residualBits,
                packedPrimary: oPayload.packedPrimary,
                packedResidual: oPayload.packedResidual,
                norms: oPayload.norms,
                primaryCodebook: oPayload.primaryCodebook,
                residualCodebook: oPayload.residualCodebook,
                seedPrimary: oPayload.seedPrimary,
                seedResidual: oPayload.seedResidual,
                blockSize: oPayload.blockSize,
                group: group,
                worldSize: resolvedWorldSize)

            let gateLayer = try TurboQuantAllToShardedLinear(
                fullInFeatures: gatePayload.inFeatures,
                rankOutFeatures: gatePayload.outFeatures,
                primaryBits: primaryBits, residualBits: residualBits,
                packedPrimary: gatePayload.packedPrimary,
                packedResidual: gatePayload.packedResidual,
                norms: gatePayload.norms,
                primaryCodebook: gatePayload.primaryCodebook,
                residualCodebook: gatePayload.residualCodebook,
                seedPrimary: gatePayload.seedPrimary,
                seedResidual: gatePayload.seedResidual,
                blockSize: gatePayload.blockSize,
                group: group)
            let upLayer = try TurboQuantAllToShardedLinear(
                fullInFeatures: upPayload.inFeatures,
                rankOutFeatures: upPayload.outFeatures,
                primaryBits: primaryBits, residualBits: residualBits,
                packedPrimary: upPayload.packedPrimary,
                packedResidual: upPayload.packedResidual,
                norms: upPayload.norms,
                primaryCodebook: upPayload.primaryCodebook,
                residualCodebook: upPayload.residualCodebook,
                seedPrimary: upPayload.seedPrimary,
                seedResidual: upPayload.seedResidual,
                blockSize: upPayload.blockSize,
                group: group)
            let downFullIn = try sidecarFullInFeatures("\(prefix).mlp.down_proj")
            let downLayer = try TurboQuantShardedToAllLinear(
                fullInFeatures: downFullIn,
                rankOutFeatures: downPayload.outFeatures,
                localInFeatures: downPayload.inFeatures,
                primaryBits: primaryBits, residualBits: residualBits,
                packedPrimary: downPayload.packedPrimary,
                packedResidual: downPayload.packedResidual,
                norms: downPayload.norms,
                primaryCodebook: downPayload.primaryCodebook,
                residualCodebook: downPayload.residualCodebook,
                seedPrimary: downPayload.seedPrimary,
                seedResidual: downPayload.seedResidual,
                blockSize: downPayload.blockSize,
                group: group,
                worldSize: resolvedWorldSize)

            let attn = DistributedQwenAttention(
                q_proj: qLayer,
                k_proj: kLayer,
                v_proj: vLayer,
                o_proj: oLayer,
                qBias: qBias,
                kBias: kBias,
                vBias: vBias
            )
            let mlp = DistributedQwenMLP(
                gate_proj: gateLayer,
                up_proj: upLayer,
                down_proj: downLayer
            )

            blocks.append(DistributedQwenTransformerBlock(
                inputLayernormWeight: inputNormWeight,
                self_attn: attn,
                postAttentionLayernormWeight: postNormWeight,
                mlp: mlp
            ))
        }
        self.layers = blocks

        super.init()
    }

    // MARK: - Forward pass

    /// Allocate one `KVCacheSimple` per transformer block. Cache
    /// ownership is the *caller*'s responsibility; the model itself
    /// remains stateless and the array is threaded through subsequent
    /// `callAsFunction(_:cache:)` invocations during incremental
    /// decode. At `worldSize == 1` each cache is sized for the full
    /// `numKeyValueHeads * headDim` per token; multi-rank with KV-head
    /// sharding would size the cache for the rank's slice instead.
    public func makeCache() -> [KVCache] {
        (0 ..< config.numHiddenLayers).map { _ in KVCacheSimple() }
    }

    /// Run the forward pass on a batch of token-id sequences.
    ///
    /// At `worldSize = 1` the per-layer choreography matches the
    /// single-rank reference oracle exactly: same RoPE call shape,
    /// same SDPA scale and mask mode, same residual ordering, same
    /// bias-add timing, and (when a cache is supplied) the same cache
    /// update sequence. Multi-rank correctness rides on the same
    /// graph plus the row-parallel layers' `allSum` collectives, which
    /// collapse to a no-op at size 1.
    ///
    /// - Parameters:
    ///   - tokenIds: Integer-typed `MLXArray` of shape
    ///     `[batch, length]`.
    ///   - cache: Optional per-block KV cache list. When `nil`, runs a
    ///     cache-less prefill identical to the original forward path.
    ///     When non-`nil`, must contain exactly `config.numHiddenLayers`
    ///     entries. Each block's cache is updated in-place with the
    ///     rotated keys / values for the new tokens, and attention
    ///     attends across the cache's accumulated history plus the new
    ///     positions. RoPE is applied at the cache's pre-update offset
    ///     so positional encodings advance correctly across decode
    ///     steps.
    /// - Returns: Logits of shape `[batch, length, vocabSize]` in
    ///   float16.
    public func callAsFunction(
        _ tokenIds: MLXArray,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        precondition(tokenIds.ndim == 2, "tokenIds must be [batch, length]; got \(tokenIds.shape)")
        if let cache {
            precondition(
                cache.count == config.numHiddenLayers,
                "cache must have one entry per transformer block; " +
                "expected \(config.numHiddenLayers), got \(cache.count)"
            )
        }

        let batch = tokenIds.dim(0)
        let length = tokenIds.dim(1)
        let hiddenSize = config.hiddenSize
        let attnScale = 1.0 / Float(headDim).squareRoot()

        // Token embedding lookup. `MLX.take` along axis 0 of the
        // materialised `[V, H]` table produces `[batch, length, H]`.
        // Cast once after the lookup; the TQ kernel is most
        // numerically consistent on fp16 inputs and keeps fp16 across
        // its internal compute, so downstream casts are unnecessary.
        var hidden = MLX.take(embed_tokens.weight, tokenIds, axis: 0)
        hidden = hidden.asType(.float16)

        // The TQ kernel sees a 2-D `[batch * length, H]` activation.
        // Flatten before each projection and unflatten afterwards.
        let qkvBatch = batch * length

        for (blockIdx, block) in layers.enumerated() {
            let blockCache = cache?[blockIdx]

            // Pre-attention RMSNorm + self-attention.
            let preAttn = MLXFast.rmsNorm(
                hidden,
                weight: block.inputLayernormWeight,
                eps: config.rmsNormEps
            )

            let preAttnFlat = preAttn.reshaped(qkvBatch, hiddenSize)

            var q = block.self_attn.q_proj(preAttnFlat)
            var k = block.self_attn.k_proj(preAttnFlat)
            var v = block.self_attn.v_proj(preAttnFlat)

            // Add q/k/v biases prior to the per-head reshape. The
            // kernel returns fp16; the bias tensors are bf16 — a
            // runtime-side fp16 cast keeps the dtype consistent for
            // the subsequent reshape and RoPE. Both projection output
            // and bias are sized to this rank's head slice.
            q = q + block.self_attn.qBias.asType(q.dtype)
            k = k + block.self_attn.kBias.asType(k.dtype)
            v = v + block.self_attn.vBias.asType(v.dtype)

            // Re-introduce the [batch, length] axes and split into
            // per-head slices. Layout follows the standard Qwen2
            // convention: (B, n_heads, L, head_dim). Each rank holds
            // `numAttentionHeads / worldSize` Q heads and
            // `numKeyValueHeads / worldSize` KV heads, so the reshape
            // operates on rank-local head counts. Cache entries are
            // stored in this same layout so the cache update can drop
            // straight in.
            q = q.reshaped(batch, length, rankNumQHeads, headDim).transposed(0, 2, 1, 3)
            k = k.reshaped(batch, length, rankNumKVHeads, headDim).transposed(0, 2, 1, 3)
            v = v.reshaped(batch, length, rankNumKVHeads, headDim).transposed(0, 2, 1, 3)

            // RoPE is applied at the cache's current offset so tokens
            // appended during decode receive the rotation appropriate
            // to their absolute position in the sequence.
            // `KVCacheSimple.update` advances `offset` by the new
            // length, so the offset must be sampled before the update
            // call below.
            let ropeOffset = blockCache?.offset ?? 0

            q = MLXFast.RoPE(
                q,
                dimensions: headDim,
                traditional: false,
                base: config.ropeTheta,
                scale: 1.0,
                offset: ropeOffset
            )
            k = MLXFast.RoPE(
                k,
                dimensions: headDim,
                traditional: false,
                base: config.ropeTheta,
                scale: 1.0,
                offset: ropeOffset
            )

            // Cache update happens after RoPE so the stored keys carry
            // the rotation. The cache returns the full `[0, offset+L)`
            // span of keys / values; `MLXFast.scaledDotProductAttention`
            // handles the GQA broadcast internally so the small
            // (numKVHeads) tensors are passed through unmodified.
            let attendKeys: MLXArray
            let attendValues: MLXArray
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if let blockCache {
                let (cachedKeys, cachedValues) = blockCache.update(keys: k, values: v)
                attendKeys = cachedKeys
                attendValues = cachedValues
                // `KVCache.makeMask` returns the right mask shape for
                // attention over the cached + new keys: `.none` for a
                // single new token, `.causal` for multi-token prefill
                // with offset 0, and an explicit array mask when the
                // cached history requires per-row masking.
                mask = blockCache.makeMask(n: length, windowSize: nil, returnArray: false)
            } else {
                attendKeys = k
                attendValues = v
                mask = .causal
            }

            let attnOut = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: attendKeys,
                values: attendValues,
                scale: attnScale,
                mask: mask
            )
            // (B, n_heads, L, head_dim) -> (B, L, rankQDim). Each
            // rank's attention output is sized to its rank-local head
            // slice and feeds the row-parallel `o_proj`, whose
            // `localInFeatures == rankQDim`. The o_proj precondition
            // checks the activation's last dim against
            // `block.self_attn.o_proj.rankInputDim`; surface a wiring
            // mismatch loudly here rather than as an opaque kernel
            // failure further in.
            precondition(
                block.self_attn.o_proj.rankInputDim == rankQDim,
                "row-parallel o_proj rankInputDim=" +
                "\(block.self_attn.o_proj.rankInputDim) does not match " +
                "the rank-local attention output width rankQDim=\(rankQDim)"
            )
            let attnFlat = attnOut.transposed(0, 2, 1, 3)
                .reshaped(batch, length, rankQDim)
            let attnFlat2D = attnFlat.reshaped(qkvBatch, rankQDim)
            let oOut = block.self_attn.o_proj(attnFlat2D)
                .reshaped(batch, length, hiddenSize)

            hidden = hidden + oOut

            // Pre-MLP RMSNorm + SwiGLU MLP.
            let preMLP = MLXFast.rmsNorm(
                hidden,
                weight: block.postAttentionLayernormWeight,
                eps: config.rmsNormEps
            )
            let preMLPFlat = preMLP.reshaped(qkvBatch, hiddenSize)

            let gate = block.mlp.gate_proj(preMLPFlat)
            let up = block.mlp.up_proj(preMLPFlat)
            let activated = MLXNN.silu(gate) * up
            // The SwiGLU output's last dim is the rank-local
            // intermediate slice and must match the row-parallel
            // `down_proj`'s input. A mismatch here would indicate a
            // shard-loading bug; surface it as a clear precondition
            // rather than as an opaque kernel failure.
            precondition(
                block.mlp.down_proj.rankInputDim == rankIntermediate,
                "row-parallel down_proj rankInputDim=" +
                "\(block.mlp.down_proj.rankInputDim) does not match the " +
                "rank-local SwiGLU output width rankIntermediate=" +
                "\(rankIntermediate)"
            )
            let down = block.mlp.down_proj(activated)
                .reshaped(batch, length, hiddenSize)

            hidden = hidden + down
        }

        // Final norm and lm_head projection. The lm_head matmul reuses
        // the embedding table directly rather than re-running the TQ
        // kernel — the kernel's output would be numerically identical
        // to the dequant trick already paid for at construction.
        let finalHidden = MLXFast.rmsNorm(
            hidden,
            weight: finalNormWeight,
            eps: config.rmsNormEps
        )
        let logits = finalHidden.matmul(embed_tokens.weight.transposed(1, 0))
        return logits
    }
}
