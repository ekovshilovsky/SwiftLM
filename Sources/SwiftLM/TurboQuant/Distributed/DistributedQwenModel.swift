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

    public let embed_tokens: ReplicatedEmbedding
    public let layers: [DistributedQwenTransformerBlock]
    /// Final RMSNorm scale, replicated and stored in the safetensors
    /// dtype (bfloat16 in the Qwen2.5-Coder-3B fixture). Applied via
    /// `MLXFast.rmsNorm` so the loaded scale survives without an
    /// implicit dtype cast.
    public let finalNormWeight: MLXArray
    public let lm_head: ReplicatedLMHead

    /// Build a rank-local distributed Qwen model.
    ///
    /// - Parameters:
    ///   - metadata: Decoded `tq_shard_metadata.json` sidecar.
    ///   - modelDir: Directory holding the safetensors files plus
    ///     the standard HuggingFace `config.json`.
    ///   - group: MLX distributed group; `group.rank` and
    ///     `group.size` drive shard selection.
    ///   - primaryBits / residualBits: TurboQuant bit widths. Default
    ///     to the 4+4 layout shipped by tq-convert; override only for
    ///     fixtures produced with a non-standard quantizer
    ///     configuration.
    public init(
        metadata: ShardMetadata,
        modelDir: URL,
        group: DistributedGroup,
        primaryBits: Int = 4,
        residualBits: Int = 4
    ) throws {
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try DistributedQwenConfiguration.load(from: configURL)
        self.config = config
        self.group = group
        let rank = group.rank
        let worldSize = group.size
        self.rank = rank
        self.worldSize = worldSize

        // Embedding table materialisation. The Qwen2.5-Coder-3B fixture stores
        // the embedding in TQ-packed form (no plain row table on
        // disk); the shared helper runs the kernel on an identity
        // input and transposes the result to recover the standard
        // `[vocabSize, hiddenSize]` row table. Both `embed_tokens`
        // and (when tied) `lm_head` reuse the same MLXArray instance
        // so identity checks on the tied path hold.
        //
        // The current implementation ships only the replicated-embedding strategy. A
        // future vocab-parallel implementation would materialise just
        // this rank's slice, but every rank still owns the full table
        // here.
        let embeddingTable = try materialiseEmbeddingTable(
            modelDir: modelDir,
            metadata: metadata,
            embeddingLayerName: "model.embed_tokens",
            hiddenSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            primaryBits: primaryBits,
            residualBits: residualBits
        )
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
        // payload; Qwen2.5-Coder-3B fixtures all ship tied embeddings, so the
        // untied branch is left as a precondition until needed.
        let lmHeadWeight: MLXArray
        if config.tieWordEmbeddings {
            lmHeadWeight = embeddingTable
        } else {
            preconditionFailure(
                "untied lm_head not yet supported by DistributedQwenModel; " +
                "config.tieWordEmbeddings=false but no separate load path is " +
                "wired"
            )
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
        for layerIdx in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(layerIdx)"

            // Replicated norm scales and q/k/v biases. All bfloat16
            // in the Qwen2.5-Coder-3B fixture; loaded as-is so the forward
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
            let qBias = try loadReplicatedTensor(
                modelDir: modelDir,
                metadata: metadata,
                tensorName: "\(prefix).self_attn.q_proj.bias",
                expectedDType: "BF16"
            )
            let kBias = try loadReplicatedTensor(
                modelDir: modelDir,
                metadata: metadata,
                tensorName: "\(prefix).self_attn.k_proj.bias",
                expectedDType: "BF16"
            )
            let vBias = try loadReplicatedTensor(
                modelDir: modelDir,
                metadata: metadata,
                tensorName: "\(prefix).self_attn.v_proj.bias",
                expectedDType: "BF16"
            )

            let qPayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).self_attn.q_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .columnParallel, rank: rank, worldSize: worldSize)
            let kPayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).self_attn.k_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .columnParallel, rank: rank, worldSize: worldSize)
            let vPayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).self_attn.v_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .columnParallel, rank: rank, worldSize: worldSize)
            let oPayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).self_attn.o_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .rowParallel, rank: rank, worldSize: worldSize)

            let gatePayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).mlp.gate_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .columnParallel, rank: rank, worldSize: worldSize)
            let upPayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).mlp.up_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .columnParallel, rank: rank, worldSize: worldSize)
            let downPayload = try loadShardedTQLayerPayload(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).mlp.down_proj",
                primaryBits: primaryBits, residualBits: residualBits,
                role: .rowParallel, rank: rank, worldSize: worldSize)

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
            // input slice; `fullInFeatures` is the original input dim
            // and feeds the kernel's combined_scale derivation.
            let oLayer = try TurboQuantShardedToAllLinear(
                fullInFeatures: oPayload.inFeatures * worldSize,
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
                group: group)

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
            let downLayer = try TurboQuantShardedToAllLinear(
                fullInFeatures: downPayload.inFeatures * worldSize,
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
                group: group)

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
        let numQHeads = config.numAttentionHeads
        let numKVHeads = config.numKeyValueHeads
        let headDim = hiddenSize / numQHeads
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
            // the subsequent reshape and RoPE.
            q = q + block.self_attn.qBias.asType(q.dtype)
            k = k + block.self_attn.kBias.asType(k.dtype)
            v = v + block.self_attn.vBias.asType(v.dtype)

            // Re-introduce the [batch, length] axes and split into
            // per-head slices. Layout follows the standard Qwen2
            // convention: (B, n_heads, L, head_dim). Cache entries
            // are stored in this same layout so the cache update can
            // drop straight in.
            q = q.reshaped(batch, length, numQHeads, headDim).transposed(0, 2, 1, 3)
            k = k.reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)
            v = v.reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)

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
            // (B, n_heads, L, head_dim) -> (B, L, n_heads * head_dim).
            let attnFlat = attnOut.transposed(0, 2, 1, 3)
                .reshaped(batch, length, numQHeads * headDim)
            let attnFlat2D = attnFlat.reshaped(qkvBatch, numQHeads * headDim)
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
