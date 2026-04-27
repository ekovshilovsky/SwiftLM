// Tensor-parallel Qwen2/Qwen3-family decoder-only model. Constructs
// a rank-local transformer stack from a `ShardMetadata` sidecar plus
// the safetensors files in the model directory.
//
// Layer-to-sharding mapping for each transformer block:
//
//   - `input_layernorm` (RMSNorm, pre-attention) — replicated.
//   - `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj` —
//     column-parallel, each a separate `TurboQuantAllToShardedLinear`.
//     The fixture ships split QKV (no fused tensor); each rank owns
//     its slice of the output rows of the three projections.
//   - `self_attn.o_proj` — row-parallel
//     (`TurboQuantShardedToAllLinear`). Consumes the rank-local
//     attention output and `allSum`s the partial back to the
//     replicated hidden state.
//   - `post_attention_layernorm` (RMSNorm, pre-MLP) — replicated.
//   - `mlp.gate_proj`, `mlp.up_proj` — column-parallel, each a
//     separate `TurboQuantAllToShardedLinear`. Split MLP layout, no
//     fused gate-up tensor in the fixture.
//   - `mlp.down_proj` — row-parallel (`TurboQuantShardedToAllLinear`).
//
// Top-level:
//
//   - `embed_tokens` — `ReplicatedEmbedding` (Phase 3 ships only the
//     replicated strategy; vocab-parallel can swap in later).
//   - `layers` — array of `DistributedQwenTransformerBlock` sized to
//     `config.numHiddenLayers`.
//   - final `RMSNorm` over the hidden dim.
//   - `lm_head` — `ReplicatedLMHead`. When `tieWordEmbeddings` is
//     true the lm_head shares its weight tensor with `embed_tokens`.
//
// Task 12 scope. This file lands the structural skeleton: the types,
// the per-layer wiring, and the construction-time loader that
// materialises every sharded layer from disk. The forward pass
// (`callAsFunction`) is intentionally absent — Task 13 wires it.
//
// Embedding-table loading. The Phase 3 fixture stores the embedding
// table in TurboQuant-packed form (no raw `embed_tokens.weight` row
// table on disk). Dequantising the V x H table to the rank-local
// embedding requires the TQ kernel which runs only at inference
// time, not during model construction. For Task 12 the embedding
// layer is constructed with a zero-filled placeholder matching the
// fixture's dtype and shape so the type-correctness assertions can
// run; Task 13 will replace the placeholder with the real
// dequantised table.

import Foundation
import MLX
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
/// layers that the transformer block dispatches into. The attention
/// math itself (RoPE, mask, scaled dot-product) is wired in Task 13.
public final class DistributedQwenAttention: Module {
    public let q_proj: TurboQuantAllToShardedLinear
    public let k_proj: TurboQuantAllToShardedLinear
    public let v_proj: TurboQuantAllToShardedLinear
    public let o_proj: TurboQuantShardedToAllLinear

    public init(
        q_proj: TurboQuantAllToShardedLinear,
        k_proj: TurboQuantAllToShardedLinear,
        v_proj: TurboQuantAllToShardedLinear,
        o_proj: TurboQuantShardedToAllLinear
    ) {
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        super.init()
    }
}

/// Sharded MLP sub-module (split gate / up / down). The SiLU gating
/// math is wired in Task 13.
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
/// Task 12 ships only the wiring; the forward pass is added in
/// Task 13.
public final class DistributedQwenTransformerBlock: Module {
    public let input_layernorm: RMSNorm
    public let self_attn: DistributedQwenAttention
    public let post_attention_layernorm: RMSNorm
    public let mlp: DistributedQwenMLP

    public init(
        input_layernorm: RMSNorm,
        self_attn: DistributedQwenAttention,
        post_attention_layernorm: RMSNorm,
        mlp: DistributedQwenMLP
    ) {
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp
        super.init()
    }
}

// MARK: - Top-level model

/// Tensor-parallel Qwen decoder model. Construction loads every
/// sharded layer from the model directory and assembles the block
/// stack. Task 12 stops here; Task 13 adds `callAsFunction`.
public final class DistributedQwenModel: Module {
    public let config: DistributedQwenConfiguration
    public let group: DistributedGroup
    public let rank: Int
    public let worldSize: Int

    public let embed_tokens: ReplicatedEmbedding
    public let layers: [DistributedQwenTransformerBlock]
    public let norm: RMSNorm
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

        // Embedding + lm_head.
        //
        // The Phase 3 fixture stores the embedding table in TQ-packed
        // form (no plain row table on disk). Dequantising it requires
        // the TQ kernel and is deferred to Task 13's forward path. A
        // zero-filled placeholder of the right shape and dtype keeps
        // construction-time invariants intact for the structural
        // skeleton; replace this with the real dequantised table when
        // wiring the forward pass.
        let embeddingShape = [config.vocabSize, config.hiddenSize]
        let embeddingPlaceholder = MLXArray.zeros(embeddingShape, dtype: .float16)
        let embed = ReplicatedEmbedding.make(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            weight: embeddingPlaceholder,
            strategy: .replicated,
            group: group
        )
        self.embed_tokens = embed

        // When `tie_word_embeddings` is true the LM head reuses the
        // embedding's row table; otherwise it has its own table on
        // disk. Task 13 will plug in the real load path; Task 12
        // mirrors the same placeholder so the type assertion holds
        // and the tied/untied branch is exercised.
        let lmHeadWeight: MLXArray
        if config.tieWordEmbeddings {
            lmHeadWeight = embeddingPlaceholder
        } else {
            lmHeadWeight = MLXArray.zeros(embeddingShape, dtype: .float16)
        }
        self.lm_head = ReplicatedLMHead.make(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            weight: lmHeadWeight,
            strategy: .replicated,
            group: group
        )

        // Final norm — replicated.
        let finalNormEntry = "model.norm.weight"
        // Verify the metadata claims this is a replicated tensor; its
        // weight content is loaded by Task 13 along with the other
        // RMSNorm parameters. The structural skeleton uses the
        // default unit weight from `RMSNorm.init`.
        if let entry = metadata.tensors[finalNormEntry] {
            precondition(
                entry.shardStrategy == .replicated,
                "model.norm.weight must be replicated; got \(entry.shardStrategy)"
            )
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Per-layer block construction. Every block loads its six TQ
        // linear layers (q/k/v + o, gate/up + down) from the
        // safetensors file the sidecar points at, applies the
        // appropriate rank slicing, and wraps the rank-local payloads
        // in the matching sharded-layer types.
        var blocks: [DistributedQwenTransformerBlock] = []
        blocks.reserveCapacity(config.numHiddenLayers)
        for layerIdx in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(layerIdx)"

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
                o_proj: oLayer
            )
            let mlp = DistributedQwenMLP(
                gate_proj: gateLayer,
                up_proj: upLayer,
                down_proj: downLayer
            )
            // RMSNorm parameters are loaded by Task 13 alongside the
            // forward path. Default unit weights stand in here so the
            // structural skeleton can be exercised by type-assertion
            // tests without running compute.
            let preAttn = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            let preMLP = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

            blocks.append(DistributedQwenTransformerBlock(
                input_layernorm: preAttn,
                self_attn: attn,
                post_attention_layernorm: preMLP,
                mlp: mlp
            ))
        }
        self.layers = blocks

        super.init()
    }
}
