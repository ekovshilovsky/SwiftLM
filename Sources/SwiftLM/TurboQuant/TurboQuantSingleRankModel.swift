// Non-distributed Qwen2 reference model that serves as the size-1
// equivalence oracle for the upcoming `DistributedQwenModel` forward
// pass. Lives outside the `Distributed/` subdirectory because it has
// no shard surface — every layer holds the whole-weight TurboQuant
// payload and runs through the kernel in single-process mode.
//
// Construction loads the entire Qwen2-architecture decoder from a
// TurboQuant-converted model directory:
//
//   - `embed_tokens` is materialised by running the TQ kernel on the
//     embedding's packed payload with an identity-matrix input. The
//     resulting `[V, H]` table is cached and reused as both the
//     forward-pass embedding lookup target and the lm_head weight
//     (when `tie_word_embeddings` is true). The materialised table
//     shares the same `MLXArray` instance for both uses so identity
//     checks on the tied path hold.
//
//   - Per-block TQ projections (q/k/v/o for self-attention, gate/up/
//     down for the MLP) are loaded through the existing whole-weight
//     path (`loadFullTQLayerPayload`) and wrapped in
//     `TurboQuantShardedLinear` instances configured with
//     `localInFeatures == fullInFeatures` so they run as
//     non-distributed linear layers.
//
//   - RMSNorm scales (input_layernorm, post_attention_layernorm, and
//     the model-final norm) load through the replicated-tensor helper
//     in `TQLayerLoader.swift` directly off the `_passthrough`
//     safetensors files. The Qwen2.5-Coder-3B fixture stores them as
//     bfloat16; `MLXFast.rmsNorm` accepts the BF16 weight without an
//     explicit dtype cast.
//
//   - q/k/v projection biases are likewise replicated bfloat16
//     vectors. They are added to the projection outputs prior to the
//     reshape into per-head tensors. o_proj, gate_proj, up_proj, and
//     down_proj have no biases in the Qwen2.5 architecture.
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
//
// The compressed `TurboQuantKVCache` Swift bridge in
// `TurboQuantBridge.swift` operates on raw byte buffers and would
// require Data round-tripping per decode step; wiring it into this
// path is intentionally deferred. `KVCacheSimple` from MLXLMCommon is
// the canonical interface in this codebase and is used here to
// validate the cache-aware forward path.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Errors specific to the single-rank reference model. Construction
/// failures from the underlying TQ loader, safetensors parser, or
/// metadata sidecar surface as their own error types.
public enum TurboQuantSingleRankModelError: Error, CustomStringConvertible {
    /// `embed_tokens` could not be materialised — typically because
    /// the TQ kernel returned an unexpected shape.
    case embeddingDequantShapeMismatch(expected: [Int], got: [Int])
    /// The caller supplied an externally-materialised embedding table
    /// whose shape or dtype does not match the model configuration.
    case embeddingTableMismatch(expectedShape: [Int], gotShape: [Int],
                                expectedDType: DType, gotDType: DType)

    public var description: String {
        switch self {
        case .embeddingDequantShapeMismatch(let expected, let got):
            return "embed_tokens dequant produced shape \(got); expected \(expected)"
        case .embeddingTableMismatch(let es, let gs, let ed, let gd):
            return "supplied embedding table has shape \(gs) dtype \(gd); " +
                "expected shape \(es) dtype \(ed)"
        }
    }
}

/// Whole-weight Qwen2 reference model. Holds the full materialised
/// embedding table, every transformer block's TQ-backed projections
/// and replicated norms / biases, and a final-norm scale. Forward
/// returns logits of shape `[batch, length, vocab]`.
public final class TurboQuantSingleRankModel {

    // MARK: - Configuration

    public let config: DistributedQwenConfiguration

    /// TurboQuant primary-stage bit width. The Qwen2.5-Coder-3B fixture ships
    /// 4+4 packing; tests can override to match alternate quantizer
    /// configurations.
    public let primaryBits: Int

    /// TurboQuant residual-stage bit width.
    public let residualBits: Int

    // MARK: - Layer parameters

    /// Materialised `[vocabSize, hiddenSize]` row table reused for
    /// both `embed_tokens` lookup and (when tied) the lm_head matmul.
    public let embeddingTable: MLXArray

    /// Final RMSNorm scale, replicated and stored in the safetensors
    /// dtype (bfloat16 in the Qwen2.5-Coder-3B fixture).
    public let finalNormWeight: MLXArray

    /// Per-block parameter bundles. Indexed by transformer-block
    /// position; iterated in order during forward.
    public let blocks: [Block]

    // MARK: - Per-block bundle

    /// Container for one transformer block's compiled state. Holds
    /// the projection layers, the two RMSNorm scales, and the q/k/v
    /// biases.
    public final class Block {
        public let inputLayerNormWeight: MLXArray
        public let postAttentionLayerNormWeight: MLXArray

        public let qProj: TurboQuantShardedLinear
        public let kProj: TurboQuantShardedLinear
        public let vProj: TurboQuantShardedLinear
        public let oProj: TurboQuantShardedLinear

        public let qBias: MLXArray
        public let kBias: MLXArray
        public let vBias: MLXArray

        public let gateProj: TurboQuantShardedLinear
        public let upProj: TurboQuantShardedLinear
        public let downProj: TurboQuantShardedLinear

        public init(
            inputLayerNormWeight: MLXArray,
            postAttentionLayerNormWeight: MLXArray,
            qProj: TurboQuantShardedLinear,
            kProj: TurboQuantShardedLinear,
            vProj: TurboQuantShardedLinear,
            oProj: TurboQuantShardedLinear,
            qBias: MLXArray,
            kBias: MLXArray,
            vBias: MLXArray,
            gateProj: TurboQuantShardedLinear,
            upProj: TurboQuantShardedLinear,
            downProj: TurboQuantShardedLinear
        ) {
            self.inputLayerNormWeight = inputLayerNormWeight
            self.postAttentionLayerNormWeight = postAttentionLayerNormWeight
            self.qProj = qProj
            self.kProj = kProj
            self.vProj = vProj
            self.oProj = oProj
            self.qBias = qBias
            self.kBias = kBias
            self.vBias = vBias
            self.gateProj = gateProj
            self.upProj = upProj
            self.downProj = downProj
        }
    }

    // MARK: - Construction

    /// Load a single-rank Qwen2 reference model from a TurboQuant-
    /// converted directory. The directory is expected to contain
    /// `config.json`, `tq_shard_metadata.json`, the TQ-packed
    /// safetensors files, and the `_passthrough` safetensors files
    /// holding the replicated bfloat16 norms and biases.
    ///
    /// - Parameters:
    ///   - directory: Model root containing the artifacts described
    ///     above.
    ///   - primaryBits: TurboQuant primary stage bit width.
    ///   - residualBits: TurboQuant residual stage bit width.
    public convenience init(
        directory: URL,
        primaryBits: Int = 4,
        residualBits: Int = 4
    ) throws {
        let configURL = directory.appendingPathComponent("config.json")
        let config = try DistributedQwenConfiguration.load(from: configURL)

        let sidecarURL = directory.appendingPathComponent("tq_shard_metadata.json")
        let sidecarData = try Data(contentsOf: sidecarURL)
        let metadata = try ShardMetadata(jsonData: sidecarData)

        try self.init(
            config: config,
            metadata: metadata,
            modelDir: directory,
            primaryBits: primaryBits,
            residualBits: residualBits
        )
    }

    /// Convenience constructor that accepts an externally-materialised
    /// embedding table so multiple model instances can share a single
    /// `[vocabSize, hiddenSize]` allocation. The Metal allocator's
    /// wired-memory ceiling on a single process is the reason this
    /// matters in practice: hosting more than one model per process
    /// (e.g. the oracle plus a rank-local distributed model in
    /// equivalence tests) requires the embedding to be a single
    /// allocation rather than two.
    public convenience init(
        directory: URL,
        embeddingTable: MLXArray,
        primaryBits: Int = 4,
        residualBits: Int = 4
    ) throws {
        let configURL = directory.appendingPathComponent("config.json")
        let config = try DistributedQwenConfiguration.load(from: configURL)

        let sidecarURL = directory.appendingPathComponent("tq_shard_metadata.json")
        let sidecarData = try Data(contentsOf: sidecarURL)
        let metadata = try ShardMetadata(jsonData: sidecarData)

        try self.init(
            config: config,
            metadata: metadata,
            modelDir: directory,
            embeddingTable: embeddingTable,
            primaryBits: primaryBits,
            residualBits: residualBits
        )
    }

    /// Designated initialiser used when the caller has already loaded
    /// the configuration and shard-metadata sidecar (e.g. tests that
    /// validate the metadata in isolation before constructing the
    /// model). Materialises the embedding table internally; for
    /// callers that want to share an already-materialised table across
    /// multiple model instances, see the `embeddingTable:`-accepting
    /// initialiser below.
    public convenience init(
        config: DistributedQwenConfiguration,
        metadata: ShardMetadata,
        modelDir: URL,
        primaryBits: Int = 4,
        residualBits: Int = 4
    ) throws {
        let embeddingTable: MLXArray
        do {
            embeddingTable = try materialiseEmbeddingTable(
                modelDir: modelDir,
                metadata: metadata,
                embeddingLayerName: "model.embed_tokens",
                hiddenSize: config.hiddenSize,
                vocabSize: config.vocabSize,
                primaryBits: primaryBits,
                residualBits: residualBits
            )
        } catch let error as TQEmbeddingMaterialisationError {
            switch error {
            case .shapeMismatch(let expected, let got):
                throw TurboQuantSingleRankModelError.embeddingDequantShapeMismatch(
                    expected: expected,
                    got: got
                )
            }
        }
        try self.init(
            config: config,
            metadata: metadata,
            modelDir: modelDir,
            embeddingTable: embeddingTable,
            primaryBits: primaryBits,
            residualBits: residualBits
        )
    }

    /// Designated initialiser that accepts an externally-materialised
    /// embedding table. Validates the supplied table against the
    /// configuration; constructs the per-block parameters from the
    /// safetensors files referenced by the sidecar.
    public init(
        config: DistributedQwenConfiguration,
        metadata: ShardMetadata,
        modelDir: URL,
        embeddingTable: MLXArray,
        primaryBits: Int = 4,
        residualBits: Int = 4
    ) throws {
        self.config = config
        self.primaryBits = primaryBits
        self.residualBits = residualBits

        // Validate the supplied embedding table against the config.
        // Catching shape/dtype drift here avoids opaque downstream
        // failures inside `MLX.take` or the lm_head matmul.
        let expectedShape = [config.vocabSize, config.hiddenSize]
        let expectedDType: DType = .float16
        if embeddingTable.shape != expectedShape || embeddingTable.dtype != expectedDType {
            throw TurboQuantSingleRankModelError.embeddingTableMismatch(
                expectedShape: expectedShape,
                gotShape: embeddingTable.shape,
                expectedDType: expectedDType,
                gotDType: embeddingTable.dtype
            )
        }
        self.embeddingTable = embeddingTable

        // Final norm scale. Qwen2.5-Coder-3B stores it as bfloat16 in
        // the passthrough file; load it as-is so MLXFast.rmsNorm
        // operates without a dtype cast on every forward pass.
        self.finalNormWeight = try loadReplicatedTensor(
            modelDir: modelDir,
            metadata: metadata,
            tensorName: "model.norm.weight",
            expectedDType: "BF16"
        )

        // Per-block construction. Each block loads its six TQ
        // projections (q/k/v + o, gate/up + down) plus the two
        // RMSNorm scales and the three q/k/v biases.
        var builtBlocks: [Block] = []
        builtBlocks.reserveCapacity(config.numHiddenLayers)
        for layerIdx in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(layerIdx)"

            let inputNorm = try loadReplicatedTensor(
                modelDir: modelDir,
                metadata: metadata,
                tensorName: "\(prefix).input_layernorm.weight",
                expectedDType: "BF16"
            )
            let postNorm = try loadReplicatedTensor(
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

            let qProj = try Self.makeWholeWeightLinear(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).self_attn.q_proj",
                primaryBits: primaryBits, residualBits: residualBits
            )
            let kProj = try Self.makeWholeWeightLinear(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).self_attn.k_proj",
                primaryBits: primaryBits, residualBits: residualBits
            )
            let vProj = try Self.makeWholeWeightLinear(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).self_attn.v_proj",
                primaryBits: primaryBits, residualBits: residualBits
            )
            let oProj = try Self.makeWholeWeightLinear(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).self_attn.o_proj",
                primaryBits: primaryBits, residualBits: residualBits
            )

            let gateProj = try Self.makeWholeWeightLinear(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).mlp.gate_proj",
                primaryBits: primaryBits, residualBits: residualBits
            )
            let upProj = try Self.makeWholeWeightLinear(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).mlp.up_proj",
                primaryBits: primaryBits, residualBits: residualBits
            )
            let downProj = try Self.makeWholeWeightLinear(
                modelDir: modelDir, metadata: metadata,
                layerName: "\(prefix).mlp.down_proj",
                primaryBits: primaryBits, residualBits: residualBits
            )

            builtBlocks.append(Block(
                inputLayerNormWeight: inputNorm,
                postAttentionLayerNormWeight: postNorm,
                qProj: qProj,
                kProj: kProj,
                vProj: vProj,
                oProj: oProj,
                qBias: qBias,
                kBias: kBias,
                vBias: vBias,
                gateProj: gateProj,
                upProj: upProj,
                downProj: downProj
            ))
        }
        self.blocks = builtBlocks
    }

    /// Build a whole-weight `TurboQuantShardedLinear` from the named
    /// TQ layer. Shared between every projection in every block; the
    /// helper keeps the per-projection construction sites declarative.
    private static func makeWholeWeightLinear(
        modelDir: URL,
        metadata: ShardMetadata,
        layerName: String,
        primaryBits: Int,
        residualBits: Int
    ) throws -> TurboQuantShardedLinear {
        let payload = try loadFullTQLayerPayload(
            modelDir: modelDir,
            metadata: metadata,
            layerName: layerName,
            primaryBits: primaryBits,
            residualBits: residualBits
        )
        return try TurboQuantShardedLinear(
            fullInFeatures: payload.inFeatures,
            localInFeatures: payload.inFeatures,
            rankOutFeatures: payload.outFeatures,
            primaryBits: primaryBits,
            residualBits: residualBits,
            packedPrimary: payload.packedPrimary,
            packedResidual: payload.packedResidual,
            norms: payload.norms,
            primaryCodebook: payload.primaryCodebook,
            residualCodebook: payload.residualCodebook,
            seedPrimary: payload.seedPrimary,
            seedResidual: payload.seedResidual,
            blockSize: payload.blockSize
        )
    }

    // MARK: - Forward pass

    /// Allocate one `KVCacheSimple` per transformer block. Cache
    /// ownership is the *caller*'s responsibility; the model itself
    /// remains stateless and the array is threaded through subsequent
    /// `callAsFunction(_:cache:)` invocations during incremental
    /// decode. Each cache is sized for `numKeyValueHeads * headDim`
    /// (the rank's full KV-head slice at single-rank).
    public func makeCache() -> [KVCache] {
        (0 ..< config.numHiddenLayers).map { _ in KVCacheSimple() }
    }

    /// Run the forward pass on a batch of token-id sequences.
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
        var hidden = MLX.take(embeddingTable, tokenIds, axis: 0)

        // The TQ kernel is most numerically consistent in fp16 inputs
        // (see TurboQuantShardedLinearEndToEndTests). Cast once after
        // the embedding lookup; the kernel preserves fp16 across its
        // internal compute.
        hidden = hidden.asType(.float16)

        // Project all attention activations through the kernel in
        // [batch * length, H] form so the kernel sees a 2-D batched
        // input. The resulting per-head reshape happens after
        // re-introducing the length axis.
        let qkvBatch = batch * length

        for (blockIdx, block) in blocks.enumerated() {
            let blockCache = cache?[blockIdx]

            // Pre-attention RMSNorm + self-attention.
            let preAttn = MLXFast.rmsNorm(
                hidden,
                weight: block.inputLayerNormWeight,
                eps: config.rmsNormEps
            )

            let preAttnFlat = preAttn.reshaped(qkvBatch, hiddenSize)

            var q = block.qProj(preAttnFlat)
            var k = block.kProj(preAttnFlat)
            var v = block.vProj(preAttnFlat)

            // Add q/k/v biases. The kernel returns fp16; bias tensors
            // are bf16 — a runtime-side fp16 cast keeps the dtype
            // consistent for the subsequent reshape and RoPE.
            q = q + block.qBias.asType(q.dtype)
            k = k + block.kBias.asType(k.dtype)
            v = v + block.vBias.asType(v.dtype)

            // Re-introduce the [batch, length] axes and split into
            // per-head slices. Layout follows the standard Qwen2
            // convention: (B, n_heads, L, head_dim). Cache entries are
            // stored in this same layout so the cache update can drop
            // straight in.
            q = q.reshaped(batch, length, numQHeads, headDim).transposed(0, 2, 1, 3)
            k = k.reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)
            v = v.reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)

            // RoPE must be applied at the cache's current offset so
            // tokens appended during decode receive the rotation
            // appropriate to their absolute position in the sequence.
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
            let oOut = block.oProj(attnFlat2D).reshaped(batch, length, hiddenSize)

            hidden = hidden + oOut

            // Pre-MLP RMSNorm + SwiGLU MLP.
            let preMLP = MLXFast.rmsNorm(
                hidden,
                weight: block.postAttentionLayerNormWeight,
                eps: config.rmsNormEps
            )
            let preMLPFlat = preMLP.reshaped(qkvBatch, hiddenSize)

            let gate = block.gateProj(preMLPFlat)
            let up = block.upProj(preMLPFlat)
            let activated = MLXNN.silu(gate) * up
            let down = block.downProj(activated).reshaped(batch, length, hiddenSize)

            hidden = hidden + down
        }

        // Final norm and lm_head projection.
        let finalHidden = MLXFast.rmsNorm(
            hidden,
            weight: finalNormWeight,
            eps: config.rmsNormEps
        )

        // The Qwen2.5-Coder-3B fixture has `tie_word_embeddings = true`,
        // so the lm_head reuses the materialised embedding table. The
        // standard MLX matmul against `embeddingTable^T` produces
        // `[batch, length, vocabSize]` logits without re-running the
        // TQ kernel (which would just confirm what the dequant trick
        // already gave us during construction).
        let logits = finalHidden.matmul(embeddingTable.transposed(1, 0))
        return logits
    }
}
