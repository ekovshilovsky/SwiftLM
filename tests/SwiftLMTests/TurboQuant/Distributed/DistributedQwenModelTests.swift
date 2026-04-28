import XCTest
import Foundation
import MLX
import MLXNN
@testable import TurboQuantKit

/// Type-correctness assertions for the structural skeleton.
/// Constructs a `DistributedQwenModel` from the Qwen2.5-Coder-3B-TQ8
/// fixture and verifies that every per-layer slot ends up wrapping the
/// concrete sharded type expected by the split-QKV / split-MLP layout.
/// The forward pass is not exercised here; that is the forward-pass
/// test's territory.
///
/// The Qwen2.5-Coder-3B fixture is large (36 layers × 7 TQ tensors per layer),
/// so this test is gated on the fixture being installed locally.
/// When the sidecar is absent the test skips cleanly the same way
/// the end-to-end correctness tests do.
final class DistributedQwenModelTests: XCTestCase {

    override class func setUp() {
        super.setUp()
        installMetallibIfMissing()
    }

    // SwiftPM's xctest host does not emit the Cmlx metallib next to
    // the test binary. The other Distributed tests carry the same
    // workaround; copying it here keeps this file independent of the
    // end-to-end harness's class-level setup.
    private static func installMetallibIfMissing() {
        let fm = FileManager.default
        let hostPath = Bundle(for: DistributedQwenModelTests.self)
            .executablePath ?? CommandLine.arguments[0]
        let hostDir = URL(fileURLWithPath: hostPath).deletingLastPathComponent()
        let colocated = hostDir.appendingPathComponent("mlx.metallib")
        if fm.fileExists(atPath: colocated.path) { return }

        var searchDir = hostDir
        for _ in 0..<8 {
            let candidate = searchDir
                .appendingPathComponent("arm64-apple-macosx")
                .appendingPathComponent("release")
                .appendingPathComponent("mlx.metallib")
            if fm.fileExists(atPath: candidate.path) {
                try? fm.copyItem(at: candidate, to: colocated)
                return
            }
            searchDir.deleteLastPathComponent()
        }
    }

    private func resolvedFixtureRoot() throws -> URL {
        let root = try TurboQuantTestFixtures.requireQwenCoder3B()
        let configFile = root.appendingPathComponent("config.json")
        if !FileManager.default.fileExists(atPath: configFile.path) {
            throw XCTSkip(
                "Qwen2.5-Coder-3B fixture at \(root.path) is missing config.json. " +
                "DistributedQwenModel structural tests require a complete " +
                "Qwen2.5-Coder-3B-TQ8 layout."
            )
        }
        return root
    }

    /// Configuration parsing is itself part of the public surface; a
    /// dedicated assertion here makes the failure mode obvious if the
    /// upstream `config.json` ever drops a required field.
    func testConfigurationDecodesAllRequiredFields() throws {
        let fixtureRoot = try resolvedFixtureRoot()

        let configURL = fixtureRoot.appendingPathComponent("config.json")
        let config = try DistributedQwenConfiguration.load(from: configURL)

        // Verified against the upstream Qwen2.5-Coder-3B HuggingFace
        // config; pinning these here documents the fixture and would
        // surface any silent regression if the on-disk file changes.
        XCTAssertEqual(config.hiddenSize, 2048)
        XCTAssertEqual(config.numHiddenLayers, 36)
        XCTAssertEqual(config.intermediateSize, 11008)
        XCTAssertEqual(config.numAttentionHeads, 16)
        XCTAssertEqual(config.numKeyValueHeads, 2)
        XCTAssertEqual(config.vocabSize, 151936)
        XCTAssertTrue(config.tieWordEmbeddings,
                      "tied embeddings flag drives the lm_head weight reuse path")
    }

    /// Every per-layer slot must end up wrapping the concrete sharded
    /// type per the layout the model targets (split-QKV column-parallel
    /// into row-parallel `o_proj`; split-MLP column-parallel into
    /// row-parallel `down_proj`).
    func testModelConstructionWiresExpectedLayerTypes() throws {
        let fixtureRoot = try resolvedFixtureRoot()

        let sidecarURL = fixtureRoot.appendingPathComponent("tq_shard_metadata.json")
        let sidecarData = try Data(contentsOf: sidecarURL)
        let metadata = try ShardMetadata(jsonData: sidecarData)

        let configURL = fixtureRoot.appendingPathComponent("config.json")
        let config = try DistributedQwenConfiguration.load(from: configURL)

        // Share the materialised embedding table with the model so the
        // construction matches the path the equivalence tests
        // exercise. Construction-time wiring is independent of which
        // init is used; this avoids paying the dequant twice when the
        // suite runs.
        let embeddingTable = try materialiseEmbeddingTable(
            modelDir: fixtureRoot,
            metadata: metadata,
            embeddingLayerName: "model.embed_tokens",
            hiddenSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            primaryBits: 4,
            residualBits: 4
        )

        let group = DistributedGroup()
        let model = try DistributedQwenModel(
            metadata: metadata,
            modelDir: fixtureRoot,
            group: group,
            embeddingTable: embeddingTable
        )

        // Layer count matches the architecture config.
        XCTAssertEqual(model.layers.count, model.config.numHiddenLayers)
        XCTAssertEqual(model.layers.count, 36,
                       "Qwen2.5-Coder-3B has 36 transformer blocks")

        // Embedding + lm_head: the replicated peer types from
        // commit 19f233a.
        XCTAssertTrue(type(of: model.embed_tokens) == ReplicatedEmbedding.self,
                      "embed_tokens must be ReplicatedEmbedding")
        XCTAssertTrue(type(of: model.lm_head) == ReplicatedLMHead.self,
                      "lm_head must be ReplicatedLMHead")

        // Tied embeddings: the lm_head weight must be the same MLXArray
        // identity as the embedding weight when the config flag is
        // set. Comparing the underlying array shape and dtype isn't
        // enough — they must be the same instance.
        if model.config.tieWordEmbeddings {
            XCTAssertTrue(
                model.lm_head.weight === model.embed_tokens.weight,
                "tied embeddings must share the same MLXArray instance"
            )
        }

        // Spot-check the first block's slot types. The construction
        // loop wires every block identically, so checking layer 0
        // catches any wiring regression for all 36.
        let block0 = model.layers[0]

        XCTAssertTrue(type(of: block0.self_attn.q_proj) == TurboQuantAllToShardedLinear.self,
                      "self_attn.q_proj must be column-parallel (TurboQuantAllToShardedLinear)")
        XCTAssertTrue(type(of: block0.self_attn.k_proj) == TurboQuantAllToShardedLinear.self,
                      "self_attn.k_proj must be column-parallel (TurboQuantAllToShardedLinear)")
        XCTAssertTrue(type(of: block0.self_attn.v_proj) == TurboQuantAllToShardedLinear.self,
                      "self_attn.v_proj must be column-parallel (TurboQuantAllToShardedLinear)")
        XCTAssertTrue(type(of: block0.self_attn.o_proj) == TurboQuantShardedToAllLinear.self,
                      "self_attn.o_proj must be row-parallel (TurboQuantShardedToAllLinear)")

        XCTAssertTrue(type(of: block0.mlp.gate_proj) == TurboQuantAllToShardedLinear.self,
                      "mlp.gate_proj must be column-parallel (TurboQuantAllToShardedLinear)")
        XCTAssertTrue(type(of: block0.mlp.up_proj) == TurboQuantAllToShardedLinear.self,
                      "mlp.up_proj must be column-parallel (TurboQuantAllToShardedLinear)")
        XCTAssertTrue(type(of: block0.mlp.down_proj) == TurboQuantShardedToAllLinear.self,
                      "mlp.down_proj must be row-parallel (TurboQuantShardedToAllLinear)")

        // Pre-attention, pre-MLP, and final RMSNorm scales are stored
        // as raw bfloat16 weight tensors so the loaded scale survives
        // without an implicit dtype cast at construction. The forward
        // path applies them via `MLXFast.rmsNorm`.
        XCTAssertEqual(block0.inputLayernormWeight.shape, [model.config.hiddenSize],
                       "input_layernorm weight must be [hiddenSize]")
        XCTAssertEqual(block0.inputLayernormWeight.dtype, .bfloat16,
                       "input_layernorm weight must be bfloat16 (fixture dtype)")
        XCTAssertEqual(block0.postAttentionLayernormWeight.shape, [model.config.hiddenSize],
                       "post_attention_layernorm weight must be [hiddenSize]")
        XCTAssertEqual(block0.postAttentionLayernormWeight.dtype, .bfloat16,
                       "post_attention_layernorm weight must be bfloat16 (fixture dtype)")
        XCTAssertEqual(model.finalNormWeight.shape, [model.config.hiddenSize],
                       "model.finalNormWeight must be [hiddenSize]")
        XCTAssertEqual(model.finalNormWeight.dtype, .bfloat16,
                       "model.finalNormWeight must be bfloat16 (fixture dtype)")

        // q/k/v biases are replicated bfloat16 vectors sized to each
        // projection's full output dim. The fixture has q_proj
        // outputs of size 2048 and k/v outputs of size 256 (GQA).
        XCTAssertEqual(block0.self_attn.qBias.shape, [model.config.hiddenSize],
                       "q_proj bias must be [hiddenSize]")
        XCTAssertEqual(block0.self_attn.qBias.dtype, .bfloat16)
        XCTAssertEqual(
            block0.self_attn.kBias.shape,
            [model.config.numKeyValueHeads * (model.config.hiddenSize / model.config.numAttentionHeads)],
            "k_proj bias must be [numKVHeads * head_dim]"
        )
        XCTAssertEqual(block0.self_attn.kBias.dtype, .bfloat16)
        XCTAssertEqual(
            block0.self_attn.vBias.shape,
            [model.config.numKeyValueHeads * (model.config.hiddenSize / model.config.numAttentionHeads)],
            "v_proj bias must be [numKVHeads * head_dim]"
        )
        XCTAssertEqual(block0.self_attn.vBias.dtype, .bfloat16)
    }
}
