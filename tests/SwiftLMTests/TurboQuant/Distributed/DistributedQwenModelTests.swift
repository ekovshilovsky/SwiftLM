import XCTest
import Foundation
import MLX
import MLXNN
@testable import TurboQuantKit

/// Type-correctness assertions for the Task 12 structural skeleton.
/// Constructs a `DistributedQwenModel` from the Phase 3 Qwen2.5-Coder-3B-TQ8
/// fixture and verifies that every per-layer slot ends up wrapping the
/// concrete sharded type expected by the §4.1 layout (split-QKV /
/// split-MLP variant). The forward pass is not exercised here; that
/// is Task 13's territory.
///
/// The Phase 3 fixture is large (36 layers × 7 TQ tensors per layer),
/// so this test is gated on the fixture being installed locally.
/// When the sidecar is absent the test skips cleanly the same way
/// the end-to-end correctness tests do.
final class DistributedQwenModelTests: XCTestCase {

    /// Fixture root used by every Tier-4 / structural test on this
    /// model.
    private static let fixtureRoot = URL(fileURLWithPath:
        "/Users/eugenekovshilovsky/Code/turboquant-mlx-models/converted/Qwen2.5-Coder-3B-TQ8"
    )

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

    private func skipIfFixtureMissing() throws {
        let fm = FileManager.default
        let sidecar = Self.fixtureRoot.appendingPathComponent("tq_shard_metadata.json")
        let configFile = Self.fixtureRoot.appendingPathComponent("config.json")
        if !fm.fileExists(atPath: sidecar.path) || !fm.fileExists(atPath: configFile.path) {
            throw XCTSkip(
                "Phase 3 fixture not present at \(Self.fixtureRoot.path). " +
                "DistributedQwenModel structural tests require the " +
                "Qwen2.5-Coder-3B-TQ8 model (Task 1a install)."
            )
        }
    }

    /// Configuration parsing is itself part of the public surface; a
    /// dedicated assertion here makes the failure mode obvious if the
    /// upstream `config.json` ever drops a required field.
    func testConfigurationDecodesAllRequiredFields() throws {
        try skipIfFixtureMissing()

        let configURL = Self.fixtureRoot.appendingPathComponent("config.json")
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

    /// The structural skeleton: every per-layer slot must end up
    /// wrapping the concrete sharded type per the layout the model
    /// targets at Phase 3 (split-QKV column-parallel into row-parallel
    /// `o_proj`; split-MLP column-parallel into row-parallel
    /// `down_proj`).
    func testModelConstructionWiresExpectedLayerTypes() throws {
        try skipIfFixtureMissing()

        let sidecarURL = Self.fixtureRoot.appendingPathComponent("tq_shard_metadata.json")
        let sidecarData = try Data(contentsOf: sidecarURL)
        let metadata = try ShardMetadata(jsonData: sidecarData)

        let group = DistributedGroup()
        let model = try DistributedQwenModel(
            metadata: metadata,
            modelDir: Self.fixtureRoot,
            group: group
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

        // Pre-attention and pre-MLP norms are replicated RMSNorms.
        XCTAssertTrue(type(of: block0.input_layernorm) == RMSNorm.self,
                      "input_layernorm must be a replicated RMSNorm")
        XCTAssertTrue(type(of: block0.post_attention_layernorm) == RMSNorm.self,
                      "post_attention_layernorm must be a replicated RMSNorm")

        // Final norm.
        XCTAssertTrue(type(of: model.norm) == RMSNorm.self,
                      "model.norm must be a replicated RMSNorm")
    }
}
