import XCTest
@testable import TurboQuantKit

final class ShardMetadataTests: XCTestCase {

    func testParsesMinimalMetadataFromJSON() throws {
        let json = #"""
        {
            "format_version": 1,
            "model_architecture": "qwen2",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "intermediate_size": 11008,
            "tensors": {
                "model.layers.0.self_attn.q_proj.weight": {
                    "shape": [3072, 2048],
                    "dtype": "tq8",
                    "file": "model-00001-of-00002.safetensors",
                    "byte_offset": 98304,
                    "byte_length": 6291456,
                    "shard_axis": 0,
                    "shard_strategy": "column_parallel",
                    "codebook_key": "model.layers.0.self_attn.q_proj.codebook"
                }
            }
        }
        """#

        let md = try ShardMetadata(jsonData: Data(json.utf8))
        XCTAssertEqual(md.formatVersion, 1)
        XCTAssertEqual(md.modelArchitecture, "qwen2")
        XCTAssertEqual(md.hiddenSize, 2048)
        XCTAssertEqual(md.numAttentionHeads, 16)
        XCTAssertEqual(md.intermediateSize, 11008)

        let q = try XCTUnwrap(md.tensors["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertEqual(q.shape, [3072, 2048])
        XCTAssertEqual(q.dtype, "tq8")
        XCTAssertEqual(q.byteOffset, 98304)
        XCTAssertEqual(q.byteLength, 6291456)
        XCTAssertEqual(q.shardAxis, 0)
        XCTAssertEqual(q.shardStrategy, .columnParallel)
        XCTAssertEqual(q.codebookKey, "model.layers.0.self_attn.q_proj.codebook")
    }

    func testReplicatedTensorHasNilShardAxis() throws {
        let json = #"""
        {
            "format_version": 1,
            "model_architecture": "qwen2",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "tensors": {
                "model.embed_tokens.weight": {
                    "shape": [152064, 2048],
                    "dtype": "tq8",
                    "file": "model-00001-of-00002.safetensors",
                    "byte_offset": 0,
                    "byte_length": 311427072,
                    "shard_axis": null,
                    "shard_strategy": "replicated"
                }
            }
        }
        """#
        let md = try ShardMetadata(jsonData: Data(json.utf8))
        let e = try XCTUnwrap(md.tensors["model.embed_tokens.weight"])
        XCTAssertNil(e.shardAxis)
        XCTAssertEqual(e.shardStrategy, .replicated)
    }

    func testRejectsUnknownShardStrategy() {
        let json = #"""
        {
            "format_version": 1, "model_architecture": "qwen2", "hidden_size": 0,
            "num_attention_heads": 0,
            "tensors": { "x": { "shape": [1], "dtype": "tq8", "file": "f", "byte_offset": 0, "byte_length": 0, "shard_axis": null, "shard_strategy": "bogus" } }
        }
        """#
        XCTAssertThrowsError(try ShardMetadata(jsonData: Data(json.utf8)))
    }

    /// Additional integration-ish test: load the real Phase 3 fixture if present.
    /// This exercises the decoder against actual convert-tool output.
    func testLoadsRealPhase3FixtureIfPresent() throws {
        let fixtureURL = URL(fileURLWithPath: "/Users/eugenekovshilovsky/Code/turboquant-mlx-models/converted/Qwen2.5-Coder-3B-TQ8/tq_shard_metadata.json")
        guard FileManager.default.fileExists(atPath: fixtureURL.path) else {
            throw XCTSkip("Phase 3 fixture not present: \(fixtureURL.path)")
        }
        let data = try Data(contentsOf: fixtureURL)
        let md = try ShardMetadata(jsonData: data)
        XCTAssertEqual(md.formatVersion, 1)
        XCTAssertEqual(md.modelArchitecture, "qwen2")
        XCTAssertEqual(md.hiddenSize, 2048)
        XCTAssertEqual(md.numAttentionHeads, 16)
        XCTAssertGreaterThan(md.tensors.count, 300, "Qwen2.5-Coder-3B has many tensors")
        let strategies = Set(md.tensors.values.map { $0.shardStrategy })
        XCTAssertTrue(strategies.contains(.columnParallel))
        XCTAssertTrue(strategies.contains(.rowParallel))
        XCTAssertTrue(strategies.contains(.replicated))
        XCTAssertFalse(strategies.contains(.expertParallel), "dense model has no experts")
    }
}
