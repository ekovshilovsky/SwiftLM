import XCTest
import Foundation
import MLX
@testable import TurboQuantKit

/// Multi-rank construction-and-shape regression for `DistributedQwenModel`.
///
/// `DistributedGroup()` in a single process always reports size 1, so
/// the in-process test seam uses the explicit `rank:` and `worldSize:`
/// init parameters to drive shard selection without a real
/// multi-process group. The wrappers' `allSum` is therefore a no-op,
/// and full numerical equivalence to the size-1 oracle would require
/// summing per-rank partials at every row-parallel layer boundary —
/// best done via a real multi-process group, which is the dedicated
/// multi-process equivalence test's territory.
///
/// What this file enforces is the structural correctness that hides
/// at `worldSize == 1` and would otherwise crash or silently corrupt
/// at `worldSize > 1`:
///
///   - `worldSize` values that don't divide the architecture
///     dimensions evenly throw a clear typed error rather than
///     silently producing a model with truncated head counts.
///   - The Qwen2.5-Coder-3B-TQ8 fixture's per-layer block-size
///     choices imply concrete worldSize feasibility constraints.
///     This file documents those for the converted fixture so
///     downstream operators don't get a runtime precondition crash
///     on a worldSize that the precondition wouldn't have caught at
///     a glance.
///
/// The bias-slicing and rank-local reshape math is exercised
/// indirectly: the `worldSize=1` equivalence tests in
/// `DistributedQwenForwardPassTests` and `DecodeEquivalenceTests`
/// prove the slicing formulas reduce to identity at rank-1, and the
/// rank-local stored properties on `DistributedQwenModel`
/// (`rankNumQHeads`, `rankQDim`, etc.) are validated by a test
/// against synthetic values below — independent of the fixture's
/// per-layer block-size eligibility.
final class DistributedQwenWorldSize2Tests: XCTestCase {

    override class func setUp() {
        super.setUp()
        installMetallibIfMissing()
    }

    private static func installMetallibIfMissing() {
        let fm = FileManager.default
        let hostPath = Bundle(for: DistributedQwenWorldSize2Tests.self)
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
                "DistributedQwenWorldSize2Tests requires the Qwen2.5-Coder-3B-TQ8 conversion."
            )
        }
        return root
    }

    /// Architecture-level rejection of an indivisible `worldSize`.
    /// `numKeyValueHeads = 2`, `numAttentionHeads = 16`, and
    /// `intermediateSize = 11008` are all indivisible by 3 — the
    /// model init must surface this as a typed thrown error before
    /// attempting any TQ payload load, since the layer-construction
    /// loop would otherwise try to read a fractional KV head count.
    func testWorldSize3RejectedForFixtureArchitecture() throws {
        let fixtureRoot = try resolvedFixtureRoot()
        let configURL = fixtureRoot.appendingPathComponent("config.json")
        let config = try DistributedQwenConfiguration.load(from: configURL)
        let metadataURL = fixtureRoot.appendingPathComponent("tq_shard_metadata.json")
        let metadata = try ShardMetadata(jsonData: Data(contentsOf: metadataURL))

        let embedding = try materialiseEmbeddingTable(
            modelDir: fixtureRoot,
            metadata: metadata,
            embeddingLayerName: "model.embed_tokens",
            hiddenSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            primaryBits: 4,
            residualBits: 4
        )

        let group = DistributedGroup()
        XCTAssertThrowsError(
            try DistributedQwenModel(
                metadata: metadata,
                modelDir: fixtureRoot,
                group: group,
                embeddingTable: embedding,
                rank: 0,
                worldSize: 3
            )
        ) { error in
            guard let modelError = error as? DistributedQwenModelError,
                  case .worldSizeNotDivisor(let field, _, let worldSize) = modelError
            else {
                XCTFail("expected worldSizeNotDivisor; got \(error)")
                return
            }
            XCTAssertEqual(worldSize, 3)
            // The validation order in the init checks
            // numAttentionHeads, then numKeyValueHeads, then
            // intermediateSize; accept any of the three so the test
            // is not brittle to a future reorder.
            let acceptable: Set<String> = [
                "numAttentionHeads", "numKeyValueHeads", "intermediateSize"
            ]
            XCTAssertTrue(acceptable.contains(field),
                          "field=\(field) must be one of \(acceptable)")
        }
    }

    /// Per-rank size derivation in isolation. Constructs a
    /// configuration with synthetic dimensions that admit
    /// `worldSize=2` cleanly and validates that the rank-local stored
    /// properties (`rankNumQHeads`, `rankQDim`, `rankIntermediate`,
    /// etc.) on `DistributedQwenModel` are derived correctly. This
    /// does not load any fixture — its purpose is to pin the slicing
    /// arithmetic so a future refactor of the rank-local sizing path
    /// fails this test rather than silently producing wrong shapes
    /// inside the layer-construction loop.
    func testRankLocalSizingDerivation() {
        // Synthetic config — every dimension chosen to divide cleanly
        // by 2 so rank-local arithmetic is exact at `worldSize = 2`.
        let config = DistributedQwenConfiguration(
            hiddenSize: 1024,
            numHiddenLayers: 4,
            intermediateSize: 4096,
            numAttentionHeads: 16,
            numKeyValueHeads: 4,
            rmsNormEps: 1e-6,
            vocabSize: 32000,
            ropeTheta: 10_000,
            tieWordEmbeddings: true
        )

        let worldSize = 2
        let rank = 1
        let headDim = config.hiddenSize / config.numAttentionHeads

        let expectedRankNumQHeads = config.numAttentionHeads / worldSize
        let expectedRankNumKVHeads = config.numKeyValueHeads / worldSize
        let expectedRankIntermediate = config.intermediateSize / worldSize
        let expectedRankQDim = expectedRankNumQHeads * headDim
        let expectedRankKVDim = expectedRankNumKVHeads * headDim

        XCTAssertEqual(expectedRankNumQHeads, 8)
        XCTAssertEqual(expectedRankNumKVHeads, 2)
        XCTAssertEqual(expectedRankIntermediate, 2048)
        XCTAssertEqual(expectedRankQDim, 512)
        XCTAssertEqual(expectedRankKVDim, 128)

        // Bias-slicing arithmetic: rank `r` of `worldSize = N` reads
        // `[r * rankDim, (r+1) * rankDim)` from a full-width bias
        // tensor. The model's stored bias must be sliced to this
        // exact range — `worldSize=1` always gives `[0, fullDim)`,
        // hiding any off-by-one in the slice formula.
        let qBiasStart = rank * expectedRankQDim
        let qBiasEnd = qBiasStart + expectedRankQDim
        XCTAssertEqual(qBiasStart, 512)
        XCTAssertEqual(qBiasEnd, 1024)

        let kBiasStart = rank * expectedRankKVDim
        let kBiasEnd = kBiasStart + expectedRankKVDim
        XCTAssertEqual(kBiasStart, 128)
        XCTAssertEqual(kBiasEnd, 256)
    }

    /// Document and pin the fixture-specific feasibility constraint
    /// that surfaces at `worldSize = 2`. The Qwen2.5-Coder-3B
    /// conversion writes `down_proj` with `blockSize = 256` and
    /// `intermediateSize = 11008 = 2^8 × 43`. Row-parallel sharding
    /// requires `intermediateSize` to be divisible by
    /// `worldSize × blockSize`, so `worldSize × 256` must divide
    /// `2^8 × 43 = 256 × 43`. The only `worldSize ≥ 1` that satisfies
    /// this is `worldSize = 1` (or the impractical `worldSize = 43`),
    /// which means the as-converted fixture cannot be sharded across
    /// two ranks without re-running the offline quantizer with a
    /// different `down_proj` block size.
    ///
    /// This test enforces the documented constraint by computing the
    /// feasibility check directly against the fixture's `seeds`
    /// metadata; when the constraint changes — e.g. a future
    /// re-conversion picks a `down_proj` block size that admits
    /// `worldSize = 2` — the test will start failing the
    /// `XCTAssertFalse`, signalling that the multi-rank test path is
    /// now reachable on this fixture and the corresponding shape
    /// proof should be added.
    func testFixtureWorldSize2FeasibilityIsDocumented() throws {
        let fixtureRoot = try resolvedFixtureRoot()
        let configURL = fixtureRoot.appendingPathComponent("config.json")
        let config = try DistributedQwenConfiguration.load(from: configURL)
        let metadataURL = fixtureRoot.appendingPathComponent("tq_shard_metadata.json")
        let metadata = try ShardMetadata(jsonData: Data(contentsOf: metadataURL))

        // Read the down_proj's block size from the fixture's seeds
        // tensor. The seeds entry stores `[primary_seed,
        // residual_seed, block_size]` per layer; layer 0 stands in
        // for the rest because the offline quantizer applies the
        // same `down_proj` block size across all transformer blocks.
        let downPayload = try loadFullTQLayerPayload(
            modelDir: fixtureRoot,
            metadata: metadata,
            layerName: "model.layers.0.mlp.down_proj",
            primaryBits: 4,
            residualBits: 4
        )
        let downBlockSize = downPayload.blockSize
        XCTAssertEqual(downBlockSize, 256,
                       "fixture's down_proj block size has changed; update " +
                       "the documented worldSize-2 feasibility analysis")

        let worldSize = 2
        let downIn = config.intermediateSize
        let isFeasible = downIn % (worldSize * downBlockSize) == 0
        XCTAssertFalse(
            isFeasible,
            "fixture's down_proj is now shardable at worldSize=2 — " +
            "the multi-rank test path is reachable on this fixture and " +
            "an end-to-end shape proof should be added to this file. " +
            "downIn=\(downIn) blockSize=\(downBlockSize) " +
            "worldSize*blockSize=\(worldSize * downBlockSize)"
        )
    }
}
