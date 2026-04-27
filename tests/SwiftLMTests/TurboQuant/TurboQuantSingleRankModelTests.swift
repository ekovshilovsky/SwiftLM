import XCTest
import Foundation
import MLX
@testable import TurboQuantKit

/// End-to-end forward-pass tests for the single-rank Qwen2 reference
/// model. The model is the size-1 equivalence oracle for the upcoming
/// `DistributedQwenModel` forward path, so this suite verifies it can
/// load a real TurboQuant-converted checkpoint, materialise the
/// embedding table, and produce finite logits with a sensible shape
/// on a small prompt. External-reference comparisons (HuggingFace
/// transformers, PyTorch) are out of scope for this iteration.
///
/// The fixture is large; the tests skip cleanly when it is absent.
final class TurboQuantSingleRankModelTests: XCTestCase {

    /// Phase 3 fixture root used by every model-level test.
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
        let hostPath = Bundle(for: TurboQuantSingleRankModelTests.self)
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
                "TurboQuantSingleRankModel forward-pass tests require the " +
                "Qwen2.5-Coder-3B-TQ8 model (Task 1a install)."
            )
        }
    }

    // MARK: - Test 1: forward on a small prompt

    /// Build the model from disk, run prefill on a 5-token prompt,
    /// and assert the output is shape-correct, finite, has non-trivial
    /// magnitude (a properly-loaded LM head produces logits well above
    /// noise), and produces in-vocabulary argmax tokens at every
    /// position. This is the principal acceptance test for the
    /// oracle: anything broken upstream of the lm_head matmul surfaces
    /// here as a NaN, all-zero, or out-of-range argmax.
    func testModelLoadsAndRunsForwardPassOnSmallPrompt() throws {
        try skipIfFixtureMissing()

        let model = try TurboQuantSingleRankModel(directory: Self.fixtureRoot)

        // Diagnostic: check the materialised embedding table is real
        // (non-zero, finite) before running the forward pass. A
        // silent-zero dequant would propagate to every downstream
        // layer and only surface as a degenerate argmax at the end;
        // checking here pins the failure to the dequant step.
        let table32 = model.embeddingTable.asType(.float32)
        MLX.eval(table32)
        let tableMaxAbs = MLX.max(MLX.abs(table32)).item(Float.self)
        let tableMeanAbs = MLX.mean(MLX.abs(table32)).item(Float.self)
        print("[diagnostic] embedding table: maxAbs=\(tableMaxAbs) meanAbs=\(tableMeanAbs)")
        XCTAssertTrue(tableMaxAbs.isFinite,
                      "embedding table must be finite; got maxAbs=\(tableMaxAbs)")
        XCTAssertGreaterThan(tableMaxAbs, 1e-3,
                             "embedding table looks like zeros; maxAbs=\(tableMaxAbs)")
        XCTAssertGreaterThan(tableMeanAbs, 1e-5,
                             "embedding table mean magnitude too small: meanAbs=\(tableMeanAbs)")

        // Five-token deterministic prompt. Token ids are within the
        // first thousand entries of the Qwen2.5 tokenizer's vocab so
        // the values are definitely valid; specific content is
        // irrelevant for the smoke test.
        let tokenIds = MLXArray([1, 2, 3, 4, 5]).reshaped(1, 5)
        MLX.eval(tokenIds)

        let logits = model(tokenIds)
        MLX.eval(logits)

        XCTAssertEqual(logits.shape, [1, 5, model.config.vocabSize],
                       "logits shape must be [batch=1, length=5, vocabSize]")

        let logits32 = logits.asType(.float32)
        MLX.eval(logits32)

        let maxAbs = MLX.max(MLX.abs(logits32)).item(Float.self)
        let meanAbs = MLX.mean(MLX.abs(logits32)).item(Float.self)
        print("[diagnostic] logits: maxAbs=\(maxAbs) meanAbs=\(meanAbs)")
        XCTAssertTrue(maxAbs.isFinite,
                      "logits must be finite; got maxAbs=\(maxAbs)")
        XCTAssertGreaterThan(maxAbs, 1.0,
                             "logits magnitude too low — lm_head likely silent: maxAbs=\(maxAbs)")

        // No NaN anywhere.
        let hasNaN = MLX.any(logits32 .!= logits32).item(Bool.self)
        XCTAssertFalse(hasNaN, "logits contain NaN values")

        // Argmax token ids are in-range for the vocab.
        let argmax = MLX.argMax(logits32, axis: -1)
        MLX.eval(argmax)
        let maxId = MLX.max(argmax).item(Int32.self)
        let minId = MLX.min(argmax).item(Int32.self)
        XCTAssertGreaterThanOrEqual(minId, 0,
                                    "argmax token id must be >= 0; got \(minId)")
        XCTAssertLessThan(maxId, Int32(model.config.vocabSize),
                          "argmax token id must be < vocab; got \(maxId)")
    }

    // MARK: - Test 2: embedding table determinism

    /// Construct two independent models from the same fixture and
    /// confirm their materialised embedding tables match within
    /// fp16 noise. Catches non-deterministic dequant behaviour
    /// (e.g. seed mismatch between runs) and any buffer-aliasing bug
    /// where the second construction's dequant overwrites the first.
    func testEmbeddingTableMaterialisationIsStable() throws {
        try skipIfFixtureMissing()

        let modelA = try TurboQuantSingleRankModel(directory: Self.fixtureRoot)
        let modelB = try TurboQuantSingleRankModel(directory: Self.fixtureRoot)

        let tableA = modelA.embeddingTable.asType(.float32)
        let tableB = modelB.embeddingTable.asType(.float32)
        MLX.eval(tableA, tableB)

        XCTAssertEqual(tableA.shape, tableB.shape,
                       "both embedding tables must have the same shape")

        let diff = MLX.abs(tableA - tableB)
        let maxDiff = MLX.max(diff).item(Float.self)
        print("[diagnostic] dequant determinism: maxAbsDiff=\(maxDiff)")

        // fp16 noise tolerance — bit-exact equality is not required
        // because the kernel may compute via fp16 accumulators whose
        // result rounding depends on transient buffer state. 1e-3 is
        // comfortably above fp16 ULP (~1e-3 near magnitude 1) and
        // well below the magnitude of real embedding rows.
        XCTAssertLessThan(maxDiff, 1e-3,
                          "embedding table dequant is non-deterministic: maxAbsDiff=\(maxDiff)")
    }
}
