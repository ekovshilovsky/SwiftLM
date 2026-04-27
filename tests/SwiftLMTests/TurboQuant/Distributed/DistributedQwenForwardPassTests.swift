import XCTest
import Foundation
import MLX
@testable import TurboQuantKit

// Note: this file calls `MLX.eval(...)` — the MLX framework's lazy-graph
// materialisation entry point. It is unrelated to JavaScript / Python
// `eval` and never evaluates user-supplied source.

/// Phase 3 Task 13 acceptance test. Constructs both the size-1
/// `DistributedQwenModel` and the non-distributed
/// `TurboQuantSingleRankModel` reference oracle from the same Phase 3
/// fixture, runs prefill on a deterministic five-token prompt through
/// each, and asserts the two logits tensors agree within fp16 noise.
///
/// At `worldSize = 1` the column-parallel and row-parallel TurboQuant
/// wrappers both fall through to a whole-weight `TurboQuantShardedLinear`
/// kernel call (no shard, no collective), so the only sources of
/// divergence between the distributed forward path and the oracle are
/// (a) the placeholder weights that Task 12 left in the construction
/// path and (b) any choreography mismatch in the per-block forward
/// math. This test covers both.
///
/// Runtime cost. Both models pay a one-time embedding-table dequant
/// at construction (~100 s on the M5 Max in the cleanroom dev
/// environment). The 5-token prefill itself takes a few seconds. The
/// test therefore runs for several minutes; it is gated on the
/// fixture being installed locally so CI without the fixture skips
/// cleanly.
final class DistributedQwenForwardPassTests: XCTestCase {

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
        let hostPath = Bundle(for: DistributedQwenForwardPassTests.self)
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
                "DistributedQwenModel forward-pass equivalence requires the " +
                "Qwen2.5-Coder-3B-TQ8 model (Task 1a install)."
            )
        }
    }

    /// Size-1 distributed model produces logits within fp16 noise of
    /// the single-rank oracle on a deterministic prefill.
    func testSize1DistributedModelMatchesOracle() throws {
        try skipIfFixtureMissing()

        let metadataURL = Self.fixtureRoot.appendingPathComponent("tq_shard_metadata.json")
        let metadata = try ShardMetadata(jsonData: Data(contentsOf: metadataURL))

        // `DistributedGroup()` on a singleton process returns rank 0,
        // size 1 — the size-1 equivalence configuration.
        let group = DistributedGroup()
        let dist = try DistributedQwenModel(
            metadata: metadata,
            modelDir: Self.fixtureRoot,
            group: group
        )
        let ref = try TurboQuantSingleRankModel(directory: Self.fixtureRoot)

        // Five-token deterministic prompt. Covers prefill, RoPE
        // application across positions, and gives the equivalence
        // assertion a non-trivial logits tensor to compare. Token
        // ids stay within the first thousand entries of the Qwen2.5
        // tokenizer so they are guaranteed in-range.
        let tokenIds = MLXArray([1, 2, 3, 4, 5]).reshaped(1, 5)
        MLX.eval(tokenIds)

        let logitsDist = dist.callAsFunction(tokenIds)
        let logitsRef = ref.callAsFunction(tokenIds)
        MLX.eval(logitsDist, logitsRef)

        XCTAssertEqual(logitsDist.shape, logitsRef.shape,
                       "distributed and oracle logits must share shape; " +
                       "got dist=\(logitsDist.shape) ref=\(logitsRef.shape)")

        let diff32 = MLX.abs((logitsDist - logitsRef).asType(.float32))
        let refMax32 = MLX.abs(logitsRef.asType(.float32))
        let diff = MLX.max(diff32).item(Float.self)
        let refMax = MLX.max(refMax32).item(Float.self)
        print("[diag] dist vs ref maxAbsDiff=\(diff) refMax=\(refMax)")

        XCTAssertGreaterThan(refMax, 1.0,
                             "reference logits must be non-trivial; refMax=\(refMax)")
        // fp16 noise tolerance. The size-1 distributed path goes
        // through the same TQ kernel as the oracle but with the
        // wrappers' allSum/no-op graph nodes, which can introduce a
        // small reordering of fp16 accumulation. The plan budgets
        // 1e-2 for that noise.
        XCTAssertLessThan(diff, 1e-2,
                          "size-1 distributed model must match oracle within " +
                          "fp16 noise; maxAbsDiff=\(diff)")
    }
}
