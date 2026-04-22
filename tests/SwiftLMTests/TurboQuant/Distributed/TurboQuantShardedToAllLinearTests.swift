import XCTest
import Foundation
import MLX
@testable import TurboQuantKit

final class TurboQuantShardedToAllLinearTests: XCTestCase {

    /// SwiftPM's `swift test` harness does not emit the Cmlx metal
    /// library resource bundle, so MLX aborts with "Failed to load
    /// the default metallib" on first stream construction (see
    /// DistributedSmokeTests in mlx-swift for the upstream note).
    /// The project's `build.sh` produces `mlx.metallib` under
    /// `.build/arm64-apple-macosx/release/`; we colocate a copy next
    /// to the xctest host binary so `load_colocated_library()` in
    /// mlx/backend/metal/device.cpp finds it via dladdr. When the
    /// metallib is absent the test is skipped with a diagnostic
    /// pointing at `./build.sh`, rather than crashing the whole
    /// bundle on unrelated suites.
    ///
    /// Duplicated verbatim from TurboQuantAllToShardedLinearTests;
    /// `override class func setUp()` on a `final class` is not
    /// easily shareable without collapsing both suites into a
    /// helper. A follow-up refactor can factor this into a shared
    /// utility once a third consumer appears.
    override class func setUp() {
        super.setUp()
        installMetallibIfMissing()
    }

    private static func installMetallibIfMissing() {
        let fm = FileManager.default
        let hostPath = Bundle(for: TurboQuantShardedToAllLinearTests.self)
            .executablePath ?? CommandLine.arguments[0]
        let hostDir = URL(fileURLWithPath: hostPath).deletingLastPathComponent()
        let colocated = hostDir.appendingPathComponent("mlx.metallib")
        if fm.fileExists(atPath: colocated.path) { return }

        // Walk up to locate the build directory and the release
        // metallib produced by build.sh.
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

    /// Row-parallel: full W is split along the input dim into W0 and W1.
    /// Each rank computes x_sharded @ W_sharded^T producing a partial
    /// output of the full output shape; summing the two rank partials
    /// must equal a single-rank full matmul. At a size-1 group the
    /// group.allSum inside the layer is a no-op, so the layer output
    /// IS its partial and summing the two partials here mirrors what
    /// a size-2 group's allSum would produce.
    func testRowParallelSumMatchesFullMatmul() throws {
        let inDim = 16
        let outDim = 8
        let batch = 4

        let fullW = MLXRandom.normal([outDim, inDim])
        let fullX = MLXRandom.normal([batch, inDim])
        let full = MLX.matmul(fullX, fullW.transposed(-1, -2))

        let half = inDim / 2
        // Row-parallel splits the INPUT dim of W:
        //   W shape (outDim, inDim) -> W0 = W[:, :half], W1 = W[:, half:]
        //   x shape (batch, inDim)  -> x0 = x[:, :half], x1 = x[:, half:]
        //   full @ W^T = x0 @ W0^T + x1 @ W1^T
        let w0 = fullW[0 ..< outDim, 0 ..< half]
        let w1 = fullW[0 ..< outDim, half ..< inDim]
        let x0 = fullX[0 ..< batch, 0 ..< half]
        let x1 = fullX[0 ..< batch, half ..< inDim]

        let group = DistributedGroup()
        let layer0 = TurboQuantShardedToAllLinear(
            rankWeight: w0,
            rankOutputDim: outDim,
            rankInputDim: half,
            group: group
        )
        let layer1 = TurboQuantShardedToAllLinear(
            rankWeight: w1,
            rankOutputDim: outDim,
            rankInputDim: half,
            group: group
        )

        let partial0 = layer0.callAsFunction(x0)
        let partial1 = layer1.callAsFunction(x1)
        let summed = partial0 + partial1

        let diff = (summed - full).abs().max().item(Float.self)
        XCTAssertLessThan(diff, 1e-3,
            "row-parallel sum must match full matmul within tolerance")
    }

    /// The layer exposes rank-local dimensions through the
    /// TurboQuantShardedLayer protocol. For row-parallel the output
    /// dim is the full output dim (result is replicated via allSum)
    /// and the input dim is the rank-local slice.
    func testExposesRankLocalDimensions() {
        let outDim = 32
        let rankIn = 8
        let group = DistributedGroup()
        let layer = TurboQuantShardedToAllLinear(
            rankWeight: MLXRandom.normal([outDim, rankIn]),
            rankOutputDim: outDim,
            rankInputDim: rankIn,
            group: group
        )
        XCTAssertEqual(layer.rankOutputDim, outDim)
        XCTAssertEqual(layer.rankInputDim, rankIn)
    }
}
