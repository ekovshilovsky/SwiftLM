import XCTest
import Foundation
import MLX
@testable import TurboQuantKit

final class TurboQuantAllToShardedLinearTests: XCTestCase {

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
    override class func setUp() {
        super.setUp()
        installMetallibIfMissing()
    }

    private static func installMetallibIfMissing() {
        let fm = FileManager.default
        let hostPath = Bundle(for: TurboQuantAllToShardedLinearTests.self)
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

    /// Two rank-slice layers must produce the same concatenated
    /// output as a single full-weight matmul. Column-parallel
    /// doesn't allSum, so running two instances on a size-1 group
    /// with the two halves of W simulates a 2-rank split cleanly.
    func testColumnParallelConcatenationMatchesFullMatmul() throws {
        let inDim = 16
        let outDim = 32
        let batch = 4

        let weight = MLXRandom.normal([outDim, inDim])
        let x = MLXRandom.normal([batch, inDim])
        let fullOut = MLX.matmul(x, weight.transposed(-1, -2))

        let half = outDim / 2
        let w0 = weight[0 ..< half, 0 ..< inDim]
        let w1 = weight[half ..< outDim, 0 ..< inDim]

        let group = DistributedGroup()
        let layer0 = TurboQuantAllToShardedLinear(
            rankWeight: w0,
            rankOutputDim: half,
            rankInputDim: inDim,
            group: group
        )
        let layer1 = TurboQuantAllToShardedLinear(
            rankWeight: w1,
            rankOutputDim: half,
            rankInputDim: inDim,
            group: group
        )

        let out0 = layer0.callAsFunction(x)
        let out1 = layer1.callAsFunction(x)
        let reconstructed = MLX.concatenated([out0, out1], axis: 1)

        let diff = (reconstructed - fullOut).abs().max().item(Float.self)
        XCTAssertLessThan(diff, 1e-3,
            "column-parallel concat must match full matmul within tolerance")
    }

    /// The layer exposes rank-local dimensions through the
    /// TurboQuantShardedLayer protocol.
    func testExposesRankLocalDimensions() {
        let inDim = 8
        let outDim = 16
        let group = DistributedGroup()
        let layer = TurboQuantAllToShardedLinear(
            rankWeight: MLXRandom.normal([outDim, inDim]),
            rankOutputDim: outDim,
            rankInputDim: inDim,
            group: group
        )
        XCTAssertEqual(layer.rankOutputDim, outDim)
        XCTAssertEqual(layer.rankInputDim, inDim)
    }
}
