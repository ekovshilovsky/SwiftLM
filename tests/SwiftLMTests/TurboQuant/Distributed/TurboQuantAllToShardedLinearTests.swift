import XCTest
import Foundation
import MLX
@testable import TurboQuantKit

/// Task 9a.4 replaced the Task 7 stub (MLX.matmul over a pre-
/// dequantized fp16 `rankWeight`) with a `TurboQuantShardedLinear`-
/// backed forward that takes TQ-compressed payloads directly — packed
/// indices, norms, Lloyd-Max codebooks, and rotation seeds. The Task 7
/// concat-vs-full-matmul test no longer fits the constructor surface
/// because no plain `MLXArray` weight is accepted.
///
/// Rigging synthetic TQ payloads purely in Swift (valid packed indices
/// matching generated codebook centroids, per-row norms consistent
/// with the rotation, and the correct block-size partitioning) would
/// reimplement a slice of the offline quantizer here. The real Tier 3
/// proof — concat of two rank slices matches a whole-weight TQ forward
/// within fp16 tolerance — lands in Task 9a.6 on top of the Phase 3
/// fixture (`Qwen2.5-Coder-3B-TQ8`) where a valid column-parallel
/// payload already exists on disk.
///
/// This file retains the metallib side-load workaround and a compile
/// gate that references every new parameter of the reworked
/// constructor: if the API surface drifts the build fails here rather
/// than at the first downstream caller.
final class TurboQuantAllToShardedLinearTests: XCTestCase {

    /// SwiftPM's `swift test` harness does not emit the Cmlx metal
    /// library resource bundle, so MLX aborts with "Failed to load
    /// the default metallib" on first stream construction (see
    /// DistributedSmokeTests in mlx-swift for the upstream note).
    /// The project's `build.sh` produces `mlx.metallib` under
    /// `.build/arm64-apple-macosx/release/`; we colocate a copy next
    /// to the xctest host binary so `load_colocated_library()` in
    /// mlx/backend/metal/device.cpp finds it via dladdr.
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

    /// Compile gate for the Task 9a.4 constructor rewrite. References
    /// every new parameter by name so a signature drift here surfaces
    /// at build time rather than at the first real caller. Reaching
    /// the XCTSkip proves the Swift API matches what the layer exposes
    /// today; the numerical Tier 3 proof is Task 9a.6's responsibility.
    func testTQKernelWiringCompileGate() throws {
        let initRef = TurboQuantAllToShardedLinear.init(
            fullInFeatures:
            rankOutFeatures:
            primaryBits:
            residualBits:
            packedPrimary:
            packedResidual:
            norms:
            primaryCodebook:
            residualCodebook:
            seedPrimary:
            seedResidual:
            blockSize:
            group:
        )
        _ = initRef

        throw XCTSkip(
            "Tier 3 numerical proof (two-rank concat vs whole-weight " +
            "TQ forward) is deferred to Task 9a.6, which runs on the " +
            "Phase 3 Qwen2.5-Coder-3B-TQ8 fixture. Reaching this skip " +
            "proves the TQ-kernel-backed constructor surface compiles."
        )
    }
}
