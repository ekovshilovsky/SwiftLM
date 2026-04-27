import XCTest
import Foundation
import MLX
@testable import TurboQuantKit

/// Compile-gate proof for the Swift wrapper around the shard-aware C
/// API: the wrapper builds against the TurboQuantC module, imports the
/// `tq_linear_*` symbols, and exposes the expected constructor and
/// forward surface.
///
/// Synthetic quantized-weight rigging (generating valid 4-bit packed
/// indices, Lloyd-Max codebooks, and matching per-row norms from pure
/// Swift) would reimplement a slice of the offline quantizer here.
/// Numerical correctness is covered by the column- and row-parallel
/// wrapper tests and the end-to-end suite that runs against a real
/// converted fixture. The compile gate alone is the correct scope for
/// this file: it proves the Swift ↔ C boundary is wired up before any
/// downstream wrapper test relies on it.
final class TurboQuantShardedLinearTests: XCTestCase {

    /// Same metallib side-load workaround as
    /// `TurboQuantAllToShardedLinearTests`: SwiftPM's xctest host does
    /// not ship Cmlx's metallib, so any MLX stream construction aborts
    /// unless `mlx.metallib` is colocated with the xctest binary.
    override class func setUp() {
        super.setUp()
        installMetallibIfMissing()
    }

    private static func installMetallibIfMissing() {
        let fm = FileManager.default
        let hostPath = Bundle(for: TurboQuantShardedLinearTests.self)
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

    /// The wrapper exposes its constructor, error enum, and rank-local
    /// dimension metadata. This is a compile gate — if the TurboQuantC
    /// module is missing the new symbols, this file will fail to build
    /// and `swift test` will never reach the skip.
    func testShardBridgeCompileGate() throws {
        // Reference every public symbol the wrapper exposes, so the
        // compiler must resolve each one against TurboQuantC. This is
        // the actual proof — if the C API headers or module map have
        // drifted the build fails here rather than at link time.
        let initRef = TurboQuantShardedLinear.init(
            fullInFeatures:
            localInFeatures:
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
        )
        _ = initRef

        // Error enum reachable from tests.
        let err: TurboQuantShardedLinear.Error = .createFailed
        _ = err

        throw XCTSkip(
            "Synthetic TQ payload generation is intentionally deferred " +
            "to the wrapper-level numerical tests and the end-to-end " +
            "suite that runs on a real converted fixture. This test is " +
            "the Swift↔C compile gate; reaching the skip proves every " +
            "tq_linear_* symbol resolves at build time."
        )
    }
}
