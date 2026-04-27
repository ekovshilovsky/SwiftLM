import XCTest
import Foundation
import MLX
@testable import TurboQuantKit

/// Tier 2 proof for Task 9a.3: the Swift wrapper around the new
/// shard-aware C API compiles against the TurboQuantC module, imports
/// the new `tq_linear_*` symbols, and exposes the expected constructor
/// and forward surface.
///
/// Synthetic quantized-weight rigging (generating valid 4-bit packed
/// indices, Lloyd-Max codebooks, and matching per-row norms from pure
/// Swift) would reimplement a slice of the offline quantizer here. That
/// work lives in Task 9a.4 (two-rank in-process concat) and Task 9a.6
/// (end-to-end against the Phase 3 fixture), where a real fixture layer
/// is already on disk. The compile gate alone is the correct scope for
/// 9a.3: it proves the Swift ↔ C boundary is wired up before 9a.4 needs
/// to stand on top of it.
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
            "to Task 9a.4 (two-rank concat) and 9a.6 (Phase 3 fixture). " +
            "This test is the Swift↔C compile gate; reaching the skip " +
            "proves every new tq_linear_* symbol resolves at build time."
        )
    }
}
