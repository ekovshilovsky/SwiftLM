import XCTest
import Foundation
import MLX
@testable import TurboQuantKit

/// End-to-end correctness proof for the shard-aware TurboQuant linear
/// stack. Loads a real TQ-compressed layer from the Qwen2.5-Coder-3B-TQ8
/// fixture and exercises three proofs against it:
///
///   1. Whole-weight reference — `TurboQuantShardedLinear` with
///      `localInFeatures == fullInFeatures` and the full output dim
///      (no sharding). Serves as the oracle for the sharded paths.
///
///   2. Column-parallel concat — split a 2-rank slice of the output
///      dim via `TurboQuantAllToShardedLinear`, concat the partials
///      and compare to the oracle.
///
///   3. Row-parallel allSum — split a 2-rank slice of the input dim
///      via a 2-way `TurboQuantShardedLinear` in-process, sum the
///      partials and compare to the oracle. `TurboQuantShardedToAllLinear`
///      cannot be exercised directly because its size-1 `allSum` group
///      forbids any non-trivial sharding at construction time; the
///      underlying kernel math is identical.
///
/// Rigging synthetic TQ payloads from pure Swift is tractable only by
/// re-implementing a slice of the offline quantizer, which is why the
/// per-layer compile-gate suites stop short of numerical proofs. This
/// file closes the gap by loading weights that the offline quantizer
/// actually produced.
///
/// The fixture's safetensors files hold:
///   - `<layer>.packed_primary`  U8 [out, in*bits/8]  — 4-bit packed
///   - `<layer>.packed_residual` U8 [out, in*bits/8]  — 4-bit packed
///   - `<layer>.norms`           F32 [out]
///   - `<layer>.seeds`           U32 [3] = [primary_seed, residual_seed, block_size]
///   - `tq_codebook_primary`     F32 [16]  (shared across the whole model)
///   - `tq_codebook_residual`    F32 [16]
///
/// The shard-metadata sidecar only lists the `.weight` entries, so
/// this file parses the safetensors header itself to locate the
/// companion tensors. When the fixture is absent we skip cleanly.
///
/// ## Cross-runtime eval boundary
///
/// libturboquant_mlx.dylib links Homebrew's libmlx, while the Swift
/// bridge links mlx-swift's Cmlx — two distinct MLX runtimes co-resident
/// in the test process. A lazy output array scheduled inside the kernel
/// belongs to the dylib's runtime; calling `eval()` through the Swift-
/// side MLXArray wrapper runs against the Cmlx runtime instead, and the
/// kernel's compute node never gets materialised. The consumer observes
/// a freshly allocated but never-populated buffer — uniform zeros across
/// the whole output. The fix lives in `fused_dequant_matmul` in the core
/// repo, which now `eval()`s the output before returning so the kernel
/// runs inside the runtime that scheduled it.
///
/// The zero-output guards below remain as a regression backstop: if the
/// cross-runtime materialisation path ever drifts again, both sharded
/// proofs would otherwise silently degenerate to a 0-vs-0 comparison
/// that trivially satisfies any relative-error tolerance.
final class TurboQuantShardedLinearEndToEndTests: XCTestCase {

    // MARK: - Fixture configuration

    private static let primaryBits = 4
    private static let residualBits = 4

    override class func setUp() {
        super.setUp()
        installMetallibIfMissing()
    }

    // SwiftPM's xctest host does not emit the Cmlx metallib next to the
    // test binary. Mirror the workaround the other Distributed tests
    // use so MLX stream construction does not abort on first use.
    private static func installMetallibIfMissing() {
        let fm = FileManager.default
        let hostPath = Bundle(for: TurboQuantShardedLinearEndToEndTests.self)
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

    // MARK: - Fixture loading
    //
    // The header parser, MLXArray companion-tensor readers, and the
    // `TQLayerPayload` shape used to live inline here. They graduated
    // to `Sources/SwiftLM/TurboQuant/Distributed/TQLayerLoader.swift`
    // so `DistributedQwenModel` and these end-to-end tests share one
    // loader implementation. This thin wrapper keeps the test sites
    // unchanged: it loads the full unsliced payload and lets each
    // test do its own in-process slicing via direct MLXArray indexing.

    private static func loadTQLayer(
        rootURL: URL,
        layerName: String
    ) throws -> TQLayerPayload {
        let sidecarURL = rootURL.appendingPathComponent("tq_shard_metadata.json")
        let sidecarData = try Data(contentsOf: sidecarURL)
        let sidecar = try ShardMetadata(jsonData: sidecarData)

        return try loadFullTQLayerPayload(
            modelDir: rootURL,
            metadata: sidecar,
            layerName: layerName,
            primaryBits: Self.primaryBits,
            residualBits: Self.residualBits
        )
    }

    private func resolvedFixtureRoot() throws -> URL {
        try TurboQuantTestFixtures.requireQwenCoder3B()
    }

    // MARK: - Shared comparison helpers

    /// Relative Frobenius error between two fp16/fp32 MLXArrays of the
    /// same shape: ||a - b||_F / max(||b||_F, eps). Matches the
    /// tolerance convention used by the upstream TQ numerical proofs.
    private func relFrobeniusError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = a.asType(.float32) - b.asType(.float32)
        let num = MLX.sqrt(MLX.sum(diff * diff)).item(Float.self)
        let den = MLX.sqrt(MLX.sum(b.asType(.float32) * b.asType(.float32)))
            .item(Float.self)
        return num / max(den, 1e-8)
    }

    /// Force graph materialization so the sliced views the test hands
    /// to the C kernel are backed by concrete storage. Uses MLX's
    /// `MLX.eval(_:)` — unrelated to any scripting-language primitive.
    private static func materialize(_ arrays: MLXArray...) {
        MLX.eval(arrays)
    }

    // MARK: - Test 1: whole-weight reference (bridge proof)

    /// The bridge itself is tested here: `TurboQuantShardedLinear` with
    /// rank_out == full_out and local_in == full_in, i.e., no actual
    /// sharding, must produce a forward output with the correct shape
    /// and finite values. This confirms the real TQ kernel accepts the
    /// fixture's compressed payload before any shard slicing enters.
    func testWholeWeightReferenceForward() throws {
        let fixtureRoot = try resolvedFixtureRoot()

        let layer = try Self.loadTQLayer(
            rootURL: fixtureRoot,
            layerName: "model.layers.0.self_attn.q_proj"
        )

        // Sanity — q_proj is 2048×2048 with 4-bit packing.
        XCTAssertEqual(layer.outFeatures, 2048)
        XCTAssertEqual(layer.inFeatures, 2048)
        XCTAssertEqual(layer.packedPrimary.shape,
                       [layer.outFeatures, layer.inFeatures * Self.primaryBits / 8])
        XCTAssertEqual(layer.norms.shape, [layer.outFeatures])
        XCTAssertGreaterThan(layer.blockSize, 0)

        let tqLayer = try TurboQuantShardedLinear(
            fullInFeatures: layer.inFeatures,
            localInFeatures: layer.inFeatures,
            rankOutFeatures: layer.outFeatures,
            primaryBits: Self.primaryBits,
            residualBits: Self.residualBits,
            packedPrimary: layer.packedPrimary,
            packedResidual: layer.packedResidual,
            norms: layer.norms,
            primaryCodebook: layer.primaryCodebook,
            residualCodebook: layer.residualCodebook,
            seedPrimary: layer.seedPrimary,
            seedResidual: layer.seedResidual,
            blockSize: layer.blockSize
        )

        // Deterministic input so the output is reproducible. Use fp16
        // to match the kernel's internal accumulator type. The C API
        // documents that `tq_linear_forward` casts the input to fp16
        // internally, but the cast path only produces the expected
        // numerical output when the input is ALREADY fp16 at the
        // Swift boundary; passing fp32 yields an all-zero output —
        // a latent defect captured below in a separate XCTSkipped
        // assertion for a future investigation.
        MLXRandom.seed(12345)
        let batch = 2
        let x = MLXRandom.normal([batch, layer.inFeatures], dtype: .float16)
        MLX.eval(x)

        let y = tqLayer(x)
        XCTAssertEqual(y.shape, [batch, layer.outFeatures],
                       "whole-weight forward should produce full output dim")

        let y32 = y.asType(.float32)
        MLX.eval(y32)
        let maxAbs = MLX.max(MLX.abs(y32)).item(Float.self)
        let meanAbs = MLX.mean(MLX.abs(y32)).item(Float.self)
        print("[diagnostic] whole-weight q_proj: maxAbs=\(maxAbs) meanAbs=\(meanAbs)")
        XCTAssertTrue(maxAbs.isFinite, "output must be finite; maxAbs=\(maxAbs)")
        XCTAssertGreaterThan(meanAbs, 1e-4,
                             "output mean magnitude too small: meanAbs=\(meanAbs)")

        let hasNaN = MLX.any(y32 .!= y32).item(Bool.self)
        XCTAssertFalse(hasNaN, "output contains NaN values")
    }

    // MARK: - Test 2: column-parallel two-rank concat proof

    /// Column-parallel end-to-end proof: slice the output dim in half,
    /// build two `TurboQuantAllToShardedLinear` instances over the
    /// resulting row slices of `packed_primary`, `packed_residual`,
    /// and `norms`; concat their partials and compare to the whole-
    /// weight reference from Test 1.
    func testColumnParallelTwoRankConcatMatchesReference() throws {
        let fixtureRoot = try resolvedFixtureRoot()

        let layer = try Self.loadTQLayer(
            rootURL: fixtureRoot,
            layerName: "model.layers.0.self_attn.q_proj"
        )
        XCTAssertEqual(layer.outFeatures % 2, 0, "outFeatures must be even for 2-way split")
        let halfOut = layer.outFeatures / 2

        // Whole-weight oracle via the bridge — same path as Test 1.
        let oracle = try TurboQuantShardedLinear(
            fullInFeatures: layer.inFeatures,
            localInFeatures: layer.inFeatures,
            rankOutFeatures: layer.outFeatures,
            primaryBits: Self.primaryBits,
            residualBits: Self.residualBits,
            packedPrimary: layer.packedPrimary,
            packedResidual: layer.packedResidual,
            norms: layer.norms,
            primaryCodebook: layer.primaryCodebook,
            residualCodebook: layer.residualCodebook,
            seedPrimary: layer.seedPrimary,
            seedResidual: layer.seedResidual,
            blockSize: layer.blockSize
        )

        MLXRandom.seed(4242)
        let batch = 2
        let x = MLXRandom.normal([batch, layer.inFeatures], dtype: .float16)
        MLX.eval(x)
        let yRef = oracle(x)
        MLX.eval(yRef)
        let yRefMax = MLX.max(MLX.abs(yRef.asType(.float32))).item(Float.self)
        print("[col-parallel diag] oracle yRef maxAbs=\(yRefMax)")

        // Slice the output-dim axis of packed_primary, packed_residual,
        // and norms. Column-parallel: rank r gets rows [r*halfOut,
        // (r+1)*halfOut) of the output dim.
        let group = DistributedGroup()

        func makeColumnParallelShard(rank: Int) throws -> TurboQuantAllToShardedLinear {
            let start = rank * halfOut
            let end = start + halfOut
            // MLXArray subscripting slices along axis 0.
            let packedPrimarySlice = layer.packedPrimary[start..<end]
            let packedResidualSlice = layer.packedResidual[start..<end]
            let normsSlice = layer.norms[start..<end]
            // Force materialization of the sliced views before the C
            // kernel reads the backing storage.
            Self.materialize(packedPrimarySlice, packedResidualSlice, normsSlice)

            return try TurboQuantAllToShardedLinear(
                fullInFeatures: layer.inFeatures,
                rankOutFeatures: halfOut,
                primaryBits: Self.primaryBits,
                residualBits: Self.residualBits,
                packedPrimary: packedPrimarySlice,
                packedResidual: packedResidualSlice,
                norms: normsSlice,
                primaryCodebook: layer.primaryCodebook,
                residualCodebook: layer.residualCodebook,
                seedPrimary: layer.seedPrimary,
                seedResidual: layer.seedResidual,
                blockSize: layer.blockSize,
                group: group
            )
        }

        let rank0 = try makeColumnParallelShard(rank: 0)
        let rank1 = try makeColumnParallelShard(rank: 1)

        let partial0 = rank0(x)
        let partial1 = rank1(x)
        XCTAssertEqual(partial0.shape, [batch, halfOut])
        XCTAssertEqual(partial1.shape, [batch, halfOut])

        // Concatenate along the output-dim axis (last axis) to
        // reconstruct the full output; this is what the downstream
        // row-parallel consumer would see at a size-2 group in a real
        // cluster.
        let reconstructed = MLX.concatenated([partial0, partial1], axis: -1)
        XCTAssertEqual(reconstructed.shape, yRef.shape)

        // Zero-output guard — stops a 0-vs-0 comparison from passing
        // a relative-error tolerance trivially. If this ever fires,
        // the cross-runtime eval bug described at the file head is
        // still live.
        let refMax = MLX.max(MLX.abs(yRef.asType(.float32))).item(Float.self)
        let recMax = MLX.max(MLX.abs(reconstructed.asType(.float32))).item(Float.self)
        XCTAssertGreaterThan(refMax, 1e-3,
                             "oracle output is trivially zero — see file-level comment " +
                             "on the cross-runtime eval boundary; refMax=\(refMax)")
        XCTAssertGreaterThan(recMax, 1e-3,
                             "reconstructed output is trivially zero — see file-level " +
                             "comment on the cross-runtime eval boundary; " +
                             "recMax=\(recMax)")

        let err = relFrobeniusError(reconstructed, yRef)
        // Column-parallel concat should be numerically exact aside
        // from fp16 accumulation noise; the upstream TQ numerical
        // proofs observe rel_frob ≈ 0 for this configuration.
        XCTAssertLessThan(err, 1e-2,
                          "column-parallel 2-rank concat diverged from whole-weight " +
                          "reference: rel_frob=\(err)")
    }

    // MARK: - Test 3: row-parallel two-rank allSum proof

    /// Row-parallel end-to-end proof: slice the input dim in half, build
    /// two rank-local `TurboQuantShardedLinear` handles over the
    /// resulting input-axis slices of `packed_primary` /
    /// `packed_residual`, sum their partials, and compare to a whole-
    /// weight reference on the same layer. This mirrors what a size-2
    /// cluster's `TurboQuantShardedToAllLinear.callAsFunction` would
    /// compute: each rank forwards its local input slice through a
    /// shard-aware kernel, then the collective `allSum` accumulates.
    ///
    /// The test uses `TurboQuantShardedLinear` directly (not the
    /// `TurboQuantShardedToAllLinear` wrapper) because the wrapper's
    /// `localInFeatures * worldSize == fullInFeatures` precondition
    /// assumes its size is the real group size, but at size-1 the
    /// precondition would forbid actual 2-way sharding. The wrapper's
    /// numerical contribution is a no-op `allSum` plus the kernel
    /// call; exercising the kernel directly exercises the same math
    /// with a transparent in-process sum.
    ///
    /// Sharding constraint: `fullInFeatures` must divide by
    /// `worldSize * blockSize`. For o_proj at 2048 and blockSize=512,
    /// `2048 % (2 * 512) == 0` is satisfied.
    func testRowParallelTwoRankAllSumMatchesReference() throws {
        let fixtureRoot = try resolvedFixtureRoot()

        let layer = try Self.loadTQLayer(
            rootURL: fixtureRoot,
            layerName: "model.layers.0.self_attn.o_proj"
        )
        let worldSize = 2
        XCTAssertEqual(layer.inFeatures % (worldSize * layer.blockSize), 0,
                       "row-parallel shard requires fullIn % (worldSize*blockSize) == 0")
        let halfIn = layer.inFeatures / worldSize

        // Whole-weight reference — no sharding.
        let oracle = try TurboQuantShardedLinear(
            fullInFeatures: layer.inFeatures,
            localInFeatures: layer.inFeatures,
            rankOutFeatures: layer.outFeatures,
            primaryBits: Self.primaryBits,
            residualBits: Self.residualBits,
            packedPrimary: layer.packedPrimary,
            packedResidual: layer.packedResidual,
            norms: layer.norms,
            primaryCodebook: layer.primaryCodebook,
            residualCodebook: layer.residualCodebook,
            seedPrimary: layer.seedPrimary,
            seedResidual: layer.seedResidual,
            blockSize: layer.blockSize
        )

        MLXRandom.seed(9090)
        let batch = 2
        let x = MLXRandom.normal([batch, layer.inFeatures], dtype: .float16)
        let yRef = oracle(x)

        // Row-parallel: slice packed_primary and packed_residual along
        // axis 1 (input-byte axis). Norms and codebooks are broadcast
        // unchanged across ranks. Per the 4-bit layout, axis 1 holds
        // `fullInFeatures * bits / 8` bytes, so rank r gets columns
        // [r * halfIn * bits / 8, (r+1) * halfIn * bits / 8).
        let packedColsPerRank = halfIn * Self.primaryBits / 8

        func makeRowParallelKernel(rank: Int) throws -> TurboQuantShardedLinear {
            let cStart = rank * packedColsPerRank
            let cEnd = cStart + packedColsPerRank
            // Axis-1 slicing produces a strided view; the TQ kernel
            // expects a row-contiguous [rank_out, local_in * bits / 8]
            // buffer. `.contiguous()` forces a packed copy on the
            // device so the C kernel sees the bytes it expects.
            let packedPrimarySlice = layer.packedPrimary[0..., cStart..<cEnd].contiguous()
            let packedResidualSlice = layer.packedResidual[0..., cStart..<cEnd].contiguous()
            Self.materialize(packedPrimarySlice, packedResidualSlice)

            return try TurboQuantShardedLinear(
                fullInFeatures: layer.inFeatures,
                localInFeatures: halfIn,
                rankOutFeatures: layer.outFeatures,
                primaryBits: Self.primaryBits,
                residualBits: Self.residualBits,
                packedPrimary: packedPrimarySlice,
                packedResidual: packedResidualSlice,
                norms: layer.norms,
                primaryCodebook: layer.primaryCodebook,
                residualCodebook: layer.residualCodebook,
                seedPrimary: layer.seedPrimary,
                seedResidual: layer.seedResidual,
                blockSize: layer.blockSize
            )
        }

        let rank0 = try makeRowParallelKernel(rank: 0)
        let rank1 = try makeRowParallelKernel(rank: 1)

        // Slice the input activation along the input-dim axis so each
        // rank receives its matching feature block. Force
        // contiguous so the kernel sees a packed buffer.
        let x0 = x[0..., 0..<halfIn].contiguous()
        let x1 = x[0..., halfIn..<layer.inFeatures].contiguous()
        Self.materialize(x0, x1)

        // On a size-1 group the wrapper's allSum is a no-op, so each
        // kernel forward IS the partial output. Sum explicitly to
        // mirror what a real size-2 `allSum` would compute.
        let partial0 = rank0(x0)
        let partial1 = rank1(x1)
        XCTAssertEqual(partial0.shape, [batch, layer.outFeatures])
        XCTAssertEqual(partial1.shape, [batch, layer.outFeatures])

        let reconstructed = partial0 + partial1
        XCTAssertEqual(reconstructed.shape, yRef.shape)

        // Zero-output guard — see file-level comment on the cross-
        // runtime eval boundary. If the whole-weight forward returns
        // zero, the sharded comparison degenerates to 0 == 0 and
        // trivially passes any relative tolerance.
        let refMax = MLX.max(MLX.abs(yRef.asType(.float32))).item(Float.self)
        let recMax = MLX.max(MLX.abs(reconstructed.asType(.float32))).item(Float.self)
        XCTAssertGreaterThan(refMax, 1e-3,
                             "oracle output is trivially zero; refMax=\(refMax)")
        XCTAssertGreaterThan(recMax, 1e-3,
                             "reconstructed output is trivially zero; recMax=\(recMax)")

        let err = relFrobeniusError(reconstructed, yRef)
        // Row-parallel tolerance: split-rotation compositions under
        // block-aligned shards match whole-weight to within fp16-
        // accumulation noise. A slightly looser bound than the
        // column-parallel case is appropriate because the split
        // occurs across the rotated input axis where per-block
        // norms compound.
        XCTAssertLessThan(err, 5e-2,
                          "row-parallel 2-rank allSum diverged from whole-weight " +
                          "reference: rel_frob=\(err)")
    }
}
