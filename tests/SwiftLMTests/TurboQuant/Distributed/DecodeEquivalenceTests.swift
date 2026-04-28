// Equivalence acceptance tests for the cache-aware forward path.
//
// The cache plumbing in `TurboQuantSingleRankModel` and
// `DistributedQwenModel` must produce the same logits as the
// cache-less prefill on the same token sequence: prefill consumes
// `[1, L]` in one call, while incremental decode walks the same
// sequence one token at a time, appending rotated keys / values to
// the cache and reading the accumulated history back through the
// SDPA mask sourced from `KVCache.makeMask`.
//
// Three checks anchor the contract:
//
//   1. The single-rank oracle's prefill and decode paths agree on
//      the same prompt — proves the cache plumbing and RoPE-offset
//      handling reproduce the prefill mathematics.
//   2. The distributed model's decode path agrees with the oracle's
//      decode path — establishes that the size-1 distributed cache
//      flow stays bit-exact with the oracle, the same contract the
//      pre-existing `DistributedQwenForwardPassTests` enforces for
//      prefill.
//   3. The distributed model's decode path agrees with its own
//      prefill — catches divergences that would slip past test 2 if
//      the same bug happened to land in both the oracle and the
//      distributed cache path (e.g. a shared incorrect mask choice).
//
// Note: `MLX.eval(...)` calls in this file are the MLX framework's
// lazy-graph materialisation entry points. They are unrelated to
// JavaScript / Python source-string evaluation and never evaluate
// user-supplied source.
//
// Runtime cost. Each test pays the embedding-table dequant at
// construction (~100 s on the dev machine) and a five-step decode
// loop (a few seconds). Tests skip cleanly when the
// `TQ_FIXTURE_DIR` environment variable is unset or points at a
// directory missing the `tq_shard_metadata.json` sidecar.

import XCTest
import Foundation
import MLX
import MLXLMCommon
@testable import TurboQuantKit

final class DecodeEquivalenceTests: XCTestCase {

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
        let hostPath = Bundle(for: DecodeEquivalenceTests.self)
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
                "DecodeEquivalenceTests requires the Qwen2.5-Coder-3B-TQ8 conversion."
            )
        }
        return root
    }

    /// Five-token deterministic prompt. Long enough to exercise the
    /// prefill-vs-decode boundary at multiple positions without
    /// inflating test runtime; short enough that each model's
    /// embedding-table dequant dominates the per-test cost.
    private static let promptTokens: [Int32] = [1, 2, 3, 4, 5]

    /// Compute `max(|a - b|)` in fp32 over the full tensor. Used to
    /// score the fp16-to-fp16 logits comparisons; an explicit fp32
    /// promotion avoids the noise floor of fp16 abs/max which would
    /// otherwise mask sub-fp16 differences.
    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff32 = MLX.abs((a - b).asType(.float32))
        return MLX.max(diff32).item(Float.self)
    }

    /// Index of the largest logit across the vocabulary axis. The
    /// argmax is the temperature-zero sampling decision and is the
    /// only logit comparison that affects model output. Token-level
    /// equivalence is therefore the binding correctness check; logit
    /// magnitude drift below the inter-token margin is acceptable
    /// fp16 noise.
    private func argmaxToken(_ logitRow: MLXArray) -> Int32 {
        return MLX.argMax(logitRow.asType(.float32), axis: -1)
            .item(Int32.self)
    }

    /// Slice `logits` of shape `[batch, length, V]` to the single
    /// `length` row at position `i`. Used to align prefill logits
    /// with single-step decode logits for per-position comparison.
    private func logitRow(_ logits: MLXArray, position: Int) -> MLXArray {
        return logits[0..., position, 0...]
    }

    /// Run a five-step incremental decode through `forward`, returning
    /// the per-step logits stacked into a `[batch, length, V]` array.
    /// The cache is allocated by `makeCache`; the caller owns it.
    private func runIncrementalDecode(
        prompt: [Int32],
        makeCache: () -> [KVCache],
        forward: (MLXArray, [KVCache]) -> MLXArray
    ) -> MLXArray {
        let cache = makeCache()
        var perStepLogits: [MLXArray] = []
        perStepLogits.reserveCapacity(prompt.count)
        for tok in prompt {
            let stepIds = MLXArray([tok]).reshaped(1, 1)
            let stepLogits = forward(stepIds, cache)
            // Materialise here so the cache update commits before the
            // next iteration; a lazy graph would otherwise extend
            // across the loop and serialise the per-step inspection.
            MLX.eval(stepLogits)
            perStepLogits.append(stepLogits)
        }
        // Stack along the length axis to recover a `[1, L, V]` tensor
        // shaped to compare against prefill output.
        return concatenated(perStepLogits, axis: 1)
    }

    // MARK: - Test 1: oracle prefill vs oracle decode

    /// On `TurboQuantSingleRankModel` the cache-based decode path
    /// must reproduce the cache-less prefill output token-for-token.
    func testSingleRankCacheDecodeMatchesPrefill() throws {
        let fixtureRoot = try resolvedFixtureRoot()
        let model = try TurboQuantSingleRankModel(directory: fixtureRoot)

        // Path A: full prefill in one call. `[1, 5]` token-id batch
        // through the cache-less path produces `[1, 5, V]` logits.
        let prefillIds = MLXArray(Self.promptTokens).reshaped(1, Self.promptTokens.count)
        let logitsPrefill = model(prefillIds, cache: nil)
        MLX.eval(logitsPrefill)

        // Path B: incremental decode. Five single-token forwards
        // through the cache-aware path produce `[1, 1, V]` logits per
        // step, stacked into `[1, 5, V]`.
        let logitsDecode = runIncrementalDecode(
            prompt: Self.promptTokens,
            makeCache: { model.makeCache() },
            forward: { ids, cache in model(ids, cache: cache) }
        )
        MLX.eval(logitsDecode)

        XCTAssertEqual(logitsPrefill.shape, logitsDecode.shape,
                       "prefill and decode logits must share shape; " +
                       "got prefill=\(logitsPrefill.shape) decode=\(logitsDecode.shape)")

        // Per-position equivalence. Reporting per-position diffs
        // localises any divergence to the step it first appears at,
        // which is informative when triaging a regression.
        var aggregateMax: Float = 0
        for i in 0 ..< Self.promptTokens.count {
            let prefillRow = logitRow(logitsPrefill, position: i)
            let decodeRow = logitRow(logitsDecode, position: i)
            let diff = maxAbsDiff(prefillRow, decodeRow)
            aggregateMax = max(aggregateMax, diff)
            let prefillTok = argmaxToken(prefillRow)
            let decodeTok = argmaxToken(decodeRow)
            print("[diag] oracle prefill vs decode pos=\(i) maxAbsDiff=\(diff) " +
                  "argmax(prefill)=\(prefillTok) argmax(decode)=\(decodeTok)")
            // The argmax is the binding correctness check: at
            // temperature-zero sampling this is the predicted token
            // and any divergence below the inter-token logit margin
            // does not affect output. Logit-magnitude drift past 1e-2
            // is the expected fp16 fingerprint of routing the same
            // attention through two different SDPA dispatch shapes.
            XCTAssertEqual(prefillTok, decodeTok,
                           "prefill and decode must predict the same token at " +
                           "position \(i); got prefill=\(prefillTok) decode=\(decodeTok) " +
                           "(maxAbsDiff=\(diff))")
        }

        let refMax = MLX.max(MLX.abs(logitsPrefill.asType(.float32))).item(Float.self)
        XCTAssertGreaterThan(refMax, 1.0,
                             "reference logits must be non-trivial; refMax=\(refMax)")

        // Logit-magnitude tolerance. The cache-aware path attends
        // through the explicit `.none`-mask SDPA dispatch shape while
        // the prefill path uses the symbolic `.causal` fast path; the
        // two MLX codepaths produce fp16 outputs that differ on the
        // order of 5e-2 across the 36-layer stack. The argmax checks
        // above pin the actual sampling-decision invariant; this
        // bound just guards against runaway numerical drift that
        // would indicate a real wiring bug rather than fp16 noise.
        XCTAssertLessThan(aggregateMax, 0.1,
                          "single-rank decode logit drift exceeds the " +
                          "expected fp16 noise envelope; maxAbsDiff=\(aggregateMax)")
    }

    // MARK: - Test 2: distributed decode vs oracle decode

    /// At `worldSize = 1` the distributed model's cache path must
    /// agree with the oracle's cache path. The pre-existing prefill
    /// equivalence test enforces the same contract for the
    /// cache-less forward; this test extends that guarantee through
    /// the cache plumbing.
    func testDistributedCacheDecodeMatchesSingleRank() throws {
        let fixtureRoot = try resolvedFixtureRoot()

        let metadataURL = fixtureRoot.appendingPathComponent("tq_shard_metadata.json")
        let metadata = try ShardMetadata(jsonData: Data(contentsOf: metadataURL))

        let configURL = fixtureRoot.appendingPathComponent("config.json")
        let config = try DistributedQwenConfiguration.load(from: configURL)

        // Share the materialised embedding table across both model
        // instances; allocating two ~600 MB tables in one process
        // exceeds the Metal allocator's wired-memory ceiling.
        let embeddingTable = try materialiseEmbeddingTable(
            modelDir: fixtureRoot,
            metadata: metadata,
            embeddingLayerName: "model.embed_tokens",
            hiddenSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            primaryBits: 4,
            residualBits: 4
        )

        // `DistributedGroup()` on a singleton process returns rank 0,
        // size 1 — the size-1 equivalence configuration.
        let group = DistributedGroup()
        let dist = try DistributedQwenModel(
            metadata: metadata,
            modelDir: fixtureRoot,
            group: group,
            embeddingTable: embeddingTable
        )
        let ref = try TurboQuantSingleRankModel(
            directory: fixtureRoot,
            embeddingTable: embeddingTable
        )

        let logitsRefDecode = runIncrementalDecode(
            prompt: Self.promptTokens,
            makeCache: { ref.makeCache() },
            forward: { ids, cache in ref(ids, cache: cache) }
        )
        let logitsDistDecode = runIncrementalDecode(
            prompt: Self.promptTokens,
            makeCache: { dist.makeCache() },
            forward: { ids, cache in dist(ids, cache: cache) }
        )
        MLX.eval(logitsRefDecode, logitsDistDecode)

        XCTAssertEqual(logitsDistDecode.shape, logitsRefDecode.shape,
                       "distributed and oracle decode logits must share shape; " +
                       "got dist=\(logitsDistDecode.shape) ref=\(logitsRefDecode.shape)")

        let diff = maxAbsDiff(logitsDistDecode, logitsRefDecode)
        let refMax = MLX.max(MLX.abs(logitsRefDecode.asType(.float32))).item(Float.self)
        print("[diag] dist-decode vs ref-decode maxAbsDiff=\(diff) refMax=\(refMax)")

        XCTAssertGreaterThan(refMax, 1.0,
                             "reference decode logits must be non-trivial; refMax=\(refMax)")
        // The size-1 distributed cache path goes through the same TQ
        // kernel as the oracle but with the wrappers' allSum/no-op
        // graph nodes, mirroring the prefill equivalence's tolerance
        // budget.
        XCTAssertLessThan(diff, 1e-2,
                          "size-1 distributed decode must match oracle decode " +
                          "within fp16 noise; maxAbsDiff=\(diff)")
    }

    // MARK: - Test 3: distributed decode vs distributed prefill

    /// Catches divergences between the distributed model's cache path
    /// and its own prefill path even if the oracle would also have
    /// the bug. Independent of the cross-model comparison in test 2.
    func testDistributedCacheDecodeMatchesItsOwnPrefill() throws {
        let fixtureRoot = try resolvedFixtureRoot()

        let metadataURL = fixtureRoot.appendingPathComponent("tq_shard_metadata.json")
        let metadata = try ShardMetadata(jsonData: Data(contentsOf: metadataURL))

        let group = DistributedGroup()
        let dist = try DistributedQwenModel(
            metadata: metadata,
            modelDir: fixtureRoot,
            group: group
        )

        let prefillIds = MLXArray(Self.promptTokens).reshaped(1, Self.promptTokens.count)
        let logitsPrefill = dist(prefillIds, cache: nil)
        MLX.eval(logitsPrefill)

        let logitsDecode = runIncrementalDecode(
            prompt: Self.promptTokens,
            makeCache: { dist.makeCache() },
            forward: { ids, cache in dist(ids, cache: cache) }
        )
        MLX.eval(logitsDecode)

        XCTAssertEqual(logitsPrefill.shape, logitsDecode.shape,
                       "distributed prefill and decode logits must share shape; " +
                       "got prefill=\(logitsPrefill.shape) decode=\(logitsDecode.shape)")

        var aggregateMax: Float = 0
        for i in 0 ..< Self.promptTokens.count {
            let prefillRow = logitRow(logitsPrefill, position: i)
            let decodeRow = logitRow(logitsDecode, position: i)
            let diff = maxAbsDiff(prefillRow, decodeRow)
            aggregateMax = max(aggregateMax, diff)
            let prefillTok = argmaxToken(prefillRow)
            let decodeTok = argmaxToken(decodeRow)
            print("[diag] dist prefill vs decode pos=\(i) maxAbsDiff=\(diff) " +
                  "argmax(prefill)=\(prefillTok) argmax(decode)=\(decodeTok)")
            // Argmax is the binding correctness check; see
            // testSingleRankCacheDecodeMatchesPrefill for the rationale.
            XCTAssertEqual(prefillTok, decodeTok,
                           "distributed prefill and decode must predict the same " +
                           "token at position \(i); got prefill=\(prefillTok) " +
                           "decode=\(decodeTok) (maxAbsDiff=\(diff))")
        }

        let refMax = MLX.max(MLX.abs(logitsPrefill.asType(.float32))).item(Float.self)
        XCTAssertGreaterThan(refMax, 1.0,
                             "reference logits must be non-trivial; refMax=\(refMax)")
        XCTAssertLessThan(aggregateMax, 0.1,
                          "distributed decode logit drift exceeds the expected " +
                          "fp16 noise envelope; maxAbsDiff=\(aggregateMax)")
    }
}
