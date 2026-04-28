// Behavioural tests for `TemperatureSampler`. The sampler is the
// stochastic counterpart to `GreedySampler`; tests verify the four
// observable contracts:
//
//   - Determinism: same `(logits, params, seed)` triple produces the
//     same token across two `TemperatureSampler` instances and across
//     the first call vs. a fresh sampler with the same seed.
//   - Top-k clipping: `topK = 1` collapses sampling to argmax
//     regardless of seed.
//   - Top-p clipping: a tiny `topP` collapses to the dominant-mass
//     token regardless of seed.
//   - Greedy short-circuit: `temperature == 0` returns argmax
//     regardless of `topK` / `topP`.
//
// All tests run over synthetic logits with no model fixture loaded.

import XCTest
import Foundation
import MLX
@testable import TurboQuantKit

final class DistributedSamplerTests: XCTestCase {

    // MARK: - Helpers

    /// Build a 1-D logits MLXArray from a Swift `[Float]` row.
    private func logits(_ row: [Float]) -> MLXArray {
        return MLXArray(row, [row.count])
    }

    /// Default sampling-policy fixture for tests that vary one knob.
    /// `maxTokens` is irrelevant for direct sampler calls — the
    /// sampler does not enforce it — but the `SamplingParams` struct
    /// requires a value, so a generic placeholder lives here.
    private func params(
        temperature: Float = 1.0,
        topK: Int? = nil,
        topP: Float? = nil
    ) -> SamplingParams {
        return SamplingParams(
            temperature: temperature,
            topK: topK,
            topP: topP,
            maxTokens: 1,
            stopTokens: []
        )
    }

    // MARK: - Test 1: per-token determinism within a single sampler

    /// A single `TemperatureSampler` instance, called repeatedly with
    /// the same logits, must produce the same per-call sequence as a
    /// second sampler seeded identically. Confirms the RNG state
    /// advances deterministically and that no hidden global state
    /// leaks between calls.
    func testDeterminismAcrossInstances() {
        // Mid-temperature with two competing heads at 3 and 12 and a
        // sharply-suppressed tail. The logit gap of 16 makes the
        // tail's combined softmax mass negligible (~1e-7), so
        // determinism plus peak-only draws are both crisply
        // observable in 32 samples.
        let row: [Float] = (0 ..< 16).map { i in
            switch i {
            case 3, 12: return 8.0
            default: return -8.0
            }
        }
        let l = logits(row)
        let p = params(temperature: 1.0)

        let samplerA = TemperatureSampler(seed: 42)
        let samplerB = TemperatureSampler(seed: 42)

        let n = 32
        var sequenceA: [Int] = []
        var sequenceB: [Int] = []
        for _ in 0 ..< n {
            sequenceA.append(samplerA.sample(logits: l, sampling: p))
            sequenceB.append(samplerB.sample(logits: l, sampling: p))
        }

        XCTAssertEqual(sequenceA, sequenceB,
                       "two samplers seeded identically must produce identical sequences")
        // Sanity: both sequences must hit only the two-peak tokens.
        for token in sequenceA {
            XCTAssertTrue([3, 12].contains(token),
                          "expected sampler to draw from the peak set; got \(token)")
        }
        // Sanity: with two roughly-equal heads, the sampler must
        // exercise both heads at least once across 32 draws.
        XCTAssertTrue(sequenceA.contains(3), "sampler never drew token 3 in 32 draws")
        XCTAssertTrue(sequenceA.contains(12), "sampler never drew token 12 in 32 draws")
    }

    // MARK: - Test 2: top-k clipping collapses to argmax

    /// `topK = 1` keeps only the highest-logit index in the support
    /// set, so the sampler must return that index on every draw
    /// regardless of seed.
    func testTopKEqualsOneCollapsesToArgmax() {
        let row: [Float] = [-1.0, 5.0, 0.5, -3.0, 2.7, 4.9]
        let argmaxIdx = 1 // value 5.0
        let l = logits(row)
        let p = params(temperature: 1.0, topK: 1)

        for seed in [UInt64(0), 1, 7, 99, UInt64.max / 2] {
            let sampler = TemperatureSampler(seed: seed)
            for _ in 0 ..< 16 {
                let token = sampler.sample(logits: l, sampling: p)
                XCTAssertEqual(token, argmaxIdx,
                               "topK=1 must collapse to argmax; got \(token) for seed \(seed)")
            }
        }
    }

    // MARK: - Test 3: top-p clipping collapses to dominant token

    /// `topP = 0.01` with one overwhelming-mass token forces the
    /// sampler to return that token on every draw. The dominant
    /// token's softmax mass exceeds 0.99 by a wide margin, so the
    /// nucleus prefix degenerates to that one token and the inverse-
    /// CDF lookup can only return its index.
    func testTopPClipsToDominantToken() {
        let vocab = 8
        var row = [Float](repeating: -10.0, count: vocab)
        // Token 4 has logit 50; its softmax mass is essentially 1.
        // The other tokens have logit -10, so their relative mass
        // is exp(-60) ~= 9e-27. The nucleus prefix is unambiguously
        // the single token at index 4.
        row[4] = 50.0
        let l = logits(row)
        let p = params(temperature: 1.0, topP: 0.01)

        for seed in [UInt64(1), 2, 3, 100, 10_000_000] {
            let sampler = TemperatureSampler(seed: seed)
            for _ in 0 ..< 8 {
                let token = sampler.sample(logits: l, sampling: p)
                XCTAssertEqual(token, 4,
                               "topP must keep dominant token only; got \(token) for seed \(seed)")
            }
        }
    }

    // MARK: - Test 4: temperature = 0 short-circuits to argmax

    /// `temperature == 0` collapses to argmax regardless of `topK` /
    /// `topP`. Matches `GreedySampler`'s contract; this test verifies
    /// `TemperatureSampler` honours the same shorthand so the engine's
    /// default factory's branching is correct on either path.
    func testTemperatureZeroIsArgmaxRegardlessOfTopKTopP() {
        let row: [Float] = [0.1, 0.2, 0.3, 9.9, 0.4, 0.5]
        let argmaxIdx = 3
        let l = logits(row)

        let policies: [SamplingParams] = [
            params(temperature: 0, topK: nil, topP: nil),
            params(temperature: 0, topK: 1, topP: nil),
            params(temperature: 0, topK: 5, topP: nil),
            params(temperature: 0, topK: nil, topP: 0.5),
            params(temperature: 0, topK: 2, topP: 0.9),
        ]

        for policy in policies {
            let sampler = TemperatureSampler(seed: 1)
            let token = sampler.sample(logits: l, sampling: policy)
            XCTAssertEqual(token, argmaxIdx,
                           "temperature=0 must return argmax for policy \(policy); got \(token)")
        }

        // Cross-check that `GreedySampler` and `TemperatureSampler`
        // agree at temperature=0 for the same input — they share the
        // same contract and divergence between them would split the
        // engine's default-factory branches.
        let greedy = GreedySampler()
        let temperature = TemperatureSampler(seed: 1)
        let greedyToken = greedy.sample(logits: l, sampling: params(temperature: 0))
        let temperatureToken = temperature.sample(logits: l, sampling: params(temperature: 0))
        XCTAssertEqual(greedyToken, temperatureToken,
                       "GreedySampler and TemperatureSampler must agree at temperature=0")
    }
}
