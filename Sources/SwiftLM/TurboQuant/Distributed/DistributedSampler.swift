// Token sampling strategy for the coordinator-side session loop.
//
// Sampling happens exclusively on the coordinator. Under replicated
// `lm_head` every rank already holds full logits at the end of each
// forward pass, but only the coordinator picks the sampled token and
// broadcasts it back to joiners; this guarantees identical KV-cache
// advancement on every rank without an additional consensus round.
//
// The strategy is factored behind a protocol so different sampling
// shapes (greedy argmax vs. temperature-scaled top-k / top-p)
// can drop in without touching the session actor. `SamplingParams`
// carries the policy knobs (`temperature`, `topK`, `topP`); a sampler
// may inspect those fields or ignore them.

import Foundation
import MLX

/// Picks one token id from a final-position logits row. Conformers
/// must be Sendable so the session actor can hold one across `await`
/// suspension points without warnings under stricter concurrency
/// settings.
public protocol DistributedSampler: Sendable {
    /// Select a token id from a final-position logits vector.
    ///
    /// - Parameters:
    ///   - logits: 1-D `MLXArray` of shape `[vocabSize]`. Caller is
    ///     responsible for slicing the last position out of the
    ///     `[batch, length, vocabSize]` forward-pass output before
    ///     calling this function.
    ///   - sampling: Per-request sampling configuration. Greedy
    ///     ignores every field; stochastic samplers consult
    ///     `temperature`, `topK`, and `topP`.
    /// - Returns: The selected vocabulary token id.
    func sample(logits: MLXArray, sampling: SamplingParams) -> Int
}

/// Argmax sampler. Selects the highest-logit token id deterministically.
/// Equivalent to setting `temperature = 0` on the request and what
/// `SamplingParams.temperature == 0` resolves to inside the engine's
/// default sampler factory.
public struct GreedySampler: DistributedSampler {
    public init() {}

    public func sample(logits: MLXArray, sampling: SamplingParams) -> Int {
        // Cast to float32 before argmax to avoid the float16 ties that
        // can show up in heavily-quantised logits and produce
        // implementation-dependent argmax results across hardware.
        let argmax = MLX.argMax(logits.asType(.float32), axis: -1)
        return Int(argmax.item(Int32.self))
    }
}

// MARK: - Seeded RNG

/// Deterministic 64-bit RNG built on splitmix64. Stateful: each
/// `next()` advances the state so successive draws inside one session
/// do not collide on the same value. Same seed across processes /
/// hardware produces the same draw sequence — the session-level
/// reproducibility contract rides on this.
///
/// `splitmix64` is the standard "good-enough" small-state seedable
/// generator: one 64-bit word of state, one multiply / xor / shift
/// chain per draw, no platform-specific dependencies. It is not
/// cryptographically strong; that is irrelevant for sampling.
struct SplitMix64RNG {
    /// Internal state. Mutated on every `next()`.
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    /// Advance state and return the next 64-bit draw.
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Return a uniform `Float` draw in `[0, 1)`. Uses the upper
    /// 24 bits of the 64-bit raw draw — enough precision for a
    /// `Float`-valued CDF lookup over a vocab on the order of 1e5
    /// without quantisation artefacts at the cumulative-mass cutoff.
    mutating func nextUnitFloat() -> Float {
        let raw = next()
        let bits = UInt32(truncatingIfNeeded: raw >> 40)
        // Map [0, 2^24) to [0, 1) by dividing by 2^24.
        return Float(bits) / Float(1 << 24)
    }
}

// MARK: - Stochastic sampler

/// Samples a token from the categorical distribution implied by the
/// logits, after applying temperature scaling, optional top-k
/// filtering, and optional top-p (nucleus) filtering. Uses a
/// splitmix64 RNG seeded from the session's `seed`, so the same
/// `(logits, sampling, seed)` triple produces the same token sequence
/// across runs and across hosts.
///
/// `temperature == 0` short-circuits to argmax, matching `GreedySampler`'s
/// behaviour exactly. Callers that want greedy semantics regardless of
/// the requested `topK` / `topP` should set `temperature == 0` rather
/// than relying on top-k=1 with a stochastic temperature.
///
/// Isolation. `TemperatureSampler` is a reference type with internal
/// mutable RNG state. The state is only mutated from inside the
/// coordinator session actor's serial executor — sampling is one-call-
/// per-decode-step on a single actor — so the lock-free reference
/// path is safe by construction. `@unchecked Sendable` annotates the
/// guarantee for the concurrency checker. Switching to a value-type
/// implementation would require either a synchronisation primitive
/// (no measurable benefit for the one-call-per-step pattern) or a
/// protocol-wide change to make `sample` `mutating`, which would
/// cascade into `CoordinatorInferenceSession`'s stored `sampler`
/// property and add no value.
public final class TemperatureSampler: DistributedSampler, @unchecked Sendable {
    /// RNG state. Advances on every `sample` call; the rest of the
    /// transformation pipeline (temperature, top-k, top-p) is pure.
    private var rng: SplitMix64RNG

    /// Construct a sampler bound to `seed`. The session passes its
    /// per-request `seed` here; two sessions with the same prompt and
    /// the same seed must reproduce the same token sequence.
    public init(seed: UInt64) {
        self.rng = SplitMix64RNG(seed: seed)
    }

    public func sample(logits: MLXArray, sampling: SamplingParams) -> Int {
        // Greedy short-circuit. `temperature == 0` means "argmax" by
        // convention; honouring it here lets the engine's default
        // factory route greedy requests through this same sampler in
        // edge cases (e.g. a custom factory that always returns
        // `TemperatureSampler` for telemetry uniformity) without
        // surprising the caller.
        if sampling.temperature == 0 {
            let argmax = MLX.argMax(logits.asType(.float32), axis: -1)
            return Int(argmax.item(Int32.self))
        }

        // The full sampling pipeline runs in float32 for numerical
        // stability: subtract-max softmax, cumsum, and the renormalise
        // step after top-p masking all behave better with the wider
        // exponent range than fp16 would supply.
        let row = logits.asType(.float32).asArray(Float.self)
        let vocabSize = row.count
        precondition(vocabSize > 0, "logits vector must be non-empty")

        // Temperature scaling. Done in-place over the local Swift
        // array — every downstream step also runs on the host because
        // the sample step culminates in a serial CDF lookup.
        var scaled = row
        let invTemp = 1.0 / sampling.temperature
        for i in 0 ..< vocabSize {
            scaled[i] *= invTemp
        }

        // Top-k filter: keep the K highest-logit indices, mask the
        // rest to -infinity so the subsequent softmax assigns them
        // zero probability. A partial sort by descending logit yields
        // the kept set; ties at the cutoff fall to whichever index
        // stable sort places first, which is deterministic for a
        // given input array.
        if let k = sampling.topK, k > 0 && k < vocabSize {
            // Indices sorted by ascending logit, then we keep the top
            // `k` from the tail. Building an explicit kept-set keeps
            // the masking branch simple and deterministic.
            let sortedIdx = (0 ..< vocabSize)
                .sorted { scaled[$0] > scaled[$1] }
            var kept = [Bool](repeating: false, count: vocabSize)
            for i in 0 ..< k {
                kept[sortedIdx[i]] = true
            }
            for i in 0 ..< vocabSize where !kept[i] {
                scaled[i] = -.infinity
            }
        }

        // Stable softmax. Subtract the running max before exponentiation
        // so the largest exponent is `exp(0) = 1` and overflow is
        // impossible. `-infinity` entries (from top-k masking) become
        // `0` after `exp`, exactly the desired behaviour.
        var maxLogit: Float = -.infinity
        for v in scaled where v > maxLogit { maxLogit = v }
        var probs = [Float](repeating: 0, count: vocabSize)
        var sum: Float = 0
        for i in 0 ..< vocabSize {
            let e = expf(scaled[i] - maxLogit)
            probs[i] = e
            sum += e
        }
        // Defensive: if every logit was masked to -infinity (k=0 case
        // is filtered above, but a degenerate input could still get
        // here), fall back to argmax of the original logits to avoid
        // dividing by zero. Argmax is the natural collapse point.
        if sum == 0 {
            let argmax = MLX.argMax(logits.asType(.float32), axis: -1)
            return Int(argmax.item(Int32.self))
        }
        let invSum = 1.0 / sum
        for i in 0 ..< vocabSize {
            probs[i] *= invSum
        }

        // Top-p (nucleus) filter. Sort probabilities descending, accumulate
        // until the cumulative mass reaches `p`, and zero out everything
        // beyond that prefix. The single-token nucleus case (one token
        // already exceeds `p`) keeps that token only and is therefore
        // equivalent to argmax — matches reference implementations.
        if let p = sampling.topP, p > 0 && p < 1 {
            let sortedIdx = (0 ..< vocabSize)
                .sorted { probs[$0] > probs[$1] }
            var cumulative: Float = 0
            var kept = [Bool](repeating: false, count: vocabSize)
            // Always keep at least the most likely token so a tiny `p`
            // (e.g. `topP = 0.01`) can never produce an all-zero mask.
            kept[sortedIdx[0]] = true
            cumulative += probs[sortedIdx[0]]
            var idx = 1
            while idx < vocabSize && cumulative < p {
                kept[sortedIdx[idx]] = true
                cumulative += probs[sortedIdx[idx]]
                idx += 1
            }
            // Zero the tail and renormalise over the kept prefix so
            // the CDF lookup below operates on a valid distribution.
            var newSum: Float = 0
            for i in 0 ..< vocabSize {
                if !kept[i] {
                    probs[i] = 0
                } else {
                    newSum += probs[i]
                }
            }
            if newSum > 0 {
                let invNewSum = 1.0 / newSum
                for i in 0 ..< vocabSize {
                    probs[i] *= invNewSum
                }
            }
        }

        // Inverse-CDF sample. Walk the probability vector once,
        // accumulating into a running CDF; return the first index
        // whose CDF crosses the uniform draw `u`. Linear in vocab
        // size, no extra allocations beyond the running scalar.
        let u = rng.nextUnitFloat()
        var cdf: Float = 0
        for i in 0 ..< vocabSize {
            cdf += probs[i]
            if u < cdf {
                return i
            }
        }
        // Floating-point round-off can leave `cdf` a hair below 1.0
        // even on a fully-normalised distribution. Falling through
        // here is rare but possible; return the last non-zero
        // probability index so the result still respects the
        // top-k / top-p mask.
        for i in stride(from: vocabSize - 1, through: 0, by: -1) where probs[i] > 0 {
            return i
        }
        // Final fallback: argmax of the unmodified logits. Should be
        // unreachable given the earlier `sum == 0` guard.
        let argmax = MLX.argMax(logits.asType(.float32), axis: -1)
        return Int(argmax.item(Int32.self))
    }
}
