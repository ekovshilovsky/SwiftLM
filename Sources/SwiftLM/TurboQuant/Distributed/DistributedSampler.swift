// Token sampling strategy for the coordinator-side session loop.
//
// Sampling happens exclusively on the coordinator. Under replicated
// `lm_head` every rank already holds full logits at the end of each
// forward pass, but only the coordinator picks the sampled token and
// broadcasts it back to joiners; this guarantees identical KV-cache
// advancement on every rank without an additional consensus round.
//
// The strategy is factored behind a protocol so future stochastic
// samplers (temperature-scaled top-k / top-p) can drop in without
// touching the session actor. v1 ships a greedy implementation only.
// `SamplingParams` carries the policy knobs (`temperature`, `topK`,
// `topP`); a sampler may inspect those fields or ignore them.

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
/// `SamplingParams.temperature == 0` should resolve to once a
/// stochastic sampler ships alongside.
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
