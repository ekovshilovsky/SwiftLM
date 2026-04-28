// Wire-format message types for the coordinator-to-joiner control
// channel that drives a distributed inference session.
//
// The control channel carries only session-level coordination:
// session lifecycle (start, end), prefill batch, decode triggers,
// and the coordinator's sampled token broadcasts. Activation tensors
// do NOT flow through this channel — they cross ranks via
// `DistributedGroup` collectives invoked inside the model's forward
// pass. The control channel and the collective channel are
// deliberately decoupled so the latter stays on the high-bandwidth
// transport (TB5 / RDMA) without being multiplexed against
// session-management traffic.
//
// Codec. v1 uses JSON via the synthesised Swift `Codable`
// conformance: human-readable on the wire, easy to inspect with
// standard tools, and adequate for the message volume (one
// `SessionStart` per request, one `Decode` plus one `InjectToken`
// per generated token, one `Prefill` per prompt). A more compact
// codec (CBOR / Protocol Buffers / a hand-rolled length-framed
// binary) is a follow-up if the per-token control overhead ever
// shows up in profiles.
//
// Authentication. The control channel runs over the same
// authenticated TCP transport established by the cluster handshake;
// these messages are exchanged after key agreement, so the wire
// codec does not need to embed authentication metadata.

import Foundation

// MARK: - Sampling parameters

/// Coordinator-side sampling configuration for one inference
/// session. Sent to joiners as part of `SessionStart` so any
/// joiner-side bookkeeping that depends on the sampling policy
/// (e.g. an eventual speculative-decode side channel) can branch
/// on it. v1 joiners do not sample — they only forward and
/// participate in collectives — but the parameters travel
/// alongside the session ID so the coordinator and joiners agree
/// on session shape from a single authoritative payload.
public struct SamplingParams: Codable, Sendable, Equatable {
    /// Softmax temperature applied to logits before sampling. A
    /// value of `0` selects greedy / argmax sampling.
    public let temperature: Float

    /// Top-k filter: keep only the `k` highest-probability tokens
    /// before re-normalising. `nil` disables top-k.
    public let topK: Int?

    /// Top-p (nucleus) filter: keep the smallest set of tokens
    /// whose cumulative probability reaches `p`. `nil` disables
    /// top-p.
    public let topP: Float?

    /// Hard upper bound on tokens to generate before forcing
    /// `SessionEnd`. Coordinator-enforced; joiners do not need to
    /// know it but it travels in `SessionStart` for completeness.
    public let maxTokens: Int

    /// Stop-token IDs that, when sampled, end the session. Empty
    /// when only `maxTokens` is binding.
    public let stopTokens: [Int]

    public init(
        temperature: Float,
        topK: Int? = nil,
        topP: Float? = nil,
        maxTokens: Int,
        stopTokens: [Int] = []
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.maxTokens = maxTokens
        self.stopTokens = stopTokens
    }
}

// MARK: - Control messages

/// One message on the coordinator-to-joiner control channel. Five
/// shapes cover the entire session lifecycle:
///
///   - `sessionStart`: opens a session, carries the session ID and
///     sampling configuration, and a deterministic seed for any
///     joiner-side stochastic state.
///   - `prefill`: prompt tokens to consume in a single forward pass
///     before the decode loop begins.
///   - `decode`: asks every rank to run one decode-step forward
///     pass against the current KV cache. Carries no args.
///   - `injectToken`: coordinator's sampled token from the previous
///     decode step. Joiners append it to their KV cache before the
///     next `decode` arrives so per-rank state stays in sync.
///   - `sessionEnd`: releases all per-session state on the joiner.
public enum InferenceControlMessage: Codable, Sendable, Equatable {
    case sessionStart(sessionID: UUID, sampling: SamplingParams, seed: UInt64)
    case prefill(promptTokens: [Int])
    case decode
    case injectToken(sampledToken: Int)
    case sessionEnd
}

// MARK: - Codec

/// JSON encoder and decoder pair tuned for the control channel:
/// stable key ordering and compact output by default. Tests and
/// the runtime layer should reuse these instances rather than
/// constructing fresh ones per message so encoding behaviour stays
/// identical across the codebase.
public enum InferenceControlCodec {

    /// Encoder used for outbound control messages. `sortedKeys`
    /// makes wire output deterministic for the same input — useful
    /// when test suites assert on encoded bytes and when triaging
    /// a deployed cluster from packet captures.
    public static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    /// Decoder used for inbound control messages.
    public static let decoder = JSONDecoder()

    /// Convenience: encode a control message to bytes.
    public static func encode(_ message: InferenceControlMessage) throws -> Data {
        return try encoder.encode(message)
    }

    /// Convenience: decode a control message from bytes.
    public static func decode(_ data: Data) throws -> InferenceControlMessage {
        return try decoder.decode(InferenceControlMessage.self, from: data)
    }
}
