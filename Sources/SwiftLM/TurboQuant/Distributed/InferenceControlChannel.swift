// Transport abstraction for the coordinator-to-joiner control flow.
//
// `InferenceControlChannel` decouples the session-driving actor
// (`CoordinatorInferenceSession`) from the underlying TCP / NWConnection
// transport that carries control messages between ranks. The actor
// holds one channel per joiner; broadcasting iterates over the
// channel array. Tests substitute an in-memory implementation so the
// session-loop semantics can be exercised without a real network
// stack; the production wiring later plugs in a NWConnection-backed
// implementation that frames messages over the cluster's
// authenticated TCP transport.
//
// The protocol intentionally exposes both directions (outbound `send`
// plus an inbound `AsyncStream`). The coordinator side does not
// currently consume `inbound` — joiners drive the message flow in
// v1 — but exposing the same shape on both ends keeps the interface
// symmetric with the matching joiner-worker contract that lands in a
// follow-up. Symmetry also leaves room for future bidirectional
// flows such as drain-acknowledgement or per-rank error reports
// without needing a second protocol.
//
// Sendability. `InferenceControlMessage` is already `Sendable`, so
// any concrete channel that ferries values across isolation domains
// (between the actor and a transport task) needs no extra
// constraints to do so safely. The protocol itself is declared
// `Sendable` so the session actor can hold an `[InferenceControlChannel]`
// property without triggering a non-Sendable storage warning.

import Foundation

/// Bidirectional message endpoint for one joiner. The coordinator
/// holds one of these per joiner; broadcasting iterates over them.
public protocol InferenceControlChannel: Sendable {
    /// Send a control message to the peer. Implementations are
    /// expected to serialise via `InferenceControlCodec` and frame
    /// the encoded bytes appropriately for their transport.
    func send(_ message: InferenceControlMessage) async throws

    /// AsyncStream of inbound messages from the peer. The coordinator
    /// side does not currently consume this — joiners drive the
    /// message flow — but exposing the same shape on both ends keeps
    /// the protocol symmetric for future bidirectional flows
    /// (drain-ack, error reports).
    var inbound: AsyncStream<InferenceControlMessage> { get }
}

// MARK: - In-memory implementation for tests

/// Test helper that records every message passed to `send` and lets
/// inbound messages be injected by the test author. Backed by an
/// internal lock so a test can assert on the recorded sequence from
/// any task while the session actor is still running on its own
/// isolation domain.
///
/// `@unchecked Sendable` is appropriate here: the lock guards every
/// access to the mutable state, so the type is effectively Sendable
/// even though the compiler cannot prove it from the storage
/// declarations alone.
public final class InMemoryInferenceControlChannel: InferenceControlChannel,
                                                    @unchecked Sendable {

    /// Lock-guarded log of every message passed to `send`. Tests
    /// snapshot this via `sentMessages` after the session completes.
    private let lock = NSLock()
    private var _sent: [InferenceControlMessage] = []

    /// Continuation for the `inbound` stream. Tests yield messages
    /// via `injectInbound(_:)`; production implementations would
    /// drive this from a network read loop instead.
    private let inboundContinuation: AsyncStream<InferenceControlMessage>.Continuation

    public let inbound: AsyncStream<InferenceControlMessage>

    public init() {
        var continuation: AsyncStream<InferenceControlMessage>.Continuation!
        self.inbound = AsyncStream { c in continuation = c }
        self.inboundContinuation = continuation
    }

    public func send(_ message: InferenceControlMessage) async throws {
        // `withLock` is the async-safe scoped form of `NSLock`'s
        // lock / unlock pair; the closure-based API survives Swift 6
        // strict-concurrency checks where the bare `lock()` /
        // `unlock()` methods do not.
        lock.withLock {
            _sent.append(message)
        }
    }

    /// Snapshot of messages sent through this channel so far. Order
    /// reflects the call order of `send`.
    public var sentMessages: [InferenceControlMessage] {
        lock.withLock { _sent }
    }

    /// Push one message into the `inbound` stream. Joiner-side tests
    /// use this to simulate messages arriving from the coordinator;
    /// for the coordinator-side session loop, which does not consume
    /// `inbound` in v1, this is unused.
    public func injectInbound(_ message: InferenceControlMessage) {
        inboundContinuation.yield(message)
    }

    /// End the inbound stream so any consumer's `for await` loop
    /// exits cleanly.
    public func finishInbound() {
        inboundContinuation.finish()
    }
}
