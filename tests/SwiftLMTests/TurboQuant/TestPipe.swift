// In-process byte-pipe pair implementing ClusterHandshakeEndpoint on
// both ends. Exists so the handshake unit tests can drive both sides
// concurrently in a single test process without opening real sockets.
//
// Wrapped in `#if DEBUG` to match the project convention for test-only
// helpers (same treatment as InMemoryClusterKeyStore). A release build
// has no legitimate use for an in-process transport and the guard
// keeps it from shipping in the release binary.

#if DEBUG

import Foundation
import TurboQuantKit

/// Paired in-process byte pipes implementing `ClusterHandshakeEndpoint`.
/// `endpointA` writes into the A-to-B channel and reads from B-to-A;
/// `endpointB` does the mirror. Tests spawn the coordinator on one
/// endpoint and the joiner on the other and await both to completion.
public final class TestPipe: @unchecked Sendable {
    public let endpointA: PipeEndpoint
    public let endpointB: PipeEndpoint

    public init() {
        let aToB = ByteChannel()
        let bToA = ByteChannel()
        self.endpointA = PipeEndpoint(incoming: bToA, outgoing: aToB)
        self.endpointB = PipeEndpoint(incoming: aToB, outgoing: bToA)
    }

    /// Simulate the A-to-B direction going EOF. Any subsequent `recv`
    /// on `endpointB` that cannot be fully satisfied from buffered
    /// bytes throws `.truncatedMessage`.
    public func closeAToB() {
        endpointB.incoming.close()
    }

    /// Simulate the B-to-A direction going EOF.
    public func closeBToA() {
        endpointA.incoming.close()
    }
}

// MARK: - Internal byte channel

// A simple unidirectional byte queue. `append` enqueues bytes, `read`
// awaits until `count` bytes are available (or the channel is closed)
// and returns them. Coarse-grained locking is fine here — the tests
// push at most a few hundred bytes per run and lock contention is a
// non-issue at that scale.
final class ByteChannel: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = Data()
    private var closed = false
    private var waiters: [(Int, CheckedContinuation<Data, Error>)] = []

    func append(_ data: Data) {
        lock.lock()
        buffer.append(data)
        let drained = drainWaitersLocked()
        lock.unlock()
        for resume in drained { resume() }
    }

    func close() {
        lock.lock()
        closed = true
        let drained = drainWaitersLocked()
        lock.unlock()
        for resume in drained { resume() }
    }

    func read(_ count: Int) async throws -> Data {
        precondition(count >= 0)
        if count == 0 { return Data() }
        return try await withCheckedThrowingContinuation { cont in
            lock.lock()
            if buffer.count >= count {
                let chunk = Data(buffer.prefix(count))
                buffer.removeFirst(count)
                lock.unlock()
                cont.resume(returning: chunk)
                return
            }
            if closed {
                lock.unlock()
                // Signal EOF via CancellationError; the endpoint
                // adapter translates it into truncatedMessage.
                cont.resume(throwing: CancellationError())
                return
            }
            waiters.append((count, cont))
            lock.unlock()
        }
    }

    // Drain any waiters that are now satisfiable or that must fail
    // because the channel closed. Returns a list of thunks for the
    // caller to invoke outside the lock — resuming under the lock can
    // deadlock if the resumption callback reenters the channel.
    private func drainWaitersLocked() -> [() -> Void] {
        var ready: [() -> Void] = []
        while let (count, cont) = waiters.first, buffer.count >= count {
            waiters.removeFirst()
            let chunk = Data(buffer.prefix(count))
            buffer.removeFirst(count)
            ready.append { cont.resume(returning: chunk) }
        }
        if closed {
            for (_, cont) in waiters {
                ready.append { cont.resume(throwing: CancellationError()) }
            }
            waiters.removeAll()
        }
        return ready
    }
}

// MARK: - Endpoint adapter

public final class PipeEndpoint: ClusterHandshakeEndpoint, @unchecked Sendable {
    let incoming: ByteChannel
    let outgoing: ByteChannel

    init(incoming: ByteChannel, outgoing: ByteChannel) {
        self.incoming = incoming
        self.outgoing = outgoing
    }

    public func send(_ data: Data) async throws {
        outgoing.append(data)
    }

    public func recv(_ count: Int) async throws -> Data {
        do {
            return try await incoming.read(count)
        } catch {
            // The ByteChannel surface signals EOF with CancellationError.
            // Reframe it in the handshake module's vocabulary so tests
            // can assert on the semantically meaningful error.
            throw ClusterHandshakeError.truncatedMessage
        }
    }
}

#endif // DEBUG
