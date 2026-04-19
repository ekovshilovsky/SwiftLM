// ClusterHandshake unit tests. Exercises the three-message HMAC
// mutual authentication plus the AES-GCM-sealed working-key delivery
// against an in-process pipe, so the tests are deterministic and
// microsecond-fast.
//
// Coverage: the positive path, wrong-passphrase on each side, a
// captured-message-4 replay into a fresh session, and an EOF-mid-
// message truncation.

#if DEBUG

import XCTest
import TurboQuantKit

final class ClusterHandshakeTests: XCTestCase {

    // MARK: - Helpers

    // A fixed-shape handshake key. Real deployments derive one via
    // Argon2id over (passphrase, clusterId) — for protocol-level tests
    // we bypass that and use the final 32-byte value directly.
    private func fakeHandshakeKey(_ fill: UInt8) -> Data {
        Data(repeating: fill, count: 32)
    }

    private func fakeWorkingClusterKey(_ fill: UInt8) -> Data {
        Data(repeating: fill, count: 32)
    }

    // MARK: - Positive path

    func testSuccessfulHandshakeDeliversWorkingKey() async throws {
        let pipe = TestPipe()
        let handshakeKey = fakeHandshakeKey(0xA1)
        let workingKey = fakeWorkingClusterKey(0x55)

        async let coordinatorSide: Void = runClusterHandshakeCoordinator(
            handshakeKey: handshakeKey,
            workingClusterKey: workingKey,
            endpoint: pipe.endpointA
        )
        async let joinerSide: Data = runClusterHandshakeJoiner(
            handshakeKey: handshakeKey,
            endpoint: pipe.endpointB
        )

        try await coordinatorSide
        let delivered = try await joinerSide

        XCTAssertEqual(delivered, workingKey,
                       "joiner must receive the coordinator's working key byte-for-byte")
        XCTAssertEqual(delivered.count, 32,
                       "working key must be 32 bytes — a drift here indicates protocol change")
    }

    // MARK: - Wrong passphrase, observed from the joiner side

    func testWrongPassphraseFailsOnJoiner() async throws {
        // Coordinator has the real handshake key; the joiner has a
        // different one (simulates a user who typed the wrong
        // passphrase). The joiner's message-2 HMAC verification must
        // fail under constant-time comparison and the call must throw
        // .peerAuthFailed specifically.
        let pipe = TestPipe()
        let rightKey = fakeHandshakeKey(0xA1)
        let wrongKey = fakeHandshakeKey(0xA2)
        let workingKey = fakeWorkingClusterKey(0x55)

        // Run the coordinator in a task; we do not assert its outcome
        // because once the joiner tears down the connection the
        // coordinator sees an EOF or a bad response-MAC and either
        // result is acceptable. The test's contract is on the joiner.
        async let coordinatorTask: Void = {
            do {
                try await runClusterHandshakeCoordinator(
                    handshakeKey: rightKey,
                    workingClusterKey: workingKey,
                    endpoint: pipe.endpointA
                )
            } catch {
                // ignore — coordinator outcome not asserted in this test
            }
        }()

        do {
            _ = try await runClusterHandshakeJoiner(
                handshakeKey: wrongKey,
                endpoint: pipe.endpointB
            )
            XCTFail("joiner should have thrown on bad coordinator MAC")
        } catch let e as ClusterHandshakeError {
            XCTAssertEqual(e, .peerAuthFailed,
                           "joiner must surface peerAuthFailed, not a transport-level error")
        } catch {
            XCTFail("unexpected error type: \(error)")
        }

        // Close the pipe so the coordinator task can complete. Without
        // this the coordinator may block forever awaiting message 3.
        pipe.closeBToA()
        await coordinatorTask
    }

    // MARK: - Wrong passphrase, observed from the coordinator's perspective

    func testWrongPassphraseFailsOnCoordinator() async throws {
        // Inverse framing of the previous test: coordinator holds the
        // wrong key, joiner holds the right one. The failure still
        // surfaces on the joiner first, because the joiner verifies the
        // coordinator's message-2 MAC before sending anything else —
        // and that MAC was computed with the wrong key. Both tests
        // exist so a future regression on either side is caught by
        // name.
        let pipe = TestPipe()
        let rightKey = fakeHandshakeKey(0xA1)
        let wrongKey = fakeHandshakeKey(0xA2)
        let workingKey = fakeWorkingClusterKey(0x55)

        async let coordinatorTask: Void = {
            do {
                try await runClusterHandshakeCoordinator(
                    handshakeKey: wrongKey,
                    workingClusterKey: workingKey,
                    endpoint: pipe.endpointA
                )
            } catch {
                // ignore — not asserted here
            }
        }()

        do {
            _ = try await runClusterHandshakeJoiner(
                handshakeKey: rightKey,
                endpoint: pipe.endpointB
            )
            XCTFail("joiner should have thrown on bad coordinator MAC")
        } catch let e as ClusterHandshakeError {
            XCTAssertEqual(e, .peerAuthFailed,
                           "joiner must detect the coordinator's mismatched MAC")
        } catch {
            XCTFail("unexpected error type: \(error)")
        }

        pipe.closeBToA()
        await coordinatorTask
    }

    // MARK: - Replayed message 4 against a fresh session

    func testReplayedMessage4FailsOnFreshSession() async throws {
        // First handshake (A): record everything the coordinator sends
        // to the joiner, then pull out message 4's bytes from that
        // capture. Second handshake (B): run coordinator and joiner
        // side-by-side, but splice handshake A's message-4 bytes into
        // the stream the joiner reads, in place of handshake B's real
        // message 4. The joiner's AES-GCM open must fail because the
        // AAD it computes (with handshake B's fresh nonces) does not
        // match the AAD under which handshake A's ciphertext was
        // sealed. Surface is .sealOpenFailed.

        let handshakeKey = fakeHandshakeKey(0xA1)
        let workingKey = fakeWorkingClusterKey(0x55)

        // ── Handshake A: run to completion, capture bytes sent by
        //    the coordinator via a recording endpoint.
        let captured = try await captureCoordinatorStream(
            handshakeKey: handshakeKey,
            workingKey: workingKey
        )

        // Message 2 layout: 1 type-byte + 32-byte nonce + 32-byte HMAC = 65 bytes.
        // Message 4 starts immediately after that in the coordinator's
        // output stream.
        let message2Length = 1 + 32 + 32
        XCTAssertGreaterThan(captured.count, message2Length,
                             "capture must include at least message 2 and part of message 4")
        let message4Bytes = captured.subdata(in: message2Length..<captured.count)
        XCTAssertEqual(message4Bytes.first, 0x04,
                       "captured message-4 bytes must start with the sealed-key type byte")

        // ── Handshake B: wire the joiner's incoming channel through a
        //    substitute that replaces the real message 4 with the
        //    handshake-A capture. The substitution is byte-for-byte at
        //    the start of message 4, which the coordinator signals by
        //    having already sent message 2 (message 4 is the next and
        //    final message on this direction).
        let pipe = TestPipe()
        let joiner = ReplayingMessage4Endpoint(
            inner: pipe.endpointB,
            replacement: message4Bytes,
            cutoverBytes: message2Length   // bytes of message 2, then swap in replay
        )

        async let coordinatorSide: Void = runClusterHandshakeCoordinator(
            handshakeKey: handshakeKey,
            workingClusterKey: workingKey,
            endpoint: pipe.endpointA
        )
        async let joinerSide: Data = runClusterHandshakeJoiner(
            handshakeKey: handshakeKey,
            endpoint: joiner
        )

        // Coordinator path completes — it sent a well-formed message 4
        // that the wrapping endpoint on the joiner side threw away.
        try await coordinatorSide
        do {
            _ = try await joinerSide
            XCTFail("joiner should have rejected the replayed message 4")
        } catch let e as ClusterHandshakeError {
            XCTAssertEqual(e, .sealOpenFailed,
                           "replay of a stale sealed key must surface as sealOpenFailed")
        } catch {
            XCTFail("unexpected error type: \(error)")
        }
    }

    // MARK: - Truncated message 2

    func testTruncatedMessage2FailsCleanly() async throws {
        // Simulate the coordinator disappearing after sending enough
        // bytes to look like the start of message 2 but before the
        // HMAC is delivered. The joiner's recv(hmacLength) for the MAC
        // field must surface as .truncatedMessage.
        let pipe = TestPipe()
        let handshakeKey = fakeHandshakeKey(0xA1)

        // Hand-send the prefix of message 2 from the coordinator side:
        // 1 type-byte + 32-byte nonce, and then close the channel to
        // signal EOF before the HMAC field would arrive.
        let fakeNonce = Data(repeating: 0x77, count: 32)
        var message2Prefix = Data()
        message2Prefix.append(0x02)
        message2Prefix.append(fakeNonce)
        try await pipe.endpointA.send(message2Prefix)
        pipe.closeAToB()

        // The joiner also sends message 1 first; stage that in the
        // opposite direction so the joiner's own send does not stall.
        // We run only the joiner half of the protocol here because
        // that is the side under test.
        async let joinerSide: Data = runClusterHandshakeJoiner(
            handshakeKey: handshakeKey,
            endpoint: pipe.endpointB
        )

        // Drain message 1 from the A side so send() does not block
        // forever (our channel is unbounded, so it actually does not,
        // but the drain keeps the test's intent explicit).
        _ = try? await pipe.endpointA.recv(1 + 32)

        do {
            _ = try await joinerSide
            XCTFail("joiner should have thrown on truncated message 2")
        } catch let e as ClusterHandshakeError {
            XCTAssertEqual(e, .truncatedMessage,
                           "EOF mid-message must surface as truncatedMessage")
        } catch {
            XCTFail("unexpected error type: \(error)")
        }
    }

    // MARK: - Test support: recording and replaying endpoints

    // Synchronous state container — all mutation happens inside a
    // sync closure so a lock is never held across an `await`. Swift 6
    // makes lock-across-await a diagnostic; using a scoped helper
    // silences the warning while preserving the underlying invariant.
    private final class Synchronized<T>: @unchecked Sendable {
        private let lock = NSLock()
        private var value: T
        init(_ value: T) { self.value = value }
        func withLock<R>(_ body: (inout T) -> R) -> R {
            lock.lock(); defer { lock.unlock() }
            return body(&value)
        }
    }

    // Records every byte sent by the coordinator so a full handshake
    // can be captured for replay-attack testing. The recorded stream
    // is the coordinator's output only (message 2 followed by message
    // 4) — the joiner's output is uninteresting here.
    private final class RecordingEndpoint: ClusterHandshakeEndpoint, @unchecked Sendable {
        let inner: ClusterHandshakeEndpoint
        private let recorded = Synchronized(Data())

        init(inner: ClusterHandshakeEndpoint) {
            self.inner = inner
        }

        func send(_ data: Data) async throws {
            recorded.withLock { $0.append(data) }
            try await inner.send(data)
        }

        func recv(_ count: Int) async throws -> Data {
            return try await inner.recv(count)
        }

        func snapshot() -> Data {
            recorded.withLock { $0 }
        }
    }

    // Runs a complete handshake and returns the bytes that the
    // coordinator wrote into its endpoint. Used by the replay test.
    private func captureCoordinatorStream(
        handshakeKey: Data,
        workingKey: Data
    ) async throws -> Data {
        let pipe = TestPipe()
        let recorder = RecordingEndpoint(inner: pipe.endpointA)

        async let coordinatorSide: Void = runClusterHandshakeCoordinator(
            handshakeKey: handshakeKey,
            workingClusterKey: workingKey,
            endpoint: recorder
        )
        async let joinerSide: Data = runClusterHandshakeJoiner(
            handshakeKey: handshakeKey,
            endpoint: pipe.endpointB
        )
        try await coordinatorSide
        _ = try await joinerSide
        return recorder.snapshot()
    }

    // Mutable state for the replay-substitution endpoint below. Kept
    // in a single struct so the whole state can be atomically
    // inspected and advanced inside a single Synchronized.withLock.
    private struct ReplayState {
        var bytesFromInnerSoFar: Int = 0
        var replayBuffer: Data
        var drainedPastCutover: Bool = false
    }

    // Wraps a joiner's endpoint so the first `cutoverBytes` bytes come
    // from the live coordinator (normal handshake through message 2),
    // then `recv` draws from `replacement` for the bytes of message 4
    // instead of reading what the live coordinator sent. The live
    // coordinator's message-4 bytes are drained into a discard buffer
    // so the coordinator's send does not stall.
    private final class ReplayingMessage4Endpoint: ClusterHandshakeEndpoint, @unchecked Sendable {
        let inner: ClusterHandshakeEndpoint
        private let cutoverBytes: Int
        private let state: Synchronized<ReplayState>

        init(inner: ClusterHandshakeEndpoint, replacement: Data, cutoverBytes: Int) {
            self.inner = inner
            self.cutoverBytes = cutoverBytes
            self.state = Synchronized(ReplayState(replayBuffer: replacement))
        }

        func send(_ data: Data) async throws {
            try await inner.send(data)
        }

        func recv(_ count: Int) async throws -> Data {
            let alreadyRead = state.withLock { $0.bytesFromInnerSoFar }

            // Before the cutover the endpoint is transparent — message
            // 2 flows through unchanged.
            if alreadyRead + count <= cutoverBytes {
                let data = try await inner.recv(count)
                state.withLock { $0.bytesFromInnerSoFar += count }
                return data
            }

            // First call that crosses the cutover: drain whatever the
            // live coordinator is about to send (its real message 4)
            // into a discard buffer so the coordinator's own `send`
            // completes, then serve the replay from the replacement
            // buffer. Subsequent calls continue serving from the
            // replacement.
            let needsDrain = state.withLock { s -> Bool in
                if !s.drainedPastCutover {
                    s.drainedPastCutover = true
                    return true
                }
                return false
            }
            if needsDrain {
                // Drain the live message 4: 1 type-byte + 12 GCM
                // nonce + 32 ciphertext + 16 tag = 61 bytes.
                _ = try await inner.recv(1 + 12 + 32 + 16)
            }

            return try state.withLock { s -> Result<Data, Error> in
                guard s.replayBuffer.count >= count else {
                    return .failure(ClusterHandshakeError.truncatedMessage)
                }
                let chunk = Data(s.replayBuffer.prefix(count))
                s.replayBuffer.removeFirst(count)
                return .success(chunk)
            }.get()
        }
    }
}

#endif // DEBUG
