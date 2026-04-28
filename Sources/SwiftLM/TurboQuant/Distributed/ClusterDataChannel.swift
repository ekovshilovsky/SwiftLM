// Encrypted, length-prefix-framed control channel that carries
// `InferenceControlMessage` values between two cluster peers over an
// already-authenticated `NWConnection`.
//
// This file owns three concerns:
//
//   1. Direction-keyed AEAD via AES-GCM. Two distinct symmetric keys
//      are derived from the post-handshake `sessionKey` using
//      HKDF-SHA256 with disjoint `info` strings — one key protects
//      coordinator-to-joiner traffic, the other protects
//      joiner-to-coordinator traffic. Keying the two directions
//      separately means a frame captured on one direction cannot be
//      replayed in the other: GCM tag verification fails because the
//      open key differs from the seal key. Direction separation also
//      eliminates the "same key, same nonce" hazard that would
//      otherwise need extra care to avoid.
//
//   2. Per-message nonce construction with a strictly monotonic
//      counter. Each direction maintains an independent send counter
//      starting at zero. The 12-byte nonce concatenates an 8-byte
//      big-endian counter with a 4-byte direction-specific salt; the
//      AAD repeats the counter and a one-byte direction tag so the
//      receive side can authenticate exactly the framing context it
//      observes. The receive side rejects any frame whose counter
//      does not strictly exceed the highest counter accepted so far,
//      which simultaneously catches replays and forces overflow to
//      surface as a hard error rather than wrapping silently.
//
//   3. Length-prefix framing on the wire. Every frame is
//
//          | 4 bytes BE length | 12-byte nonce | ciphertext+16-byte tag |
//
//      where `length` covers everything after the prefix. Reads
//      consume exactly that many bytes — short reads loop until the
//      full frame is buffered — so two frames coalesced into one
//      `NWConnection.send` call are still decoded as two distinct
//      messages on the receive side.
//
// Authentication of the underlying connection is the upstream
// handshake's responsibility (`ClusterHandshake.swift`); this type
// presupposes a successful handshake has produced `sessionKey` and
// that both peers agree on each other's identity. What it adds is
// confidentiality and integrity for every subsequent control frame
// plus replay protection within the lifetime of the connection.

import CryptoKit
import Foundation
import Network

// MARK: - Data-channel key derivation

/// HKDF-Expand subkeys derived from the handshake's `sessionKey` for
/// bulk data-channel encryption. Direction separation is enforced at
/// the key level so a coordinator-side frame can never be opened with
/// the joiner-side key and vice versa, even if an attacker could
/// otherwise produce a colliding nonce.
///
/// The `info` strings are stable wire constants — they form part of
/// the cryptographic contract between the two peers and must match
/// exactly on both sides. They are documented here in one place to
/// avoid silent drift.
public enum ClusterDataKeys {

    /// Domain-separation tag for the coordinator-to-joiner direction.
    public static let coordToJoinerInfo = "tq-data-c2j"

    /// Domain-separation tag for the joiner-to-coordinator direction.
    public static let joinerToCoordInfo = "tq-data-j2c"

    /// Width in bytes of each derived data-channel key. AES-GCM with
    /// a 256-bit key is the standard choice and matches the rest of
    /// the cluster's symmetric-crypto surface.
    public static let dataKeyLength = 32

    /// Derive the coordinator-to-joiner data key from the handshake
    /// session key. The HKDF salt is omitted; per RFC 5869 §2.2 the
    /// no-salt variant defaults to a `HashLen`-byte zero string,
    /// which is acceptable here because the input key material
    /// (`sessionKey`) is already a uniformly-random 256-bit value
    /// produced by `ClusterAuth.deriveSessionKey` from the handshake.
    /// Domain separation between the two directions is provided by
    /// the distinct `info` strings.
    public static func coordToJoinerKey(sessionKey: SymmetricKey) -> SymmetricKey {
        return HKDF<SHA256>.deriveKey(
            inputKeyMaterial: sessionKey,
            info: Data(coordToJoinerInfo.utf8),
            outputByteCount: dataKeyLength
        )
    }

    /// Derive the joiner-to-coordinator data key from the handshake
    /// session key. See the `coordToJoinerKey` doc for the rationale
    /// behind omitting the HKDF salt.
    public static func joinerToCoordKey(sessionKey: SymmetricKey) -> SymmetricKey {
        return HKDF<SHA256>.deriveKey(
            inputKeyMaterial: sessionKey,
            info: Data(joinerToCoordInfo.utf8),
            outputByteCount: dataKeyLength
        )
    }
}

// MARK: - Errors

/// Failure cases surfaced by `ClusterDataChannel`. Each case is
/// distinct enough that callers can map them onto user-facing copy or
/// onto specific telemetry metrics; none are merged into a generic
/// "transport error" bucket.
public enum ClusterDataChannelError: Error, CustomStringConvertible {

    /// The underlying `NWConnection` reached `cancelled`, `failed`,
    /// or graceful EOF before the next frame could be read or
    /// written.
    case connectionClosed

    /// The receive side observed end-of-stream after consuming part
    /// of a 4-byte length prefix. The frame format guarantees length
    /// prefixes never split, so this represents a peer that closed
    /// mid-frame or a transport error mid-prefix.
    case lengthPrefixTruncated

    /// The receive side observed end-of-stream after consuming the
    /// length prefix but before reading `length` bytes of payload.
    case payloadTruncated

    /// The decoded frame's counter did not strictly exceed the
    /// highest counter accepted so far on this direction. Indicates
    /// replay, reordering, or peer-side counter regression — all of
    /// which the channel rejects.
    case nonceCounterReuseDetected(received: UInt64, lastAccepted: UInt64)

    /// AES-GCM tag verification failed. Caused by tampering, a frame
    /// flowing in the wrong direction (and so opened with the wrong
    /// key), or a key-derivation mismatch between the two peers.
    case decryptionFailed

    /// JSON decoding of the plaintext payload failed. Carries the
    /// underlying decoder error for diagnostics.
    case decodeFailed(underlying: Error)

    /// The send-side counter has reached `UInt64.max` and refuses to
    /// emit further frames. This is unreachable in practice — at one
    /// frame per nanosecond it would take 584 years — but the API
    /// surface refuses to wrap silently.
    case sendCounterExhausted

    /// A frame's wire shape was malformed in a way that cannot be
    /// attributed to any of the more specific cases. The associated
    /// string is a diagnostic detail, not a user-facing message.
    case malformedFrame(String)

    public var description: String {
        switch self {
        case .connectionClosed:
            return "cluster data channel: connection closed"
        case .lengthPrefixTruncated:
            return "cluster data channel: length prefix truncated"
        case .payloadTruncated:
            return "cluster data channel: payload truncated before frame complete"
        case .nonceCounterReuseDetected(let received, let lastAccepted):
            return "cluster data channel: nonce counter reuse — received \(received), last accepted \(lastAccepted)"
        case .decryptionFailed:
            return "cluster data channel: AES-GCM decryption failed"
        case .decodeFailed(let underlying):
            return "cluster data channel: payload JSON decode failed — \(underlying)"
        case .sendCounterExhausted:
            return "cluster data channel: send counter exhausted"
        case .malformedFrame(let detail):
            return "cluster data channel: malformed frame — \(detail)"
        }
    }
}

// MARK: - Direction tags

/// Per-direction byte identifier mixed into the AEAD AAD. Distinct
/// from the keying domain-separation tag so that AAD authentication
/// fails (independently of GCM keying) if a frame is somehow
/// misrouted into the wrong receive loop.
private enum DirectionTag {
    static let coordToJoiner: UInt8 = 0x01
    static let joinerToCoord: UInt8 = 0x02
}

/// Per-direction four-byte salt mixed into the 12-byte AES-GCM
/// nonce. The salts are constant and public — their job is purely to
/// keep the two directions' nonce spaces disjoint as defence in
/// depth on top of the per-direction keying. Both peers must agree
/// on these values byte-for-byte.
private enum NonceSalt {
    static let coordToJoiner: [UInt8] = [0xC2, 0x4A, 0x00, 0x00]
    static let joinerToCoord: [UInt8] = [0x4A, 0xC2, 0x00, 0x00]
}

// MARK: - Frame layout constants

private enum FrameLayout {

    /// Width of the length prefix in bytes. The prefix is encoded big-
    /// endian and covers everything that follows it (nonce +
    /// ciphertext + tag), but not itself.
    static let lengthPrefixBytes = 4

    /// AES-GCM nonce size in bytes (RFC 5116 §4 standard width).
    static let nonceBytes = 12

    /// AES-GCM tag size in bytes.
    static let tagBytes = 16

    /// Counter portion of the nonce, in bytes. The remaining bytes of
    /// the nonce hold the direction-specific salt.
    static let counterBytes = 8
}

// MARK: - ClusterDataChannel

/// Bidirectional cluster data channel. Wraps an `NWConnection` that
/// has completed the cluster handshake and now carries
/// AES-GCM-encrypted, length-prefix-framed control messages between
/// two authenticated peers. Constructed by `ClusterManager` once the
/// handshake produces a session key; this type does not perform
/// authentication itself — it assumes the connection is already
/// authenticated by the upstream handshake.
///
/// `@unchecked Sendable`: every mutation of the send counter happens
/// inside `sendQueue.sync`, every mutation of the receive counter
/// and inbound continuation happens on a single `Task` started by
/// `startReceiving()`, and the lifecycle flags (`receiveStarted`,
/// `closed`) live behind a dedicated lock. The compiler cannot prove
/// these constraints from the storage shapes alone, hence the
/// explicit annotation.
public final class ClusterDataChannel: InferenceControlChannel, @unchecked Sendable {

    /// The peer's role on this channel. Determines which derived key
    /// is used for sends versus receives and which direction tag /
    /// nonce salt are mixed into the AEAD inputs.
    public enum Role: Sendable {

        /// This endpoint sends with the coordinator-to-joiner key and
        /// receives with the joiner-to-coordinator key.
        case coordinator

        /// This endpoint sends with the joiner-to-coordinator key and
        /// receives with the coordinator-to-joiner key.
        case joiner
    }

    // Underlying transport. `@unchecked Sendable` requires the
    // surrounding type to track its own thread-safety; NWConnection
    // is documented as safe to call into from any queue once started.
    private let connection: NWConnection

    // Dedicated dispatch queue for NWConnection callbacks. Network.framework
    // requires a queue parameter for `start(queue:)` and dispatches all
    // completion handlers onto it.
    private let connectionQueue: DispatchQueue

    // Direction-keyed AEAD keys derived once at init.
    private let sendKey: SymmetricKey
    private let receiveKey: SymmetricKey

    // Direction tags and nonce salts, captured at init so the send
    // and receive paths can stay branch-free per call. The receive
    // direction's salt is not stored — the nonce is read straight
    // off the wire and authenticated via the per-direction `receiveKey`,
    // which fails to open any frame whose nonce was constructed from
    // the wrong salt or counter pair.
    private let sendDirectionTag: UInt8
    private let receiveDirectionTag: UInt8
    private let sendNonceSalt: [UInt8]

    // Send-side counter. Mutated only inside `sendQueue.sync` so the
    // monotonic-increment guarantee holds across any number of
    // concurrent senders.
    private let sendQueue = DispatchQueue(label: "turboquant.cluster.data-channel.send")
    private var sendCounter: UInt64 = 0

    // Receive-side counter. Mutated only by the single receive
    // `Task` started inside `startReceiving()`; no external lock is
    // necessary because the loop is the sole writer.
    private var lastAcceptedReceiveCounter: UInt64 = 0
    private var hasAcceptedAnyReceiveFrame = false

    // Lifecycle flags. The lifecycle lock guards both flags so a
    // concurrent `close()` versus `startReceiving()` pair cannot
    // race on `receiveStarted`.
    private let lifecycleLock = NSLock()
    private var receiveStarted = false
    private var closed = false

    /// Handle for the detached receive task started by
    /// `startReceiving()`. Stored so `close()` can cancel an in-flight
    /// `NWConnection.receive` rather than waiting for the connection's
    /// own cancellation to unblock the read.
    private var receiveTask: Task<Void, Never>?

    // Inbound stream + its continuation. The continuation is
    // resolved by the receive `Task` (yields on each successful
    // decode, finishes on any error or peer EOF).
    public let inbound: AsyncStream<InferenceControlMessage>
    private let inboundContinuation: AsyncStream<InferenceControlMessage>.Continuation

    /// Construct a channel around an already-authenticated
    /// `NWConnection`. The connection must have completed the cluster
    /// handshake, and `sessionKey` must be the value that handshake
    /// produced for this peer pair. The connection's lifecycle is
    /// handed to the channel — `close()` cancels it, and the channel
    /// expects no other writer.
    public init(
        connection: NWConnection,
        sessionKey: SymmetricKey,
        role: Role
    ) {
        self.connection = connection
        self.connectionQueue = DispatchQueue(
            label: "turboquant.cluster.data-channel.connection"
        )

        switch role {
        case .coordinator:
            self.sendKey = ClusterDataKeys.coordToJoinerKey(sessionKey: sessionKey)
            self.receiveKey = ClusterDataKeys.joinerToCoordKey(sessionKey: sessionKey)
            self.sendDirectionTag = DirectionTag.coordToJoiner
            self.receiveDirectionTag = DirectionTag.joinerToCoord
            self.sendNonceSalt = NonceSalt.coordToJoiner
        case .joiner:
            self.sendKey = ClusterDataKeys.joinerToCoordKey(sessionKey: sessionKey)
            self.receiveKey = ClusterDataKeys.coordToJoinerKey(sessionKey: sessionKey)
            self.sendDirectionTag = DirectionTag.joinerToCoord
            self.receiveDirectionTag = DirectionTag.coordToJoiner
            self.sendNonceSalt = NonceSalt.joinerToCoord
        }

        var continuation: AsyncStream<InferenceControlMessage>.Continuation!
        self.inbound = AsyncStream(
            InferenceControlMessage.self,
            bufferingPolicy: .unbounded
        ) { c in
            continuation = c
        }
        self.inboundContinuation = continuation
    }

    // MARK: - Public API

    /// Begin reading from the connection and emitting decoded
    /// messages onto `inbound`. Idempotent: the second and subsequent
    /// calls return immediately without spawning additional reader
    /// tasks. Must be called once after init for the channel to
    /// deliver any inbound messages.
    public func startReceiving() {
        let shouldStart: Bool = lifecycleLock.withLock {
            if receiveStarted || closed { return false }
            receiveStarted = true
            return true
        }
        guard shouldStart else { return }

        // Start the connection on our dedicated queue. NWConnection
        // requires this before any send or receive can complete; in
        // tests a freshly constructed `NWConnection` arrives in
        // `.setup`, so the channel takes care of moving it to
        // `.ready` itself rather than relying on the upstream
        // handshake having already done it.
        let needsStart = (connection.state.needsStart)
        if needsStart {
            connection.start(queue: connectionQueue)
        }

        // Hold a strong reference for the loop's lifetime. The channel
        // owns its connection and is owned by ClusterManager, so the
        // task naturally outlives at most the channel's `close()`
        // call; cancellation is what tears the loop down, not garbage
        // collection. A `[weak self]` capture would let the loop
        // silently drop without finishing `inboundContinuation` if the
        // channel were ever deallocated mid-flight, leaving consumers
        // hung on `inbound`.
        receiveTask = Task.detached {
            await self.runReceiveLoop()
        }
    }

    /// Cancel the underlying connection and finish `inbound`. Safe to
    /// call from any task; subsequent calls are no-ops. Cancels the
    /// receive task so a stuck `NWConnection.receive` does not leave
    /// the loop alive after the channel has been closed.
    public func close() {
        let shouldClose: Bool = lifecycleLock.withLock {
            if closed { return false }
            closed = true
            return true
        }
        guard shouldClose else { return }
        receiveTask?.cancel()
        connection.cancel()
        inboundContinuation.finish()
    }

    /// Encode, encrypt, and frame `message`, then transmit it on the
    /// connection. The send path serialises through `sendQueue` so
    /// concurrent callers cannot race on the counter and the
    /// underlying NWConnection write order matches the counter order.
    public func send(_ message: InferenceControlMessage) async throws {
        let frame = try sendQueue.sync { () throws -> Data in
            guard sendCounter != UInt64.max else {
                throw ClusterDataChannelError.sendCounterExhausted
            }
            let counter = sendCounter
            // Build BEFORE advancing the counter so a throw from
            // `buildOutboundFrame` (theoretical: a future change that
            // adds a fallible step would expose this) does not leave
            // the channel in a state where the counter has advanced
            // past the actual last-sent frame. `lastTransmittedFrame`
            // and `sendCounter` are mutated together so the test
            // affordance always reflects what actually went on the wire.
            let built = try buildOutboundFrame(message: message, counter: counter)
            sendCounter += 1
            lastTransmittedFrame = built
            return built
        }
        try await transmit(frame)
    }

    // MARK: - Frame construction

    /// Encode the message via the shared codec, AES-GCM-seal the
    /// resulting bytes with a counter-derived nonce, and prepend the
    /// 4-byte big-endian length prefix. Counter and direction tag
    /// travel through the AEAD's AAD so any in-flight tampering with
    /// the counter is caught by GCM tag verification.
    private func buildOutboundFrame(
        message: InferenceControlMessage,
        counter: UInt64
    ) throws -> Data {
        let plaintext = try InferenceControlCodec.encode(message)
        let nonceBytes = makeNonceBytes(counter: counter, salt: sendNonceSalt)
        let nonce: AES.GCM.Nonce
        do {
            nonce = try AES.GCM.Nonce(data: nonceBytes)
        } catch {
            throw ClusterDataChannelError.malformedFrame(
                "AES.GCM.Nonce construction failed: \(error)"
            )
        }
        let aad = makeAAD(counter: counter, directionTag: sendDirectionTag)
        let sealed: AES.GCM.SealedBox
        do {
            sealed = try AES.GCM.seal(
                plaintext,
                using: sendKey,
                nonce: nonce,
                authenticating: aad
            )
        } catch {
            throw ClusterDataChannelError.malformedFrame(
                "AES.GCM.seal failed: \(error)"
            )
        }

        // Sealed frame body = nonce (12) | ciphertext | tag (16).
        var body = Data()
        body.reserveCapacity(FrameLayout.nonceBytes + sealed.ciphertext.count + FrameLayout.tagBytes)
        body.append(nonceBytes)
        body.append(sealed.ciphertext)
        body.append(sealed.tag)

        let totalLength = UInt32(body.count)
        var lengthBE = totalLength.bigEndian
        var frame = Data()
        frame.reserveCapacity(FrameLayout.lengthPrefixBytes + body.count)
        withUnsafeBytes(of: &lengthBE) { frame.append(contentsOf: $0) }
        frame.append(body)
        return frame
    }

    /// Compose the 12-byte nonce as `counter (BE 8) | salt (4)`. The
    /// salt is held constant per direction so the nonce is uniquely
    /// determined by the per-direction counter.
    private func makeNonceBytes(counter: UInt64, salt: [UInt8]) -> Data {
        precondition(salt.count == FrameLayout.nonceBytes - FrameLayout.counterBytes,
                     "nonce salt width must complement the counter width")
        var bytes = Data()
        bytes.reserveCapacity(FrameLayout.nonceBytes)
        var counterBE = counter.bigEndian
        withUnsafeBytes(of: &counterBE) { bytes.append(contentsOf: $0) }
        bytes.append(contentsOf: salt)
        return bytes
    }

    /// Compose the AEAD additional-data buffer as
    /// `counter (BE 8) | directionTag (1)`. Authenticating the counter
    /// here means a forwarding attacker cannot mutate the on-the-wire
    /// counter without invalidating the GCM tag — which protects the
    /// receive-side replay check from being bypassed by truncation
    /// attacks against the counter bytes inside the nonce.
    private func makeAAD(counter: UInt64, directionTag: UInt8) -> Data {
        var aad = Data()
        aad.reserveCapacity(FrameLayout.counterBytes + 1)
        var counterBE = counter.bigEndian
        withUnsafeBytes(of: &counterBE) { aad.append(contentsOf: $0) }
        aad.append(directionTag)
        return aad
    }

    // MARK: - Outbound transport

    /// Push a fully constructed frame onto the NWConnection. The
    /// ordering guarantee from `sendQueue` is preserved because we
    /// hand the frame to NW's own ordered send pipeline before
    /// returning.
    ///
    /// The send is wrapped in `withTaskCancellationHandler` so a
    /// cancelled surrounding task tears the underlying connection
    /// down rather than waiting indefinitely for an unresponsive
    /// peer's TCP receive window to drain. Without this, a stalled
    /// joiner's send blocks the coordinator's broadcast loop and the
    /// HTTP handler's request-timeout cancellation cannot unstick it.
    private func transmit(_ frame: Data) async throws {
        let connectionRef = connection
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                let box = SendContinuationBox(cont)
                connectionRef.send(
                    content: frame,
                    contentContext: .defaultMessage,
                    isComplete: false,
                    completion: .contentProcessed { error in
                        if let error {
                            box.tryResume(throwing: error)
                        } else {
                            box.tryResume(returning: ())
                        }
                    }
                )
            }
        } onCancel: {
            // Closes the underlying socket; any in-flight
            // `NWConnection.send` surfaces a cancellation error
            // through the completion handler, which the box
            // forwards to the suspended continuation.
            connectionRef.cancel()
        }
    }

    // MARK: - Receive loop

    /// Pump frames off the connection, decrypt, decode, and yield to
    /// `inbound`. Terminates on the first error condition (peer EOF,
    /// truncated frame, replay, decryption failure, decode failure)
    /// after which the channel is closed and the inbound stream is
    /// finished.
    private func runReceiveLoop() async {
        do {
            while true {
                if isClosed() { break }
                let lengthBytes = try await readExact(FrameLayout.lengthPrefixBytes,
                                                     onEOF: .lengthPrefixTruncated)
                let length = parseLengthPrefix(lengthBytes)
                guard length >= UInt32(FrameLayout.nonceBytes + FrameLayout.tagBytes) else {
                    throw ClusterDataChannelError.malformedFrame(
                        "frame length \(length) below nonce + tag floor"
                    )
                }
                let body = try await readExact(Int(length), onEOF: .payloadTruncated)
                let message = try processInboundFrame(body)
                inboundContinuation.yield(message)
            }
        } catch let error as ClusterDataChannelError {
            // Record the error before finishing the continuation. The
            // inbound stream's consumer typically polls `lastReceiveError`
            // immediately after observing the stream finish, so the
            // ordering of these two operations is observable: setting
            // the error first guarantees the consumer sees a populated
            // `lastReceiveError` rather than racing with the recording
            // path. `close()` then drops the underlying connection so
            // the peer can no longer write into a half-shut transport.
            recordLastReceiveError(error)
            inboundContinuation.finish()
            close()
        } catch {
            // Network framework surfaces transport teardown as
            // POSIX or NWError values rather than ClusterDataChannelError
            // cases. Map them onto `.connectionClosed` so callers can
            // distinguish "peer closed mid-frame" from genuine framing
            // bugs without inspecting the underlying error type.
            recordLastReceiveError(.connectionClosed)
            inboundContinuation.finish()
            close()
        }
    }

    /// Decrypt and decode one frame body (nonce + ciphertext + tag),
    /// enforcing the strict-monotonic counter rule before handing the
    /// plaintext to the JSON codec. Each failure mode maps to a
    /// distinct `ClusterDataChannelError` case so adversarial tests
    /// can assert on the exact rejection reason.
    private func processInboundFrame(_ body: Data) throws -> InferenceControlMessage {
        guard body.count >= FrameLayout.nonceBytes + FrameLayout.tagBytes else {
            throw ClusterDataChannelError.malformedFrame(
                "frame body \(body.count) bytes < nonce + tag floor"
            )
        }
        let nonceBytes = body.prefix(FrameLayout.nonceBytes)
        let remainder = body.suffix(from: body.startIndex + FrameLayout.nonceBytes)
        let tagStart = remainder.endIndex - FrameLayout.tagBytes
        let ciphertext = remainder[remainder.startIndex..<tagStart]
        let tag = remainder[tagStart..<remainder.endIndex]

        let receivedCounter = parseCounter(from: nonceBytes)

        // Replay / regression check. The peer's send counter starts
        // at 0 and increments by exactly 1 per frame; the receive side
        // enforces the same invariant. The first accepted frame must
        // have counter == 0; subsequent frames must be exactly one
        // greater than the last accepted counter (strict-monotonic).
        // Out-of-order or duplicated frames surface as
        // `nonceCounterReuseDetected` rather than silently advancing
        // the receive watermark.
        if hasAcceptedAnyReceiveFrame {
            guard receivedCounter > lastAcceptedReceiveCounter else {
                throw ClusterDataChannelError.nonceCounterReuseDetected(
                    received: receivedCounter,
                    lastAccepted: lastAcceptedReceiveCounter
                )
            }
        } else {
            guard receivedCounter == 0 else {
                throw ClusterDataChannelError.nonceCounterReuseDetected(
                    received: receivedCounter,
                    lastAccepted: 0
                )
            }
        }

        let nonce: AES.GCM.Nonce
        do {
            nonce = try AES.GCM.Nonce(data: nonceBytes)
        } catch {
            throw ClusterDataChannelError.malformedFrame(
                "AES.GCM.Nonce reconstruction failed: \(error)"
            )
        }
        let sealed: AES.GCM.SealedBox
        do {
            sealed = try AES.GCM.SealedBox(
                nonce: nonce,
                ciphertext: ciphertext,
                tag: tag
            )
        } catch {
            throw ClusterDataChannelError.malformedFrame(
                "AES.GCM.SealedBox reconstruction failed: \(error)"
            )
        }

        let aad = makeAAD(counter: receivedCounter, directionTag: receiveDirectionTag)
        let plaintext: Data
        do {
            plaintext = try AES.GCM.open(sealed, using: receiveKey, authenticating: aad)
        } catch {
            throw ClusterDataChannelError.decryptionFailed
        }

        let message: InferenceControlMessage
        do {
            message = try InferenceControlCodec.decode(plaintext)
        } catch {
            throw ClusterDataChannelError.decodeFailed(underlying: error)
        }

        // Counter is committed only after the frame has been fully
        // authenticated and decoded. A failed decryption or decode
        // does not advance the counter; this preserves the invariant
        // that `lastAcceptedReceiveCounter` always reflects an
        // accepted frame.
        lastAcceptedReceiveCounter = receivedCounter
        hasAcceptedAnyReceiveFrame = true
        return message
    }

    private func parseLengthPrefix(_ bytes: Data) -> UInt32 {
        return bytes.withUnsafeBytes { raw -> UInt32 in
            let storage = raw.load(as: UInt32.self)
            return UInt32(bigEndian: storage)
        }
    }

    private func parseCounter(from nonceBytes: Data) -> UInt64 {
        let counterPart = nonceBytes.prefix(FrameLayout.counterBytes)
        return counterPart.withUnsafeBytes { raw -> UInt64 in
            let storage = raw.load(as: UInt64.self)
            return UInt64(bigEndian: storage)
        }
    }

    /// Read exactly `count` bytes from the connection, looping over
    /// short reads. Maps EOF onto a caller-supplied error so the
    /// length-prefix and payload phases can surface distinct
    /// truncation cases.
    private func readExact(
        _ count: Int,
        onEOF eofError: ClusterDataChannelError
    ) async throws -> Data {
        var buffer = Data()
        buffer.reserveCapacity(count)
        while buffer.count < count {
            let remaining = count - buffer.count
            let chunk = try await receiveChunk(
                minimum: remaining,
                maximum: remaining
            )
            if chunk.isEmpty {
                throw eofError
            }
            buffer.append(chunk)
        }
        return buffer
    }

    /// Single `NWConnection.receive` call wrapped as an async
    /// continuation. Returns an empty `Data` on graceful EOF so the
    /// caller can decide whether that constitutes a truncated frame
    /// or a clean shutdown.
    private func receiveChunk(minimum: Int, maximum: Int) async throws -> Data {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Data, Error>) in
            let box = ReceiveContinuationBox(cont)
            connection.receive(
                minimumIncompleteLength: minimum,
                maximumLength: maximum
            ) { content, _, isComplete, error in
                if let error {
                    box.tryResume(throwing: error)
                    return
                }
                if let data = content, !data.isEmpty {
                    box.tryResume(returning: data)
                    return
                }
                if isComplete {
                    box.tryResume(returning: Data())
                    return
                }
                box.tryResume(throwing: ClusterDataChannelError.connectionClosed)
            }
        }
    }

    private func isClosed() -> Bool {
        lifecycleLock.withLock { closed }
    }

    // MARK: - Diagnostics

    private let errorLock = NSLock()
    private var _lastReceiveError: ClusterDataChannelError?

    /// Last error observed by the receive loop, if any. Exposed
    /// internally so tests can assert on the precise reason a channel
    /// closed itself; production callers should rely on `inbound`
    /// finishing as the close signal.
    internal var lastReceiveError: ClusterDataChannelError? {
        errorLock.withLock { _lastReceiveError }
    }

    private func recordLastReceiveError(_ error: ClusterDataChannelError) {
        errorLock.withLock { _lastReceiveError = error }
    }

    // MARK: - Test affordances

    /// Replay the most recently transmitted frame, if any, without
    /// advancing the send counter. Internal-only helper for replay-
    /// rejection testing — production senders never re-emit a frame.
    /// The receive side must reject the duplicate via its monotonic-
    /// counter check.
    internal func _replayLastFrameForTesting() async throws {
        let snapshot = sendQueue.sync { () -> Data? in lastTransmittedFrame }
        guard let frame = snapshot else { return }
        try await transmit(frame)
    }

    /// Emit an arbitrary byte sequence on the underlying connection
    /// without any framing or encryption. Internal-only helper used
    /// to drive truncation and framing-alignment scenarios from
    /// tests.
    internal func _sendRawForTesting(_ data: Data) async throws {
        try await transmit(data)
    }

    /// The most recently transmitted frame's bytes, captured for the
    /// replay-test helper. Tracked inside `sendQueue` for the same
    /// serialization reason the counter is.
    private var lastTransmittedFrame: Data?
}

// MARK: - Continuation helpers

/// Single-resume wrapper for the send-completion callback. The
/// network framework does not promise the completion fires exactly
/// once across all error and cancellation paths, so the channel
/// guards against double-resume by routing every callback through
/// this box.
private final class SendContinuationBox: @unchecked Sendable {
    private let lock = NSLock()
    private var resumed = false
    private let continuation: CheckedContinuation<Void, Error>

    init(_ continuation: CheckedContinuation<Void, Error>) {
        self.continuation = continuation
    }

    func tryResume(returning value: Void) {
        lock.lock()
        let first = !resumed
        resumed = true
        lock.unlock()
        if first { continuation.resume(returning: value) }
    }

    func tryResume(throwing error: Error) {
        lock.lock()
        let first = !resumed
        resumed = true
        lock.unlock()
        if first { continuation.resume(throwing: error) }
    }
}

/// Single-resume wrapper for receive-callback continuations. Same
/// rationale as `SendContinuationBox`.
private final class ReceiveContinuationBox: @unchecked Sendable {
    private let lock = NSLock()
    private var resumed = false
    private let continuation: CheckedContinuation<Data, Error>

    init(_ continuation: CheckedContinuation<Data, Error>) {
        self.continuation = continuation
    }

    func tryResume(returning value: Data) {
        lock.lock()
        let first = !resumed
        resumed = true
        lock.unlock()
        if first { continuation.resume(returning: value) }
    }

    func tryResume(throwing error: Error) {
        lock.lock()
        let first = !resumed
        resumed = true
        lock.unlock()
        if first { continuation.resume(throwing: error) }
    }
}

// MARK: - NWConnection helper

extension NWConnection.State {

    /// True when the state indicates the connection has not yet been
    /// started. Used by `ClusterDataChannel.startReceiving` to avoid
    /// calling `start(queue:)` on a connection that the upstream
    /// handshake already started — `NWConnection.start` is one-shot
    /// and a second call (especially with a different queue) is
    /// undefined.
    fileprivate var needsStart: Bool {
        switch self {
        case .setup:
            return true
        default:
            return false
        }
    }
}
