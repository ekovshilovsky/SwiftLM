// Three-message HMAC mutual-auth handshake plus an AES-GCM-sealed
// working-key delivery message. Both sides of the join flow call
// into this module with pre-connected endpoints and the locally-
// derived handshake key.
//
// The protocol is:
//
//   Joiner → Coord:  0x01 | nonceJoiner
//   Coord  → Joiner: 0x02 | nonceCoord | HMAC(hmacKey, 0x02 | nonceCoord | nonceJoiner)
//   Joiner → Coord:  0x03 | HMAC(hmacKey, 0x03 | nonceJoiner | nonceCoord)
//   Coord  → Joiner: 0x04 | nonceGCM | GCM.seal(workingKey, sessionKey, nonceGCM,
//                                               aad: 0x04 | nonceJoiner | nonceCoord)
//
// The type byte is the first byte of each HMAC input so a captured
// message-2 MAC cannot be replayed as a message-3 MAC — their HMAC
// inputs differ in that first byte even when an attacker manipulates
// the nonces. Nonce ordering also differs between the two MAC inputs
// (joiner-then-coord for message 3, coord-then-joiner for message 2)
// as a second layer of cross-message protection.
//
// Message 4's associated data binds the ciphertext to the specific
// handshake nonces: a captured message 4 replayed into a fresh session
// cannot be opened because the joiner computes AAD from its own fresh
// nonces and AES-GCM's tag verification fails.
//
// Forward secrecy is intentionally not provided — a full Diffie–Hellman
// exchange would be needed for that and is deferred.

import CryptoKit
import Foundation

// MARK: - Protocol constants

private enum Proto {
    static let msgJoinerHello: UInt8   = 0x01
    static let msgCoordChallenge: UInt8 = 0x02
    static let msgJoinerResponse: UInt8 = 0x03
    static let msgSealedKey: UInt8      = 0x04

    static let nonceHandshakeLength = 32   // matches HKDF salt width on deriveSessionKey
    static let nonceGCMLength       = 12   // AES-GCM standard nonce size
    static let hmacLength           = 32   // SHA-256 output width
    static let gcmTagLength         = 16   // AES-GCM standard tag size
}

// MARK: - Transport abstraction

/// Abstraction over a bidirectional byte transport. Production callers
/// wrap an `NWConnection` or equivalent; tests wrap an in-process pipe.
/// The protocol is synchronous-looking — each call awaits until the
/// request completes — which keeps the handshake code linear.
public protocol ClusterHandshakeEndpoint: Sendable {
    /// Transmit exactly `data.count` bytes. Implementations must either
    /// send them all or throw.
    func send(_ data: Data) async throws

    /// Receive exactly `count` bytes. Implementations must return
    /// exactly that many bytes (retrying on short reads) or throw on
    /// EOF / transport error.
    func recv(_ count: Int) async throws -> Data
}

// MARK: - Errors

public enum ClusterHandshakeError: Error, Equatable, CustomStringConvertible {
    /// The peer's HMAC did not verify. Reserved strictly for this case
    /// so callers can surface "wrong passphrase or tampering" without
    /// confusing it with a broken connection.
    case peerAuthFailed

    /// A message-type byte did not match what was expected at this step.
    case protocolVersionMismatch

    /// The transport returned EOF or fewer bytes than requested before
    /// the current message was fully delivered.
    case truncatedMessage

    /// AES-GCM tag verification failed on the sealed working-key
    /// message. Indicates a replayed or tampered message 4, or in
    /// practice a mismatched handshake key that slipped past the
    /// HMAC checks (not possible in the honest protocol but worth
    /// distinguishing from peerAuthFailed).
    case sealOpenFailed

    /// A failure inside this module's own implementation — RNG failure,
    /// bad nonce length passed to AES-GCM constructor, etc. Not
    /// attributable to the peer or the network.
    case unexpectedInternalError(String)

    public var description: String {
        switch self {
        case .peerAuthFailed:
            return "cluster handshake: peer HMAC verification failed"
        case .protocolVersionMismatch:
            return "cluster handshake: unexpected message-type byte"
        case .truncatedMessage:
            return "cluster handshake: message truncated"
        case .sealOpenFailed:
            return "cluster handshake: AES-GCM seal failed to open"
        case .unexpectedInternalError(let detail):
            return "cluster handshake: internal error — \(detail)"
        }
    }
}

// MARK: - Random-byte helper

// The crypto subsystem's RNG is effectively infallible on macOS, but
// SecRandomCopyBytes returns a status and the handshake surface refuses
// to pretend otherwise — a failure here means the platform is in an
// unrecoverable state and the caller needs to know.
private func randomBytes(_ count: Int) throws -> Data {
    var buffer = Data(count: count)
    let status = buffer.withUnsafeMutableBytes { rawBuffer -> Int32 in
        guard let base = rawBuffer.baseAddress else { return errSecParam }
        return SecRandomCopyBytes(kSecRandomDefault, count, base)
    }
    guard status == errSecSuccess else {
        throw ClusterHandshakeError.unexpectedInternalError(
            "SecRandomCopyBytes failed with status \(status)"
        )
    }
    return buffer
}

// MARK: - HMAC helpers

// Both sides build HMAC inputs with identical byte layouts. Centralizing
// the concatenation keeps a single source of truth for the exact wire
// format and prevents one side from drifting.

private func coordinatorChallengeMAC(
    hmacKey: SymmetricKey,
    nonceCoordinator: Data,
    nonceJoiner: Data
) -> Data {
    // Input: type-byte (0x02) | nonceCoord | nonceJoiner
    var input = Data()
    input.append(Proto.msgCoordChallenge)
    input.append(nonceCoordinator)
    input.append(nonceJoiner)
    let mac = HMAC<SHA256>.authenticationCode(for: input, using: hmacKey)
    return Data(mac)
}

private func joinerResponseMAC(
    hmacKey: SymmetricKey,
    nonceJoiner: Data,
    nonceCoordinator: Data
) -> Data {
    // Input: type-byte (0x03) | nonceJoiner | nonceCoord
    // Order differs from the coordinator's MAC on purpose — if the two
    // MACs consumed identically-ordered nonces, compromising one would
    // be materially closer to forging the other.
    var input = Data()
    input.append(Proto.msgJoinerResponse)
    input.append(nonceJoiner)
    input.append(nonceCoordinator)
    let mac = HMAC<SHA256>.authenticationCode(for: input, using: hmacKey)
    return Data(mac)
}

private func isCoordinatorChallengeMACValid(
    hmacKey: SymmetricKey,
    nonceCoordinator: Data,
    nonceJoiner: Data,
    candidate: Data
) -> Bool {
    var input = Data()
    input.append(Proto.msgCoordChallenge)
    input.append(nonceCoordinator)
    input.append(nonceJoiner)
    return HMAC<SHA256>.isValidAuthenticationCode(
        candidate,
        authenticating: input,
        using: hmacKey
    )
}

private func isJoinerResponseMACValid(
    hmacKey: SymmetricKey,
    nonceJoiner: Data,
    nonceCoordinator: Data,
    candidate: Data
) -> Bool {
    var input = Data()
    input.append(Proto.msgJoinerResponse)
    input.append(nonceJoiner)
    input.append(nonceCoordinator)
    return HMAC<SHA256>.isValidAuthenticationCode(
        candidate,
        authenticating: input,
        using: hmacKey
    )
}

// MARK: - AAD construction

// The associated data threaded through AES-GCM on message 4 is built
// identically on both sides. If this byte layout ever diverges between
// coordinator and joiner the seal simply fails to open — but writing
// it in one place removes the opportunity for that drift.
private func sealedKeyAAD(nonceJoiner: Data, nonceCoordinator: Data) -> Data {
    var aad = Data()
    aad.append(Proto.msgSealedKey)
    aad.append(nonceJoiner)
    aad.append(nonceCoordinator)
    return aad
}

// MARK: - Handshake outcomes

/// Outcome of a successful coordinator-side handshake. The caller can
/// pair the returned `sessionKey` with the live transport to construct
/// a `ClusterDataChannel` for the post-handshake control flow.
public struct ClusterHandshakeCoordinatorOutcome: Sendable {

    /// Per-handshake symmetric key derived from the handshake key and
    /// the negotiated nonce pair. Used by the upper layer to seed the
    /// data-channel encryption keys for this peer pairing.
    public let sessionKey: SymmetricKey

    public init(sessionKey: SymmetricKey) {
        self.sessionKey = sessionKey
    }
}

/// Outcome of a successful joiner-side handshake. The working cluster
/// key is the persisted credential the joiner stores in its key store;
/// `sessionKey` is the per-handshake symmetric key the upper layer
/// uses to seed the data-channel encryption keys for this peer
/// pairing.
public struct ClusterHandshakeJoinerOutcome: Sendable {

    /// Working cluster key delivered by the coordinator under the
    /// AES-GCM-sealed final message. Persisted by the joiner's key
    /// store so the node can rejoin the cluster across restarts.
    public let workingKey: Data

    /// Per-handshake symmetric key derived from the handshake key and
    /// the negotiated nonce pair.
    public let sessionKey: SymmetricKey

    public init(workingKey: Data, sessionKey: SymmetricKey) {
        self.workingKey = workingKey
        self.sessionKey = sessionKey
    }
}

// MARK: - Coordinator side

/// Run the coordinator side of the handshake. The caller has already
/// accepted a TCP connection from a joiner and derived `handshakeKey`
/// from `(passphrase, clusterId)`. On success the coordinator has
/// transmitted `workingClusterKey` under an authenticated seal to the
/// joiner and the returned outcome carries the per-handshake session
/// key the upper layer needs to construct an encrypted data channel
/// for this peer pairing; on any failure the caller should close the
/// endpoint and decline this join attempt.
@discardableResult
public func runClusterHandshakeCoordinator(
    handshakeKey: Data,
    workingClusterKey: Data,
    endpoint: ClusterHandshakeEndpoint
) async throws -> ClusterHandshakeCoordinatorOutcome {
    let hmacKey = ClusterAuth.deriveHmacKey(handshakeKey: handshakeKey)

    // Message 1: read joiner's hello (type-byte + nonceJoiner).
    let helloType = try await endpoint.recv(1)
    guard helloType.first == Proto.msgJoinerHello else {
        throw ClusterHandshakeError.protocolVersionMismatch
    }
    let nonceJoiner = try await endpoint.recv(Proto.nonceHandshakeLength)

    // Message 2: send our own nonce plus our MAC over (type|coord|joiner).
    let nonceCoordinator = try randomBytes(Proto.nonceHandshakeLength)
    let challengeMAC = coordinatorChallengeMAC(
        hmacKey: hmacKey,
        nonceCoordinator: nonceCoordinator,
        nonceJoiner: nonceJoiner
    )
    var message2 = Data()
    message2.append(Proto.msgCoordChallenge)
    message2.append(nonceCoordinator)
    message2.append(challengeMAC)
    try await endpoint.send(message2)

    // Message 3: verify joiner's MAC over (type|joiner|coord). Only
    // after the joiner has proven possession of the handshake key do we
    // release the working cluster key on the wire.
    let responseType = try await endpoint.recv(1)
    guard responseType.first == Proto.msgJoinerResponse else {
        throw ClusterHandshakeError.protocolVersionMismatch
    }
    let responseMAC = try await endpoint.recv(Proto.hmacLength)
    guard isJoinerResponseMACValid(
        hmacKey: hmacKey,
        nonceJoiner: nonceJoiner,
        nonceCoordinator: nonceCoordinator,
        candidate: responseMAC
    ) else {
        throw ClusterHandshakeError.peerAuthFailed
    }

    // Message 4: derive the per-handshake session key and seal the
    // working cluster key under AES-GCM with AAD binding the ciphertext
    // to this specific pair of nonces.
    let sessionKey = ClusterAuth.deriveSessionKey(
        handshakeKey: handshakeKey,
        nonceJoiner: nonceJoiner,
        nonceCoordinator: nonceCoordinator
    )
    let nonceGCMBytes = try randomBytes(Proto.nonceGCMLength)
    let gcmNonce: AES.GCM.Nonce
    do {
        gcmNonce = try AES.GCM.Nonce(data: nonceGCMBytes)
    } catch {
        throw ClusterHandshakeError.unexpectedInternalError(
            "AES.GCM.Nonce construction failed: \(error)"
        )
    }
    let aad = sealedKeyAAD(
        nonceJoiner: nonceJoiner,
        nonceCoordinator: nonceCoordinator
    )
    let sealed: AES.GCM.SealedBox
    do {
        sealed = try AES.GCM.seal(
            workingClusterKey,
            using: sessionKey,
            nonce: gcmNonce,
            authenticating: aad
        )
    } catch {
        throw ClusterHandshakeError.unexpectedInternalError(
            "AES.GCM.seal failed: \(error)"
        )
    }

    var message4 = Data()
    message4.append(Proto.msgSealedKey)
    message4.append(nonceGCMBytes)
    message4.append(sealed.ciphertext)
    message4.append(sealed.tag)
    try await endpoint.send(message4)

    return ClusterHandshakeCoordinatorOutcome(sessionKey: sessionKey)
}

// MARK: - Joiner side

/// Run the joiner side of the handshake. The caller has already
/// connected to the coordinator and derived `handshakeKey` from
/// `(passphrase, clusterId)`. Returns the working cluster key delivered
/// by the coordinator alongside the per-handshake session key the
/// upper layer needs to construct an encrypted data channel for this
/// peer pairing. On any failure the caller should tear down the
/// endpoint and report the failure to the user — `.peerAuthFailed` in
/// particular maps to "wrong passphrase" in user-facing copy.
public func runClusterHandshakeJoiner(
    handshakeKey: Data,
    endpoint: ClusterHandshakeEndpoint
) async throws -> ClusterHandshakeJoinerOutcome {
    let hmacKey = ClusterAuth.deriveHmacKey(handshakeKey: handshakeKey)

    // Message 1: say hello with our fresh nonce. No MAC here — we have
    // nothing to prove yet, and the coordinator's message-2 MAC will
    // bind this nonce into its authentication input.
    let nonceJoiner = try randomBytes(Proto.nonceHandshakeLength)
    var message1 = Data()
    message1.append(Proto.msgJoinerHello)
    message1.append(nonceJoiner)
    try await endpoint.send(message1)

    // Message 2: read coordinator challenge. Verify its MAC under
    // constant-time comparison before sending any reply — responding to
    // an invalid MAC would leak information through timing or through
    // the reply being observable.
    let challengeType = try await endpoint.recv(1)
    guard challengeType.first == Proto.msgCoordChallenge else {
        throw ClusterHandshakeError.protocolVersionMismatch
    }
    let nonceCoordinator = try await endpoint.recv(Proto.nonceHandshakeLength)
    let challengeMAC = try await endpoint.recv(Proto.hmacLength)
    guard isCoordinatorChallengeMACValid(
        hmacKey: hmacKey,
        nonceCoordinator: nonceCoordinator,
        nonceJoiner: nonceJoiner,
        candidate: challengeMAC
    ) else {
        throw ClusterHandshakeError.peerAuthFailed
    }

    // Message 3: prove we also hold the handshake key by sending our
    // MAC over (type|joiner|coord).
    let responseMAC = joinerResponseMAC(
        hmacKey: hmacKey,
        nonceJoiner: nonceJoiner,
        nonceCoordinator: nonceCoordinator
    )
    var message3 = Data()
    message3.append(Proto.msgJoinerResponse)
    message3.append(responseMAC)
    try await endpoint.send(message3)

    // Message 4: open the sealed working key. The ciphertext+tag are
    // the remaining bytes after the type byte and the GCM nonce. The
    // length of the working key is implicitly the ciphertext length —
    // callers of this module agree elsewhere on what that key looks
    // like (currently 32 bytes — `ClusterAuth.masterKeyLength`).
    let sealedType = try await endpoint.recv(1)
    guard sealedType.first == Proto.msgSealedKey else {
        throw ClusterHandshakeError.protocolVersionMismatch
    }
    let nonceGCMBytes = try await endpoint.recv(Proto.nonceGCMLength)

    // We do not know the ciphertext length from the wire because it is
    // the working-key length, which is a deployment-wide constant. Use
    // the 32-byte working key size (`ClusterAuth.masterKeyLength`). If
    // this ever needs to change it becomes a length-prefixed field in a
    // new protocol version.
    let ciphertextLength = ClusterAuth.masterKeyLength
    let ciphertext = try await endpoint.recv(ciphertextLength)
    let tag = try await endpoint.recv(Proto.gcmTagLength)

    let sessionKey = ClusterAuth.deriveSessionKey(
        handshakeKey: handshakeKey,
        nonceJoiner: nonceJoiner,
        nonceCoordinator: nonceCoordinator
    )
    let gcmNonce: AES.GCM.Nonce
    do {
        gcmNonce = try AES.GCM.Nonce(data: nonceGCMBytes)
    } catch {
        throw ClusterHandshakeError.unexpectedInternalError(
            "AES.GCM.Nonce construction failed: \(error)"
        )
    }
    let sealed: AES.GCM.SealedBox
    do {
        sealed = try AES.GCM.SealedBox(
            nonce: gcmNonce,
            ciphertext: ciphertext,
            tag: tag
        )
    } catch {
        throw ClusterHandshakeError.unexpectedInternalError(
            "AES.GCM.SealedBox construction failed: \(error)"
        )
    }
    let aad = sealedKeyAAD(
        nonceJoiner: nonceJoiner,
        nonceCoordinator: nonceCoordinator
    )
    let workingKey: Data
    do {
        workingKey = try AES.GCM.open(sealed, using: sessionKey, authenticating: aad)
    } catch {
        throw ClusterHandshakeError.sealOpenFailed
    }
    return ClusterHandshakeJoinerOutcome(
        workingKey: workingKey,
        sessionKey: sessionKey
    )
}
