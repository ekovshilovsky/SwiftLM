// Cluster formation orchestrator. Drives the end-to-end create / join
// flow on top of the lower-level primitives already defined in this
// module:
//
//   - `BonjourService` advertises + discovers peers over mDNS and
//     exposes an AsyncStream of inbound NWConnections on the
//     coordinator side.
//   - `ClusterAuth` provides the Argon2id passphrase derivation and
//     the HKDF-based discovery hash.
//   - `ClusterHandshake` runs the three-message HMAC mutual auth
//     plus AES-GCM-sealed working-key delivery.
//   - `ClusterKeyStore` persists the resulting (clusterId, workingKey)
//     pair so the node can rejoin the cluster across restarts.
//
// ClusterManager is an actor because it mutates its own and the
// BonjourService's advertised-info state in response to events
// arriving from multiple sources (coordinator's accept loop, joiner's
// outbound connect path, CLI-triggered create/join calls). Declaring
// the type an actor keeps all of that serialized under Swift's own
// concurrency model rather than by ad-hoc locking.
//
// Scope for v1 (Phase 4) is a single cluster formation per manager:
// one create or one join, then `listPeers()` passes through the
// BonjourService's stream. Rotation, heartbeats, and persistent-state
// recovery on startup are explicitly out of scope.

import Foundation
import Network
import Security

/// Orchestrator actor for cluster create / join flows. Owns a
/// `BonjourService` and a `ClusterKeyStore`; delegates advertise /
/// discover to the former and final persistence to the latter.
public actor ClusterManager {

    // MARK: - Configuration

    private let keyStore: ClusterKeyStore
    private let bonjour: BonjourService
    private let localHostname: String
    private let model: String
    private let memoryGB: Int
    private let version: String
    private let rdma: RdmaCapability

    // MARK: - Mutable state

    /// The persisted cluster record once create / join has completed.
    /// Held in addition to what the key store returns so subsequent
    /// `listPeers()` calls do not have to hit storage again, and so
    /// the manager can refuse a second create / join in-process.
    private var record: ClusterRecord?

    /// Task running the coordinator accept loop for any inbound join
    /// attempts. Started by `createCluster(...)`; cancelled on manager
    /// deinit is not possible from an actor, so callers that need a
    /// clean teardown should call `stop()` (future).
    private var coordinatorAcceptTask: Task<Void, Never>?

    /// Set to true after the underlying BonjourService has been
    /// started. Guards against a second start on repeated
    /// create / join calls.
    private var bonjourStarted: Bool = false

    // MARK: - Init

    /// Construct a manager with the injected dependencies. The
    /// BonjourService is passed in (not constructed here) because
    /// callers need to decide the advertised node name and the
    /// service-name-on-wire — and because tests drive two managers
    /// in one process with distinct BonjourService instances.
    public init(
        keyStore: ClusterKeyStore,
        bonjour: BonjourService,
        localHostname: String,
        model: String,
        memoryGB: Int,
        version: String,
        rdma: RdmaCapability
    ) {
        self.keyStore = keyStore
        self.bonjour = bonjour
        self.localHostname = localHostname
        self.model = model
        self.memoryGB = memoryGB
        self.version = version
        self.rdma = rdma
    }

    // MARK: - Create

    /// Create a new cluster. Generates a fresh UUID + random 32-byte
    /// working key, derives the handshake key from `(passphrase,
    /// uuid)`, persists the resulting `ClusterRecord`, reconfigures
    /// the Bonjour advertisement to broadcast `role = .coordinator`
    /// along with the cluster's discovery hash and UUID, starts the
    /// listener if it has not been started yet, and spawns a task
    /// that services inbound join handshakes against the generated
    /// working key. Returns the `ClusterRecord` for the caller's
    /// convenience; the same record has already been saved.
    public func createCluster(
        passphrase: String,
        clusterName: String?
    ) async throws -> ClusterRecord {
        _ = clusterName  // reserved for future UI; not used on the wire

        let uuid = UUID()
        let uuidBytes = Self.uuidToData(uuid)
        let workingKey = try Self.randomBytes(count: 32)

        let handshakeKey = ClusterAuth.deriveHandshakeKey(
            passphrase: passphrase,
            salt: uuidBytes
        )
        let discoveryHash = ClusterAuth.deriveDiscoveryHash(master: handshakeKey)

        let record = ClusterRecord(clusterId: uuidBytes, key: workingKey)
        try keyStore.save(record)
        self.record = record

        // Subscribe to incoming connections BEFORE starting the
        // listener so we do not race a fast-arriving joiner that
        // connects in the gap between listener-ready and
        // continuation-installed.
        let incoming = bonjour.incomingConnections()

        let info = DiscoveryInfo(
            model: model,
            memoryGB: memoryGB,
            role: .coordinator,
            clusterHash: discoveryHash,
            clusterId: uuid.uuidString,
            version: version,
            rdma: rdma,
            name: localHostname
        )

        if !bonjourStarted {
            bonjour.updateAdvertisedInfo(info)
            _ = try await bonjour.start()
            bonjourStarted = true
        } else {
            bonjour.updateAdvertisedInfo(info)
        }
        bonjour.setLocalClusterHash(discoveryHash)

        // Kick off the accept loop. Each inbound connection runs the
        // coordinator side of the handshake with the same handshake
        // key and working key; failures are logged (in real code we'd
        // route them through a logger) but do not take the manager
        // down — a bad passphrase from one joiner must not lock the
        // cluster against subsequent legitimate joiners.
        let handshakeKeyForAcceptor = handshakeKey
        let workingKeyForAcceptor = workingKey
        coordinatorAcceptTask = Task {
            for await connection in incoming {
                // Spawn a child task per connection so a slow or
                // hanging joiner cannot block the next one. The
                // handshake itself is fast but the TCP transition to
                // .ready can stall under load.
                Task {
                    let endpoint = NWConnectionHandshakeEndpoint(
                        connection: connection
                    )
                    do {
                        try await endpoint.waitReady()
                        try await runClusterHandshakeCoordinator(
                            handshakeKey: handshakeKeyForAcceptor,
                            workingClusterKey: workingKeyForAcceptor,
                            endpoint: endpoint
                        )
                    } catch {
                        // Intentionally swallowed — see comment above.
                        // A structured logger belongs here in Phase 5.
                    }
                    connection.cancel()
                }
            }
        }

        return record
    }

    // MARK: - Join

    /// Join an existing cluster advertised by `peer`. Reads the
    /// cluster UUID from the peer's TXT record, derives the
    /// handshake key locally from the supplied passphrase, opens a
    /// fresh TCP connection to the peer's endpoint, runs the joiner
    /// side of the handshake, and on success persists the delivered
    /// working key before reconfiguring the Bonjour advertisement to
    /// reflect the new role. Throws on any handshake failure; the
    /// key store is NOT mutated on failure so a retry with the
    /// correct passphrase starts from a clean slate.
    public func joinCluster(
        passphrase: String,
        peer: DiscoveredPeer
    ) async throws -> ClusterRecord {
        guard let clusterIdString = peer.info.clusterId,
              let uuid = UUID(uuidString: clusterIdString) else {
            throw ClusterManagerError.peerMissingClusterId
        }
        let uuidBytes = Self.uuidToData(uuid)

        let handshakeKey = ClusterAuth.deriveHandshakeKey(
            passphrase: passphrase,
            salt: uuidBytes
        )

        let connection = NWConnection(to: peer.endpoint, using: .tcp)
        let endpoint = NWConnectionHandshakeEndpoint(connection: connection)

        // The handshake is the only failure surface we persist on —
        // both the connect and the handshake are covered by one
        // do / catch so a mid-flight failure tears the connection
        // down before surfacing.
        let workingKey: Data
        do {
            try await endpoint.waitReady()
            workingKey = try await runClusterHandshakeJoiner(
                handshakeKey: handshakeKey,
                endpoint: endpoint
            )
        } catch {
            connection.cancel()
            throw error
        }
        connection.cancel()

        let record = ClusterRecord(clusterId: uuidBytes, key: workingKey)
        try keyStore.save(record)
        self.record = record

        let discoveryHash = ClusterAuth.deriveDiscoveryHash(master: handshakeKey)
        let info = DiscoveryInfo(
            model: model,
            memoryGB: memoryGB,
            role: .worker,
            clusterHash: discoveryHash,
            clusterId: uuid.uuidString,
            version: version,
            rdma: rdma,
            name: localHostname
        )
        bonjour.updateAdvertisedInfo(info)
        bonjour.setLocalClusterHash(discoveryHash)

        return record
    }

    // MARK: - Peer discovery

    /// Pass-through to `BonjourService.peers()`. Filtering by the
    /// local cluster hash (so only same-cluster peers appear, or —
    /// before cluster join — every visible cluster) is already
    /// handled inside BonjourService. ClusterManager does not add a
    /// second filter layer.
    public nonisolated func listPeers() -> AsyncStream<DiscoveredPeer> {
        return bonjour.peers()
    }

    // MARK: - Helpers

    /// Convert a UUID to its 16-byte big-endian representation.
    /// `ClusterRecord.clusterId` and the Argon2id salt are both this
    /// byte layout; keep the conversion in one place so every call
    /// site agrees on endianness.
    private static func uuidToData(_ uuid: UUID) -> Data {
        var bytes = uuid.uuid
        return withUnsafeBytes(of: &bytes) { Data($0) }
    }

    /// `SecRandomCopyBytes` wrapper with the handshake module's
    /// error vocabulary. Used to mint the 32-byte working cluster
    /// key at create time.
    private static func randomBytes(count: Int) throws -> Data {
        var buffer = Data(count: count)
        let status = buffer.withUnsafeMutableBytes { rawBuffer -> Int32 in
            guard let base = rawBuffer.baseAddress else { return errSecParam }
            return SecRandomCopyBytes(kSecRandomDefault, count, base)
        }
        guard status == errSecSuccess else {
            throw ClusterManagerError.randomBytesFailed(status: status)
        }
        return buffer
    }
}

// MARK: - Errors

/// Errors specific to ClusterManager. Handshake-level failures
/// surface as `ClusterHandshakeError`; transport and state-machine
/// failures land here.
public enum ClusterManagerError: Error, CustomStringConvertible {
    /// The discovered peer's TXT record did not include a parseable
    /// cluster UUID. Usually indicates a misconfigured advertiser or
    /// a peer still in `.discovering` state.
    case peerMissingClusterId

    /// `SecRandomCopyBytes` returned a non-success status. Practically
    /// unreachable on macOS but we refuse to pretend the return value
    /// is infallible.
    case randomBytesFailed(status: OSStatus)

    /// The NWConnection failed to reach `.ready` before a timeout or
    /// was torn down by the framework. Wraps the surfaced NWError so
    /// callers can diagnose which side gave up.
    case connectionFailed(NWError?)

    /// Generic connection-cancelled state surfaced while waiting for
    /// `.ready` or during a send/recv. Distinct from
    /// `connectionFailed` because cancellation is an expected
    /// shutdown path, not a transport fault.
    case connectionCancelled

    public var description: String {
        switch self {
        case .peerMissingClusterId:
            return "ClusterManager: discovered peer did not advertise a valid cluster UUID"
        case .randomBytesFailed(let status):
            return "ClusterManager: SecRandomCopyBytes failed with status \(status)"
        case .connectionFailed(let err):
            return "ClusterManager: NWConnection failed — \(err.map(String.init(describing:)) ?? "unknown")"
        case .connectionCancelled:
            return "ClusterManager: NWConnection was cancelled before the handshake completed"
        }
    }
}

// MARK: - NWConnection handshake adapter

/// Adapter that implements `ClusterHandshakeEndpoint` on top of an
/// NWConnection. The adapter is a class (not a struct) so the
/// NWConnection instance is stable under capture in async callbacks
/// and so state transitions mutate a shared ready-state flag rather
/// than a value-type copy.
///
/// Lifecycle:
///
///   1. Caller constructs the adapter around an NWConnection.
///   2. Caller calls `waitReady()` which starts the connection on an
///      internal queue and awaits transition to `.ready`.
///   3. Caller invokes send / recv; each await maps to exactly one
///      NWConnection.send or receive call (for send) or to a
///      short-read loop that accumulates `count` bytes (for recv).
///   4. Caller `.cancel()`s the NWConnection directly when done.
final class NWConnectionHandshakeEndpoint: ClusterHandshakeEndpoint, @unchecked Sendable {

    let connection: NWConnection
    private let queue = DispatchQueue(label: "turboquant.cluster.handshake.endpoint")

    init(connection: NWConnection) {
        self.connection = connection
    }

    /// Start the connection and await transition to `.ready`. If the
    /// connection was already started (coordinator-accept path:
    /// BonjourService starts inbound connections before yielding
    /// them) and is already ready, the call returns immediately.
    /// Subsequent calls are also idempotent — useful because
    /// send / recv do not themselves block on `.ready`.
    func waitReady() async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            // Capture `continuation resolved` state so the handler
            // cannot resume the continuation twice if the connection
            // bounces between states rapidly.
            let box = ContinuationBox(cont)
            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    box.tryResume(returning: ())
                case .failed(let err):
                    box.tryResume(throwing: ClusterManagerError.connectionFailed(err))
                case .cancelled:
                    box.tryResume(throwing: ClusterManagerError.connectionCancelled)
                default:
                    break
                }
            }
            // If the connection has already been started elsewhere
            // (coordinator path) start is a no-op on most NW states
            // and returns without mutating anything; double-start
            // with .setup -> .preparing -> .ready is the typical
            // joiner flow.
            connection.start(queue: queue)
        }
    }

    /// Send exactly `data.count` bytes, awaiting completion before
    /// returning. `isComplete` is false (we are not closing the
    /// stream half) and the context is default so NW's own framing
    /// does not add boundaries — the handshake's framing is handled
    /// at the byte level by the peer.
    func send(_ data: Data) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let box = ContinuationBox(cont)
            connection.send(
                content: data,
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
    }

    /// Receive exactly `count` bytes, accumulating across however
    /// many NWConnection.receive calls it takes. The underlying
    /// receive can return short even when `minimumIncompleteLength`
    /// is set to the remaining length — when the connection sees a
    /// graceful close it returns whatever buffered bytes remain with
    /// `isComplete == true`, so the loop has to detect that and
    /// raise `.truncatedMessage` rather than spinning.
    func recv(_ count: Int) async throws -> Data {
        var buffer = Data()
        buffer.reserveCapacity(count)
        while buffer.count < count {
            let remaining = count - buffer.count
            let chunk = try await receiveChunk(
                minimum: remaining,
                maximum: remaining
            )
            if chunk.isEmpty {
                // receiveChunk returns empty on isComplete + no
                // bytes; treat as EOF mid-message.
                throw ClusterHandshakeError.truncatedMessage
            }
            buffer.append(chunk)
        }
        return buffer
    }

    private func receiveChunk(minimum: Int, maximum: Int) async throws -> Data {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Data, Error>) in
            let box = ContinuationBox(cont)
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
                    // Graceful close with no bytes: caller interprets
                    // as truncated-message in `recv`.
                    box.tryResume(returning: Data())
                    return
                }
                // No data, no close, no error — NW should not do
                // this with a non-zero minimum, but defend against it
                // by surfacing a truncated-message error rather than
                // a silent hang.
                box.tryResume(throwing: ClusterHandshakeError.truncatedMessage)
            }
        }
    }
}

/// Small wrapper that lets a single CheckedContinuation be resumed
/// from multiple callback sites without triggering Swift's
/// runtime double-resume trap. The NW state machine does not
/// guarantee exclusive callbacks (a failed/cancelled transition can
/// fire after an initial .ready, for example), so wrapping is the
/// safest way to make the adapter robust.
private final class ContinuationBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var resumed = false
    private let continuation: CheckedContinuation<T, Error>

    init(_ continuation: CheckedContinuation<T, Error>) {
        self.continuation = continuation
    }

    func tryResume(returning value: T) {
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
