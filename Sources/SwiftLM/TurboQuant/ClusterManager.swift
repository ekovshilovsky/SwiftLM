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
// Scope for v1 is a single cluster formation per manager:
// one create or one join, then `listPeers()` passes through the
// BonjourService's stream. Rotation, heartbeats, and persistent-state
// recovery on startup are explicitly out of scope.

import CryptoKit
import Foundation
import MLX
import MLXLMCommon
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
    /// attempts. Started by `createCluster(...)` and cancelled by
    /// `stop()`. The underlying `for await connection in incoming`
    /// loop also exits when the injected `BonjourService` finishes
    /// the `incomingConnections()` stream, so either path tears the
    /// loop down cleanly.
    private var coordinatorAcceptTask: Task<Void, Never>?

    /// Set to true after `stop()` has been invoked so the manager
    /// refuses to start a second create / join flow. Construct a new
    /// manager instance to re-enter the cluster lifecycle.
    private var stopped: Bool = false

    /// Set to true after the underlying BonjourService has been
    /// started. Guards against a second start on repeated
    /// create / join calls.
    private var bonjourStarted: Bool = false

    /// Joiner-side data channel built from the live coordinator
    /// connection once `joinCluster` completes its handshake. Stays
    /// alive for the lifetime of the cluster session so the worker
    /// can pump control messages off it; `nil` until the handshake
    /// succeeds and after `stop()` has torn the channel down.
    private var coordinatorChannel: ClusterDataChannel?

    /// Coordinator-side data channels, one per joiner that has
    /// completed the handshake. Appended to from inside the
    /// accept-loop's per-connection task each time a fresh handshake
    /// succeeds. Drained on `stop()` so the underlying connections
    /// release cleanly.
    private var joinerChannels: [ClusterDataChannel] = []

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
        guard !stopped else {
            throw ClusterManagerError.managerStopped
        }
        // `clusterName` is accepted in the public signature to keep
        // the CLI layer free to pass a user-visible label
        // through without a signature change later. It is deliberately
        // not advertised on the wire — the Bonjour TXT `cluster` field
        // holds the passphrase-derived discovery hash, not a name.
        _ = clusterName

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
        //
        // On a successful handshake the connection is NOT cancelled.
        // A `ClusterDataChannel` is built around the live connection
        // so the post-handshake control flow can use it; the channel
        // is appended to `joinerChannels` for `beginInferenceSession`
        // to pick up.
        let handshakeKeyForAcceptor = handshakeKey
        let workingKeyForAcceptor = workingKey
        coordinatorAcceptTask = Task { [weak self] in
            for await connection in incoming {
                // Spawn a child task per connection so a slow or
                // hanging joiner cannot block the next one. The
                // handshake itself is fast but the TCP transition to
                // .ready can stall under load.
                Task { [weak self] in
                    let endpoint = NWConnectionHandshakeEndpoint(
                        connection: connection,
                        alreadyStarted: true
                    )
                    let outcome: ClusterHandshakeCoordinatorOutcome
                    do {
                        try await endpoint.waitReady()
                        outcome = try await runClusterHandshakeCoordinator(
                            handshakeKey: handshakeKeyForAcceptor,
                            workingClusterKey: workingKeyForAcceptor,
                            endpoint: endpoint
                        )
                    } catch {
                        // Intentionally swallowed — see comment above.
                        // A structured logger belongs here later.
                        connection.cancel()
                        return
                    }
                    // Handshake succeeded: keep the connection live,
                    // wrap it in a data channel, and register the
                    // channel on the manager. The append must hop back
                    // onto the actor's executor since this task is
                    // detached from the actor's isolation domain.
                    let channel = ClusterDataChannel(
                        connection: connection,
                        sessionKey: outcome.sessionKey,
                        role: .coordinator
                    )
                    channel.startReceiving()
                    await self?.registerJoinerChannel(channel)
                }
            }
        }

        return record
    }

    /// Actor-isolated entry point for the coordinator's accept loop to
    /// register a freshly-handshook joiner's data channel. Drops the
    /// channel cleanly if the manager has already been stopped so a
    /// late-arriving handshake does not resurrect torn-down state.
    private func registerJoinerChannel(_ channel: ClusterDataChannel) {
        guard !stopped else {
            channel.close()
            return
        }
        joinerChannels.append(channel)
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
        guard !stopped else {
            throw ClusterManagerError.managerStopped
        }
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
        // down before surfacing. On success the connection is kept
        // live for the post-handshake data channel to use.
        let outcome: ClusterHandshakeJoinerOutcome
        do {
            try await endpoint.waitReady()
            outcome = try await runClusterHandshakeJoiner(
                handshakeKey: handshakeKey,
                endpoint: endpoint
            )
        } catch {
            connection.cancel()
            throw error
        }

        let record = ClusterRecord(clusterId: uuidBytes, key: outcome.workingKey)
        try keyStore.save(record)
        self.record = record

        // Wrap the live connection in a joiner-role data channel so
        // the worker loop can pump control messages off it once the
        // caller invokes `runJoinerWorker(model:)`. The channel is
        // started immediately so it does not miss a coordinator
        // message that arrives between handshake completion and the
        // worker spinning up.
        let channel = ClusterDataChannel(
            connection: connection,
            sessionKey: outcome.sessionKey,
            role: .joiner
        )
        channel.startReceiving()
        coordinatorChannel = channel

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

    // MARK: - Shutdown

    /// Release all resources this manager owns: cancels the
    /// coordinator accept loop (if one was started by
    /// `createCluster(...)`) and stops the injected `BonjourService`,
    /// which tears down the underlying `NWListener` and `NWBrowser`
    /// and finishes the `peers()` and `incomingConnections()` async
    /// streams.
    ///
    /// Idempotent — calling `stop()` twice is safe. After `stop()`
    /// the manager is terminal: further `createCluster` or
    /// `joinCluster` calls throw `ClusterManagerError.managerStopped`.
    /// Callers that need to re-enter the lifecycle must construct a
    /// fresh `ClusterManager` with a fresh `BonjourService` (the
    /// stopped service cannot be restarted either).
    public func stop() {
        guard !stopped else { return }
        stopped = true
        coordinatorAcceptTask?.cancel()
        coordinatorAcceptTask = nil
        // Tear down any post-handshake data channels before stopping
        // the discovery service. Closing each channel cancels its
        // underlying NWConnection and finishes the inbound stream so
        // worker loops observing the channel exit cleanly.
        if let channel = coordinatorChannel {
            channel.close()
            coordinatorChannel = nil
        }
        for channel in joinerChannels {
            channel.close()
        }
        joinerChannels.removeAll()
        bonjour.stop()
        bonjourStarted = false
    }

    // MARK: - Peer discovery

    /// Pass-through to `BonjourService.peers()`. Filtering by the
    /// local cluster hash (so only same-cluster peers appear, or —
    /// before cluster join — every visible cluster) is already
    /// handled inside BonjourService. ClusterManager does not add a
    /// second filter layer.
    ///
    /// Marked `nonisolated` because the implementation reads only the
    /// `bonjour` stored property (an immutable `let` captured at
    /// init) and calls `BonjourService.peers()`, which is itself
    /// thread-safe via an internal dispatch queue. No actor-isolated
    /// state is touched, so the call does not need to suspend into
    /// the actor's executor — callers consume the stream directly.
    public nonisolated func listPeers() -> AsyncStream<DiscoveredPeer> {
        return bonjour.peers()
    }

    // MARK: - Runtime layer

    /// Spawn a joiner-side worker against the live data channel built
    /// at handshake time and run its loop to completion. Returns when
    /// the channel terminates (peer EOF, transport failure, or
    /// `stop()` closing the channel) or when the caller's surrounding
    /// task is cancelled.
    ///
    /// The closures are constructed here from the supplied model so
    /// the worker stays decoupled from `DistributedQwenModel`'s
    /// concrete shape — the same pattern the coordinator session uses.
    ///
    /// **Call site convention.** This method blocks for the cluster's
    /// useful lifetime, so production callers wrap it in a detached
    /// `Task` and store the returned handle for graceful shutdown:
    ///
    /// ```swift
    /// let workerTask = Task { try await manager.runJoinerWorker(model: model) }
    /// // ... on SIGTERM:
    /// workerTask.cancel()
    /// ```
    ///
    /// Cancelling the surrounding task closes the channel and unblocks
    /// the worker loop cleanly.
    public func runJoinerWorker(model: DistributedQwenModel) async throws {
        guard let channel = coordinatorChannel else {
            throw ClusterManagerError.noActiveJoinerChannel
        }
        let worker = JoinerInferenceWorker(
            channel: channel,
            forward: { tokenIds, cache in model(tokenIds, cache: cache) },
            makeCache: { model.makeCache() }
        )
        await worker.run()
    }

    /// Drive one inference request to completion across all currently-
    /// connected joiner channels and stream the resulting token events
    /// to the caller. The model is captured by reference so its
    /// per-process state (resident weights, configuration) is shared
    /// across every request.
    ///
    /// Size-1 clusters (no joiners attached yet) are valid: the engine
    /// runs the local model with an empty broadcast list, which
    /// matches the validated single-rank inference path.
    ///
    /// **Joiner snapshot semantics.** The joiner-channel list is
    /// captured at session-start; joiners that complete their
    /// handshake after this method is called are not included in the
    /// session's broadcast list. Callers that need a specific
    /// world-size should wait for all joiners to attach before
    /// starting inference (e.g. block on the cluster's expected size
    /// in a setup phase, then start serving requests). Mixing in-flight
    /// handshakes with active sessions is supported but produces
    /// non-deterministic world sizes per request.
    ///
    /// Marked `async` because the engine's `generate` entry point is
    /// actor-isolated; the awaited hop happens once at session-start
    /// and the returned stream itself is consumed without further
    /// actor traffic.
    public func beginInferenceSession(
        model: DistributedQwenModel,
        request: CoordinatorInferenceSession.InferenceRequest
    ) async -> AsyncThrowingStream<CoordinatorInferenceSession.TokenEvent, Error> {
        let engine = DistributedInferenceEngine(
            model: model,
            joiners: joinerChannels
        )
        return await engine.generate(request)
    }

    /// Closure-driven variant of `beginInferenceSession` for tests and
    /// for callers that already wrap a model with their own forward /
    /// cache adapters. Skips the `DistributedInferenceEngine`'s model
    /// indirection and constructs a per-request session directly so
    /// the broadcast wiring across the manager's joiner channels can
    /// be exercised without standing up a `DistributedQwenModel`
    /// instance. Production callers should prefer the
    /// model-parameterised overload above.
    public func beginInferenceSession(
        request: CoordinatorInferenceSession.InferenceRequest,
        forward: @escaping CoordinatorInferenceSession.ForwardFn,
        makeCache: @escaping CoordinatorInferenceSession.MakeCacheFn,
        sampler: any DistributedSampler = GreedySampler()
    ) -> AsyncThrowingStream<CoordinatorInferenceSession.TokenEvent, Error> {
        let joiners: [InferenceControlChannel] = joinerChannels
        let session = CoordinatorInferenceSession(
            forward: forward,
            makeCache: makeCache,
            joiners: joiners,
            sampler: sampler
        )
        return AsyncThrowingStream { continuation in
            // Bridge the session's `AsyncStream<TokenEvent>` continuation
            // shape to the outer `AsyncThrowingStream<TokenEvent>`.
            // Mirrors the engine's drain loop so a session error
            // surfaces on the consumer's `for try await` loop and a
            // cancellation closes the outer stream cleanly.
            let (sessionStream, sessionContinuation) = AsyncStream
                .makeStream(of: CoordinatorInferenceSession.TokenEvent.self)

            let serveTask = Task {
                let serveResult: Task<Void, Error> = Task {
                    try await session.serve(request, stream: sessionContinuation)
                }
                for await event in sessionStream {
                    continuation.yield(event)
                }
                do {
                    try await serveResult.value
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                serveTask.cancel()
            }
        }
    }

    /// Number of joiners that have completed the cluster handshake and
    /// are currently registered on this coordinator. Returned as a
    /// simple count rather than the channel list so callers cannot
    /// retain references to the underlying transport. The value is a
    /// snapshot at the time the actor services the call; the
    /// chat-completions router reads it to decide whether the
    /// distributed-inference path has any peers to fan out to.
    public func attachedJoinerCount() -> Int {
        return joinerChannels.count
    }

    // MARK: - Test affordances

    /// Snapshot of the current joiner-channel list. Internal-only so
    /// tests can assert on the post-handshake channel registration
    /// without exposing the underlying storage to the public API.
    internal func _joinerChannelsForTesting() -> [ClusterDataChannel] {
        return joinerChannels
    }

    /// Snapshot of the joiner-side coordinator channel. Internal-only
    /// for the same reason as `_joinerChannelsForTesting()`.
    internal func _coordinatorChannelForTesting() -> ClusterDataChannel? {
        return coordinatorChannel
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

    /// `createCluster` or `joinCluster` was invoked after `stop()`.
    /// The manager is terminal once stopped; callers must construct
    /// a fresh instance to re-enter the cluster lifecycle.
    case managerStopped

    /// `runJoinerWorker` was invoked before `joinCluster` succeeded
    /// (or after `stop()` tore the channel down). The worker has no
    /// transport to bind to without a live coordinator channel.
    case noActiveJoinerChannel

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
        case .managerStopped:
            return "ClusterManager: operation attempted after stop() — construct a new instance"
        case .noActiveJoinerChannel:
            return "ClusterManager: runJoinerWorker invoked without an active coordinator channel — call joinCluster first"
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
    private let alreadyStarted: Bool

    init(connection: NWConnection, alreadyStarted: Bool = false) {
        self.connection = connection
        self.alreadyStarted = alreadyStarted
    }

    /// Install the connection's state update handler and await the
    /// transition to `.ready`. Two distinct ownership cases:
    ///
    ///  - Joiner (outbound) path: the adapter owns the connection and
    ///    is responsible for starting it. `alreadyStarted` is false;
    ///    the handler is installed first, then `start(queue:)` is
    ///    invoked exactly once.
    ///
    ///  - Coordinator (inbound) path: BonjourService has already
    ///    called `start(queue:)` on the connection on its own internal
    ///    queue before yielding it. `NWConnection.start` is one-shot
    ///    and calling it again — especially with a different queue —
    ///    is undefined, so this path must not start the connection a
    ///    second time. `alreadyStarted` is true; the handler is
    ///    installed onto the running connection.
    ///
    /// In both cases the handler is installed before any start call,
    /// so the normal `.setup` → `.preparing` → `.ready` transitions
    /// will fire the handler. For the coordinator path the connection
    /// may have already reached a terminal state before the handler
    /// was installed (NWConnection does not guarantee handler
    /// replay), so after installation the current state is consulted
    /// inline and the continuation is resolved directly when it is
    /// already terminal. `ContinuationBox` is idempotent so any
    /// subsequent real state callback is a no-op.
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

            // Only start the connection when the adapter owns it.
            // Starting a connection that is already running (and on a
            // different queue) is undefined per NWConnection's
            // contract.
            if !alreadyStarted {
                connection.start(queue: queue)
            }

            // Cover the race where the connection reached a terminal
            // state before the handler above was installed. Use
            // pattern matching rather than `==` because
            // `NWConnection.State` is not reliably `Equatable` across
            // SDK versions. `ContinuationBox.tryResume` is idempotent
            // so if a real state callback fires afterwards it becomes
            // a no-op.
            switch connection.state {
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
