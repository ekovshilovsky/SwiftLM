// Active Bonjour advertise and browse service over Network.framework.
// Wires the previously-static DiscoveryInfo model (BonjourDiscovery.swift)
// into real _turboquant._tcp mDNS traffic: an NWListener publishes the
// local node's DiscoveryInfo via its TXT record, and an NWBrowser
// observes other nodes advertising the same service type and surfaces
// them as a de-duplicated async stream.
//
// Cluster-hash filter — the spec's §3.4 security property is applied at
// the subscription layer: the stream only yields peers whose
// DiscoveryInfo.clusterHash matches the local node's hash. Peers on a
// different cluster are therefore invisible to the caller even though
// their TXT records are technically visible on the wire — the
// filtering happens before any signal reaches ClusterManager. A nil
// local hash means the local node has not yet joined a cluster, in
// which case the filter passes everything through so the user can see
// which clusters are available to join.
//
// Endpoint surfacing — joiners need more than a DiscoveryInfo; they
// need an NWEndpoint to open a connection back. We wrap both together
// in DiscoveredPeer and emit that from peers() rather than extending
// DiscoveryInfo itself. The rationale: the wire-level TXT record has
// no concept of "where this record came from", and DiscoveryInfo
// round-trips through toTxtRecord()/init(fromTxtRecord:) in tests and
// storage paths. Adding a port field to DiscoveryInfo would either
// pollute that round-trip (advertised value always nil, received value
// populated) or require a second type anyway. A dedicated
// DiscoveredPeer keeps DiscoveryInfo pure and names the runtime-only
// association explicitly.

import Foundation
import Network

/// A peer discovered on the network. Pairs the decoded TXT record
/// with the NWEndpoint needed to open a connection. Consumers hand the
/// endpoint to `NWConnection(to:using:)` to open the join-handshake
/// transport; they do not need to resolve host/port separately because
/// NWConnection performs the mDNS resolution implicitly.
public struct DiscoveredPeer: Sendable, Equatable {
    public let info: DiscoveryInfo
    public let endpoint: NWEndpoint

    public init(info: DiscoveryInfo, endpoint: NWEndpoint) {
        self.info = info
        self.endpoint = endpoint
    }

    public static func == (lhs: DiscoveredPeer, rhs: DiscoveredPeer) -> Bool {
        // Endpoint equality is compared by description; NWEndpoint is not
        // Equatable in the public SDK but two endpoints constructed for
        // the same Bonjour service print identically.
        return lhs.info == rhs.info && lhs.endpoint.debugDescription == rhs.endpoint.debugDescription
    }
}

/// Errors surfaced by `BonjourService`. The listener-failed case is
/// distinguished from a general `NWError` so callers can branch on "we
/// never started advertising" versus "mDNS is broken in this
/// environment" — the latter is a legitimate skip reason for tests.
public enum BonjourServiceError: Error, CustomStringConvertible {
    case listenerFailed(NWError)
    case browserFailed(NWError)
    case notReady

    public var description: String {
        switch self {
        case .listenerFailed(let err): return "Bonjour listener failed: \(err)"
        case .browserFailed(let err): return "Bonjour browser failed: \(err)"
        case .notReady: return "Bonjour listener has not become ready"
        }
    }
}

/// Owns an `NWListener` and an `NWBrowser` for the TurboQuant Bonjour
/// service type. A single instance advertises the local node's
/// `DiscoveryInfo` and exposes an AsyncStream of discovered peers from
/// other nodes. Safe to call `start()` once per instance; create a new
/// instance to re-advertise with a changed `DiscoveryInfo`.
public final class BonjourService: @unchecked Sendable {

    // MARK: - Constants

    /// Bonjour service type used by all TurboQuant nodes. The `_tcp`
    /// suffix indicates the transport layer NWConnection will later
    /// use; mDNS itself is UDP but the advertised service is a TCP
    /// endpoint.
    public static let serviceType = "_turboquant._tcp"

    // MARK: - Configuration

    private let advertisedInfo: DiscoveryInfo
    private let advertisedServiceName: String

    // MARK: - Mutable state (guarded by queue)

    private let stateQueue = DispatchQueue(label: "turboquant.bonjour.state")
    private var listener: NWListener?
    private var browser: NWBrowser?
    private var localClusterHash: String?
    /// Continuation for the async port handoff; fulfilled exactly once
    /// when the listener reaches `.ready`.
    private var portContinuation: CheckedContinuation<NWEndpoint.Port, Error>?
    /// De-duplication key set so the same peer is not yielded twice.
    /// Keyed by `(clusterHash ?? "none") + "|" + name`, matching the
    /// spec's requirement that peers are deduped by cluster-hash + name.
    private var seenPeerKeys: Set<String> = []
    /// Active subscribers to the peers stream. Each started `peers()`
    /// call appends its continuation here; the browser handler fans
    /// out to all of them.
    private var peerContinuations: [UUID: AsyncStream<DiscoveredPeer>.Continuation] = [:]

    // MARK: - Init

    /// Construct a service that will advertise `info`. `localClusterHash`
    /// is the locally-derived 8-hex-char discovery hash from
    /// `ClusterAuth.deriveDiscoveryHash(master:)`; pass nil when the
    /// local node has not joined a cluster yet, in which case the
    /// peers() stream will not filter and every discovered cluster is
    /// visible. `serviceName` is the instance name registered in mDNS
    /// (distinct from the human-readable `info.name`, though they are
    /// typically the same); defaults to `info.name`.
    public init(info: DiscoveryInfo,
                localClusterHash: String? = nil,
                serviceName: String? = nil) {
        self.advertisedInfo = info
        self.advertisedServiceName = serviceName ?? info.name
        self.localClusterHash = localClusterHash
    }

    // MARK: - Public API

    /// Update the local cluster hash at runtime. Called by
    /// ClusterManager after a successful join so previously-visible
    /// peers on other clusters become invisible and the stream narrows
    /// to same-cluster peers only. Passing nil reverts to "see every
    /// cluster" mode.
    public func setLocalClusterHash(_ hash: String?) {
        stateQueue.async { [weak self] in
            guard let self else { return }
            self.localClusterHash = hash
            // Reset the de-dup set so peers that were previously
            // filtered out can be re-emitted under the new filter.
            self.seenPeerKeys.removeAll()
        }
    }

    /// Start advertising and browsing. Returns the OS-assigned TCP
    /// port once the listener reaches `.ready`. The returned port is
    /// what a joiner will connect to after resolving this node's
    /// Bonjour record. Throws if the listener fails to start.
    public func start() async throws -> NWEndpoint.Port {
        try startListener()
        startBrowser()
        return try await awaitReadyPort()
    }

    /// Stop advertising, cancel the browser, and terminate all
    /// outstanding `peers()` streams. Safe to call more than once.
    public func stop() {
        stateQueue.sync {
            listener?.cancel()
            listener = nil
            browser?.cancel()
            browser = nil
            for (_, cont) in peerContinuations {
                cont.finish()
            }
            peerContinuations.removeAll()
            seenPeerKeys.removeAll()
            if let cont = portContinuation {
                cont.resume(throwing: BonjourServiceError.notReady)
                portContinuation = nil
            }
        }
    }

    /// Async stream of peers matching the local cluster hash filter.
    /// Each subscription gets its own independent stream; a single
    /// BonjourService can support multiple observers. The stream emits
    /// each unique `(clusterHash, name)` pair at most once — re-joining
    /// the same cluster in a new BonjourService instance is how you
    /// reset dedup state.
    public func peers() -> AsyncStream<DiscoveredPeer> {
        let id = UUID()
        return AsyncStream { continuation in
            self.stateQueue.async { [weak self] in
                guard let self else { continuation.finish(); return }
                self.peerContinuations[id] = continuation
            }
            continuation.onTermination = { [weak self] _ in
                guard let self else { return }
                self.stateQueue.async {
                    self.peerContinuations.removeValue(forKey: id)
                }
            }
        }
    }

    // MARK: - Listener

    private func startListener() throws {
        let txtDict = advertisedInfo.toTxtRecord()
        var txtRecord = NWTXTRecord()
        for (k, v) in txtDict {
            txtRecord[k] = v
        }

        let parameters = NWParameters.tcp
        // allowLocalEndpointReuse keeps teardown/restart fast across
        // consecutive BonjourService instances on the same port (the
        // common case when ClusterManager updates the advertised info
        // and rebuilds the service).
        parameters.allowLocalEndpointReuse = true

        let listener: NWListener
        do {
            listener = try NWListener(using: parameters)
        } catch {
            throw BonjourServiceError.listenerFailed(error as? NWError ?? .posix(.EINVAL))
        }
        listener.service = NWListener.Service(
            name: advertisedServiceName,
            type: Self.serviceType,
            domain: nil,
            txtRecord: txtRecord
        )
        // The listener accepts inbound connections from joiners.
        // BonjourService itself does not handle them — ClusterManager
        // wires a separate coordinator listener. We still install a
        // no-op handler so the framework does not log a missing-handler
        // warning; accepted connections are immediately cancelled.
        listener.newConnectionHandler = { connection in
            connection.cancel()
        }
        // stateUpdateHandler runs on the queue passed to start(), which
        // is stateQueue — so direct state mutation here is already
        // serialized with respect to every other handler and the
        // queue-dispatched branches of start()/stop().
        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                guard let port = self.listener?.port,
                      let cont = self.portContinuation else { return }
                self.portContinuation = nil
                cont.resume(returning: port)
            case .failed(let err):
                if let cont = self.portContinuation {
                    self.portContinuation = nil
                    cont.resume(throwing: BonjourServiceError.listenerFailed(err))
                }
                // Fan the failure out to every peers() subscriber so
                // consumers know the advertise half is dead.
                for (_, pc) in self.peerContinuations {
                    pc.finish()
                }
            default:
                break
            }
        }

        stateQueue.sync {
            self.listener = listener
        }
        listener.start(queue: stateQueue)
    }

    private func awaitReadyPort() async throws -> NWEndpoint.Port {
        // If the listener is already ready by the time we await — rare
        // but possible under test — return the port directly.
        if let existing = stateQueue.sync(execute: { self.listener?.port }),
           stateQueue.sync(execute: { self.listener?.state }) == .ready {
            return existing
        }
        return try await withCheckedThrowingContinuation { continuation in
            stateQueue.async {
                // Race: listener may have reached .ready between the
                // initial check and this block. If so, resume now.
                if self.listener?.state == .ready, let port = self.listener?.port {
                    continuation.resume(returning: port)
                    return
                }
                self.portContinuation = continuation
            }
        }
    }

    // MARK: - Browser

    private func startBrowser() {
        let descriptor = NWBrowser.Descriptor.bonjourWithTXTRecord(
            type: Self.serviceType,
            domain: nil
        )
        let browser = NWBrowser(for: descriptor, using: .tcp)
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            guard let self else { return }
            self.handleBrowseResults(results)
        }
        browser.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            if case .failed = state {
                // Browser failure is not fatal to the listener —
                // advertising may still succeed — but subscribers
                // should know their stream will no longer receive new
                // peers. Close them with finish() so consumers can
                // observe the end of the stream.
                for (_, pc) in self.peerContinuations {
                    pc.finish()
                }
                self.peerContinuations.removeAll()
            }
        }

        stateQueue.sync {
            self.browser = browser
        }
        browser.start(queue: stateQueue)
    }

    private func handleBrowseResults(_ results: Set<NWBrowser.Result>) {
        // Called on stateQueue (the queue we started the browser with).
        for result in results {
            guard case .bonjour(let txt) = result.metadata else { continue }

            var dict: [String: String] = [:]
            for key in txt.dictionary.keys {
                if case .string(let value) = txt.getEntry(for: key) {
                    dict[key] = value
                }
            }
            guard let info = DiscoveryInfo(fromTxtRecord: dict) else { continue }

            // Skip our own advertisement. Matching on the advertised
            // service name is good enough — two nodes with the same
            // user-visible name colliding is rare and the worst case
            // is one self-filter miss.
            if case .service(let name, let type, _, _) = result.endpoint,
               name == advertisedServiceName, type == Self.serviceType {
                continue
            }

            // §3.4 cluster-hash filter. When the local node has no
            // cluster yet (localClusterHash == nil), pass everything
            // through so the user can see available clusters; once a
            // cluster has been joined, narrow to peers on the same
            // cluster.
            if let local = localClusterHash, info.clusterHash != local {
                continue
            }

            let key = "\(info.clusterHash ?? "none")|\(info.name)"
            if seenPeerKeys.contains(key) { continue }
            seenPeerKeys.insert(key)

            let peer = DiscoveredPeer(info: info, endpoint: result.endpoint)
            for (_, cont) in peerContinuations {
                cont.yield(peer)
            }
        }
    }
}
