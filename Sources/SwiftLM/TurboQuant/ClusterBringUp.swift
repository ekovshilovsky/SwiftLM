// Bonjour-based cluster bring-up. Owns the orchestration that turns
// the parsed CLI flags into a live `ClusterManager`: constructing the
// underlying `BonjourService` and `FileClusterKeyStore`, running the
// coordinator's create flow or the joiner's discover-and-join flow,
// and (on the joiner) wrapping the rank-local model in a worker task
// so the inference loop is pumping the moment the handshake completes.
//
// Lives in TurboQuantKit (rather than the executable target) so the
// integration tests can drive the bring-up path directly without
// standing up the full HTTP server. The executable layer adds CLI
// glue, model loading, and lifetime management on top of what this
// module exposes.

import Foundation
import MLX
import MLXLMCommon

/// Bundle of cluster-formation outputs the server keeps alive for the
/// lifetime of the HTTP listener. The manager is the entry point for
/// joiner registration and request dispatch from the chat-completions
/// handler. The worker task is non-nil only on a joiner node — it
/// owns the inference loop that pumps control messages off the
/// coordinator channel.
///
/// The optional `coordinatorModel` is the rank-local
/// `DistributedQwenModel` the coordinator hands to
/// `ClusterManager.beginInferenceSession`. Production v1 leaves this
/// `nil`; the chat-completions handler treats a nil model as "not
/// equipped to drive distributed inference" and falls through to the
/// existing single-node generation path. Coordinator-side model
/// instantiation lands in a follow-up that integrates the model
/// loader with the distributed inference engine.
///
/// `@unchecked Sendable` because `DistributedQwenModel` is an `MLXNN`
/// module subclass and inherits non-Sendable storage from `Module`.
/// In production the model is constructed once on the coordinator and
/// only entered through the `ClusterManager` actor's session APIs, so
/// concurrent access to the underlying module is mediated by the
/// actor's executor. Tests that drive the bring-up directly do the
/// same.
public struct ClusterBringUp: @unchecked Sendable {
    public let manager: ClusterManager
    public let role: DistributedNodeRole
    public let workerTask: Task<Void, Error>?
    public let coordinatorModel: DistributedQwenModel?

    public init(
        manager: ClusterManager,
        role: DistributedNodeRole,
        workerTask: Task<Void, Error>?,
        coordinatorModel: DistributedQwenModel? = nil
    ) {
        self.manager = manager
        self.role = role
        self.workerTask = workerTask
        self.coordinatorModel = coordinatorModel
    }
}

/// Errors surfaced by the Bonjour bring-up path. Descriptions are
/// printed verbatim to stderr by the CLI, so they are phrased as
/// actionable user-facing messages rather than internal diagnostics.
public enum ClusterBringUpError: Error, CustomStringConvertible {
    /// `--auto` cannot decide a role yet because the keystore-driven
    /// auto-discovery path is not implemented. Caller must pass an
    /// explicit `--primary` or `--secondary`.
    case autoRoleNotImplemented
    /// `--distributed` was requested without a `--primary`/`--secondary`
    /// override and `--auto` was not used either, so the role is
    /// undefined.
    case missingRole
    /// The joiner discovery loop did not surface a coordinator peer
    /// within the configured timeout. Carries the timeout in seconds
    /// for the diagnostic message.
    case joinerDiscoveryTimedOut(Double)
    /// The joiner role requires a TurboQuant-converted model (so a
    /// rank-local `DistributedQwenModel` can be built around the
    /// sharded weights). The supplied `--model` did not satisfy that
    /// constraint.
    case joinerRequiresTurboQuantModel
    /// Could not resolve the model directory on disk. The TurboQuant
    /// detection path needs a local snapshot to inspect for the
    /// shard-metadata sidecar.
    case modelDirectoryUnresolved

    public var description: String {
        switch self {
        case .autoRoleNotImplemented:
            return "--auto role auto-detection is not yet implemented; pass --primary or --secondary explicitly"
        case .missingRole:
            return "--distributed requires --primary or --secondary (or --auto, once auto-detection lands)"
        case .joinerDiscoveryTimedOut(let seconds):
            return "joiner did not discover a coordinator within \(Int(seconds))s; verify the coordinator is running on the same network with the same SWIFTLM_CLUSTER_PASSPHRASE"
        case .joinerRequiresTurboQuantModel:
            return "--secondary requires a TurboQuant-converted model so the joiner can build a rank-local DistributedQwenModel; supply a TQ-converted model via --model"
        case .modelDirectoryUnresolved:
            return "could not resolve a local snapshot directory for the supplied --model; download the model fully before joining the cluster"
        }
    }
}

/// Optional joiner-side affordance. The production code path supplies
/// a builder that materialises a `DistributedQwenModel` from the local
/// snapshot directory; tests substitute a closure that returns a
/// pre-built stub model so the bring-up flow can be exercised without
/// loading real weights. Returning `nil` opts the joiner out of running
/// the worker loop — useful for the path where the coordinator is
/// reachable but this node is not equipped to participate in inference.
public typealias JoinerModelBuilder =
    @Sendable (URL?) throws -> DistributedQwenModel?

/// Joiner discovery timeout used by `startCluster` when the caller does
/// not override it. Bonjour resolution under a working mDNSResponder is
/// sub-second on a healthy LAN; 30 seconds is a generous upper bound
/// that catches a slow first-resolve while still failing fast on a
/// misconfigured network or wrong passphrase.
public let defaultClusterJoinerDiscoveryTimeoutSeconds: Double = 30

/// Default cluster-protocol version broadcast in the Bonjour TXT
/// record. Bumped whenever the handshake or data-channel wire format
/// changes.
public let defaultSwiftlmClusterVersion = "0.1.0"

/// Drive the create / join flow against a freshly-constructed
/// `ClusterManager` and return the bound runtime state. Extracted from
/// the CLI entry point so integration tests can exercise the
/// cluster-formation path without standing up the full HTTP server.
///
/// On the primary path the manager is left advertising and accepting
/// joiners; on the joiner path the manager has completed its handshake,
/// established a coordinator data channel, and (when the supplied
/// `joinerModelBuilder` returns a non-nil model) spawned a detached
/// task running `runJoinerWorker(model:)`.
///
/// The joiner path starts the BonjourService explicitly before the
/// discovery loop because `ClusterManager.joinCluster` does not start
/// its own service (the coordinator side does, inside `createCluster`).
/// Without this start the underlying NWBrowser never spins up and the
/// peers stream never yields, leaving the joiner blocked until the
/// configured timeout.
///
/// - Parameters:
///   - options: Validated CLI flags. Only `role` and `isAuto` drive
///     the bring-up; the rest are diagnostic.
///   - passphrase: Cluster passphrase. Production callers source this
///     from the `SWIFTLM_CLUSTER_PASSPHRASE` environment variable.
///   - modelId: Bonjour TXT `model` field; peers with mismatching
///     model IDs ignore each other.
///   - modelDirectory: Path to the local snapshot. Forwarded to the
///     joiner model builder; `nil` is acceptable on the primary path.
///   - keyStore: Persistent storage for the resulting cluster record.
///     Production callers pass `FileClusterKeyStore.default()`; tests
///     substitute an in-memory store.
///   - bonjourFactory: Closure that produces the underlying
///     `BonjourService`. Defaults to constructing one from the
///     supplied initial discovery info; tests override to inject a
///     mock advertise / browse pair.
///   - joinerModelBuilder: Closure that materialises the joiner-side
///     `DistributedQwenModel`. Defaults to the on-disk shard-metadata
///     loader; tests substitute a stub builder so the worker spawn
///     can be exercised without real weights.
///   - hostname: Override for the local hostname. Defaults to
///     `ProcessInfo.processInfo.hostName`.
///   - memoryGB: Override for the advertised memory budget. Defaults
///     to the host's `physicalMemory`.
///   - version: Override for the broadcast version. Defaults to the
///     module-level `defaultSwiftlmClusterVersion`.
///   - rdma: Override for the advertised RDMA capability.
///   - discoveryTimeoutSeconds: Override for the joiner discovery
///     timeout. Defaults to `defaultClusterJoinerDiscoveryTimeoutSeconds`.
public func startCluster(
    options: DistributedCLIOptions,
    passphrase: String,
    modelId: String,
    modelDirectory: URL?,
    keyStore: ClusterKeyStore,
    bonjourFactory: (DiscoveryInfo) -> BonjourService = { info in
        BonjourService(info: info, localClusterHash: nil)
    },
    joinerModelBuilder: JoinerModelBuilder = defaultJoinerModelBuilder,
    coordinatorModelBuilder: CoordinatorModelBuilder = defaultCoordinatorModelBuilder,
    hostname: String = ProcessInfo.processInfo.hostName,
    memoryGB: Int = Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)),
    version: String = defaultSwiftlmClusterVersion,
    rdma: RdmaCapability = .unsupported,
    discoveryTimeoutSeconds: Double = defaultClusterJoinerDiscoveryTimeoutSeconds
) async throws -> ClusterBringUp {
    let role = try resolveDistributedRole(options: options)

    // Discovery info advertised on the wire prior to cluster formation.
    // The coordinator overwrites the role + cluster fields inside
    // `createCluster`; the joiner overwrites them inside `joinCluster`
    // once the handshake produces a working key.
    let initialInfo = DiscoveryInfo(
        model: modelId,
        memoryGB: memoryGB,
        role: .discovering,
        clusterHash: nil,
        clusterId: nil,
        version: version,
        rdma: rdma,
        name: hostname
    )

    let bonjour = bonjourFactory(initialInfo)
    let manager = ClusterManager(
        keyStore: keyStore,
        bonjour: bonjour,
        localHostname: hostname,
        model: modelId,
        memoryGB: memoryGB,
        version: version,
        rdma: rdma
    )

    switch role {
    case .primary:
        // Coordinators do not need the model on the wire to start
        // accepting joiners — the model only matters when a session
        // begins. The createCluster accept loop fans inbound
        // connections out to per-handshake tasks and registers each
        // successful joiner channel on the manager.
        _ = try await manager.createCluster(
            passphrase: passphrase,
            clusterName: nil
        )

        // The coordinator-side `DistributedQwenModel` is only
        // constructed when the supplied snapshot is TurboQuant-
        // converted. Non-TQ snapshots return nil from the builder so
        // the coordinator continues to advertise as a cluster head
        // while the chat-completions handler falls through to the
        // existing single-node generation path. Holding both the
        // standard `ModelContainer` and a coordinator
        // `DistributedQwenModel` in one process roughly doubles the
        // resident model footprint while the distributed branch is
        // active; a tokenizer-only loader on the upstream model side
        // would close that gap.
        let coordinatorModel: DistributedQwenModel?
        do {
            coordinatorModel = try coordinatorModelBuilder(modelDirectory)
            if coordinatorModel == nil {
                FileHandle.standardError.write(Data((
                    "[SwiftLM] --distributed --primary supplied a snapshot " +
                    "without a tq_shard_metadata.json sidecar; the " +
                    "coordinator will accept joiners but chat completions " +
                    "will use the single-node generation path until a " +
                    "TurboQuant-converted snapshot is provided.\n"
                ).utf8))
            }
        } catch {
            await manager.stop()
            throw error
        }
        return ClusterBringUp(
            manager: manager,
            role: .primary,
            workerTask: nil,
            coordinatorModel: coordinatorModel
        )

    case .secondary:
        // Start the joiner's BonjourService so its NWBrowser is live
        // before the discovery loop begins. `ClusterManager.joinCluster`
        // does not start the service itself — only the coordinator
        // path needs the listener, and the joiner has historically
        // relied on whoever called it to bring browsing up first.
        do {
            _ = try await bonjour.start()
        } catch {
            await manager.stop()
            throw error
        }
        let peer = try await discoverCoordinatorPeer(
            via: manager,
            timeoutSeconds: discoveryTimeoutSeconds
        )
        _ = try await manager.joinCluster(passphrase: passphrase, peer: peer)

        // The joiner participates in inference only when a rank-local
        // DistributedQwenModel can be built from the local snapshot.
        // The builder closure returns nil to opt this node out of the
        // worker loop (e.g. when the snapshot is not TQ-converted but
        // the operator still wants the joiner to expose its health
        // endpoints to the cluster).
        let model = try joinerModelBuilder(modelDirectory)
        guard let resolvedModel = model else {
            return ClusterBringUp(manager: manager, role: .secondary, workerTask: nil)
        }
        let workerTask = Task { [manager] in
            try await manager.runJoinerWorker(model: resolvedModel)
        }
        return ClusterBringUp(manager: manager, role: .secondary, workerTask: workerTask)
    }
}

/// Run the cluster shutdown sequence: cancel the joiner worker task
/// (if any) and tear down the manager so the underlying Bonjour
/// listener and any joiner connections close cleanly. Safe to call
/// with a `nil` bring-up; idempotent against a manager that has
/// already been stopped (`ClusterManager.stop()` short-circuits when
/// it has been invoked previously).
public func tearDownCluster(_ bringUp: ClusterBringUp?) async {
    guard let bringUp else { return }
    bringUp.workerTask?.cancel()
    await bringUp.manager.stop()
}

/// Resolve the role flag set into a concrete `DistributedNodeRole`.
/// `--auto` is reserved for the keystore-driven discovery path, which
/// is not implemented here yet — callers must pass an explicit role
/// override until that work lands.
private func resolveDistributedRole(
    options: DistributedCLIOptions
) throws -> DistributedNodeRole {
    if let explicit = options.role {
        return explicit
    }
    if options.isAuto {
        // Sketch of the future implementation: load a record from the
        // configured key store; if present, attempt to join the same
        // cluster (secondary); if absent, create a new one (primary).
        // Deferred so this layer lands without conflating bring-up
        // wiring with the auto-mode keystore semantics.
        throw ClusterBringUpError.autoRoleNotImplemented
    }
    throw ClusterBringUpError.missingRole
}

/// Wait for the first peer surfaced by the manager's Bonjour browser
/// whose advertised role is `.coordinator`. The cluster-hash filter
/// inside `BonjourService.peers()` already narrows the stream to the
/// matching cluster (or to every cluster, before the joiner has its
/// own working key), so the role check here is the only post-filter
/// the bring-up path needs. Times out cleanly after the supplied
/// budget elapses.
private func discoverCoordinatorPeer(
    via manager: ClusterManager,
    timeoutSeconds: Double
) async throws -> DiscoveredPeer {
    let stream = manager.listPeers()
    return try await withThrowingTaskGroup(of: DiscoveredPeer?.self) { group in
        group.addTask {
            for await peer in stream where peer.info.role == .coordinator {
                return peer
            }
            return nil
        }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(timeoutSeconds * 1_000_000_000))
            return nil
        }
        let result = try await group.next() ?? nil
        group.cancelAll()
        guard let peer = result else {
            throw ClusterBringUpError.joinerDiscoveryTimedOut(timeoutSeconds)
        }
        return peer
    }
}

/// Default joiner model builder. Loads the `tq_shard_metadata.json`
/// sidecar from the supplied snapshot and constructs a rank-local
/// `DistributedQwenModel` over the singleton distributed group. Non-TQ
/// snapshots throw `joinerRequiresTurboQuantModel` so the operator
/// gets a precise diagnostic instead of an opaque metadata-decode
/// failure.
public let defaultJoinerModelBuilder: JoinerModelBuilder = { modelDirectory in
    guard let modelDir = modelDirectory else {
        throw ClusterBringUpError.modelDirectoryUnresolved
    }
    let sidecarURL = modelDir.appendingPathComponent("tq_shard_metadata.json")
    guard FileManager.default.fileExists(atPath: sidecarURL.path) else {
        throw ClusterBringUpError.joinerRequiresTurboQuantModel
    }
    do {
        let data = try Data(contentsOf: sidecarURL)
        let metadata = try ShardMetadata(jsonData: data)
        let group = DistributedGroup()
        return try DistributedQwenModel(
            metadata: metadata,
            modelDir: modelDir,
            group: group
        )
    } catch let err as ClusterBringUpError {
        throw err
    } catch {
        // Any sidecar parse or shard-loading failure surfaces as the
        // same actionable message — the joiner cannot proceed without
        // a well-formed TQ snapshot.
        throw ClusterBringUpError.joinerRequiresTurboQuantModel
    }
}
