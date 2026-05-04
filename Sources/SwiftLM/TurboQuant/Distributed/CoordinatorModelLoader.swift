// Coordinator-side `DistributedQwenModel` loader. Mirrors the joiner-
// side builder in `ClusterBringUp.swift` but with permissive failure
// semantics: a non-TurboQuant model directory is not an error on the
// coordinator (the chat-completions handler falls through to the
// existing single-node generation path when `coordinatorModel` is
// nil), so the loader signals that case via a typed throw the caller
// translates into a logged warning rather than a hard exit.
//
// Memory note: when the supplied directory IS TurboQuant-converted,
// constructing a coordinator `DistributedQwenModel` allocates the
// rank-local layer payloads plus the materialised embedding table
// (~600 MB at hidden=2048, vocab=151936). The coordinator process
// also holds the standard `ModelContainer` (used for tokenizer access
// during chat templating), so peak resident memory roughly doubles
// the model's compressed footprint while the distributed branch is
// active. A tokenizer-only loader on the MLXLLM side would close
// that gap; deferred so this layer ships against the unmodified
// upstream model loader.

import Foundation
import MLX

/// Failure cases surfaced by `loadCoordinatorDistributedModel`.
/// Coordinators that hit `missingShardMetadata` should fall through
/// to the existing single-node path; the other cases indicate the
/// supplied snapshot is broken in a way the operator must address.
public enum CoordinatorModelLoaderError: Error, CustomStringConvertible {
    /// The supplied directory does not contain `tq_shard_metadata.json`.
    /// The coordinator caller treats this as a soft failure and
    /// continues without a `DistributedQwenModel`.
    case missingShardMetadata(URL)

    /// The shard-metadata sidecar exists but failed to decode. Almost
    /// always indicates a partial download or a version mismatch
    /// between the convert tool and this build of SwiftLM.
    case shardMetadataDecodeFailed(URL, underlying: Error)

    /// `DistributedQwenModel` construction failed after the metadata
    /// loaded. Carries the wrapped error so the caller can surface
    /// the precise reason (worldSize divisibility, missing weight
    /// entry, embedding-table shape mismatch, etc.).
    case modelConstructionFailed(underlying: Error)

    public var description: String {
        switch self {
        case .missingShardMetadata(let url):
            return "no tq_shard_metadata.json sidecar found at \(url.path); not a TurboQuant-converted model directory"
        case .shardMetadataDecodeFailed(let url, let err):
            return "failed to decode tq_shard_metadata.json at \(url.path): \(err)"
        case .modelConstructionFailed(let err):
            return "DistributedQwenModel construction failed: \(err)"
        }
    }
}

/// Construct a coordinator-side `DistributedQwenModel` from the
/// TurboQuant-converted snapshot at `modelDir`. The model is built
/// against the supplied `DistributedGroup` (typically the singleton
/// `DistributedGroup()` at process start) and is suitable for
/// `ClusterManager.beginInferenceSession(model:request:)` once the
/// cluster has joiners attached.
///
/// The function is split out from `ClusterBringUp.swift` so tests can
/// drive the loader directly against synthetic directories without
/// going through the full create-cluster bring-up path. Production
/// callers use it via the default `coordinatorModelBuilder` injected
/// into `startCluster`.
///
/// - Parameters:
///   - modelDir: Path to the local snapshot. Must contain
///     `tq_shard_metadata.json` for the loader to succeed; otherwise
///     `missingShardMetadata` is thrown.
///   - group: Distributed group the model registers against.
///   - rank: Optional rank override; defaults to `group.rank`. Tests
///     drive multi-rank scenarios in-process via this override (see
///     the worldSize=2 structural tests).
///   - worldSize: Optional world-size override; defaults to
///     `group.size`.
public func loadCoordinatorDistributedModel(
    modelDir: URL,
    group: DistributedGroup,
    rank: Int? = nil,
    worldSize: Int? = nil
) throws -> DistributedQwenModel {
    let sidecarURL = modelDir.appendingPathComponent("tq_shard_metadata.json")
    guard FileManager.default.fileExists(atPath: sidecarURL.path) else {
        throw CoordinatorModelLoaderError.missingShardMetadata(modelDir)
    }
    let metadata: ShardMetadata
    do {
        let data = try Data(contentsOf: sidecarURL)
        metadata = try ShardMetadata(jsonData: data)
    } catch {
        throw CoordinatorModelLoaderError.shardMetadataDecodeFailed(sidecarURL, underlying: error)
    }
    do {
        return try DistributedQwenModel(
            metadata: metadata,
            modelDir: modelDir,
            group: group,
            rank: rank,
            worldSize: worldSize
        )
    } catch {
        throw CoordinatorModelLoaderError.modelConstructionFailed(underlying: error)
    }
}

/// Closure that produces a coordinator-side `DistributedQwenModel` from
/// the local snapshot directory, or returns `nil` when the directory
/// is not TurboQuant-converted (in which case the chat-completions
/// handler falls through to the single-node path). The default builder
/// uses `loadCoordinatorDistributedModel` and translates
/// `missingShardMetadata` into a `nil` return; tests substitute a
/// closure that returns a pre-built stub model so the bring-up flow
/// can be exercised without loading real weights.
public typealias CoordinatorModelBuilder =
    @Sendable (URL?) throws -> DistributedQwenModel?

/// Default coordinator model builder. Returns `nil` when the directory
/// is unresolved or not TurboQuant-converted — both are valid
/// configurations on the coordinator path. Other failures (corrupted
/// sidecar, model construction error) propagate so the operator sees
/// the precise diagnostic instead of a silent fallback.
public let defaultCoordinatorModelBuilder: CoordinatorModelBuilder = { modelDirectory in
    guard let modelDir = modelDirectory else { return nil }
    do {
        let group = DistributedGroup()
        return try loadCoordinatorDistributedModel(modelDir: modelDir, group: group)
    } catch CoordinatorModelLoaderError.missingShardMetadata {
        return nil
    }
}
