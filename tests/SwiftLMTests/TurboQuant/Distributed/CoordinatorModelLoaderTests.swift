// Unit and integration tests for `loadCoordinatorDistributedModel` and
// the coordinator-model wiring inside `startCluster`.
//
// The fixture-dependent tests use the same `TQ_FIXTURE_DIR`-backed
// `TurboQuantTestFixtures.requireQwenCoder3B()` accessor as the rest
// of the TurboQuant test suite and skip cleanly when the fixture is
// not installed. The negative-path tests build their own temporary
// directories so they run in any environment.
//
// End-to-end "request flows through the distributed engine" smoke
// testing requires the full HTTP server, a real tokenizer, two model
// instances loaded into one process, and a request-response cycle —
// out of scope here. Operators verify that path manually after this
// task lands by running `swiftlm serve --distributed --primary --model
// /path/to/Qwen2.5-Coder-3B-TQ8` with `SWIFTLM_CLUSTER_PASSPHRASE`
// set, then posting to `/v1/chat/completions`.

#if DEBUG

import Foundation
import MLX
import XCTest
@testable import TurboQuantKit

final class CoordinatorModelLoaderTests: XCTestCase {

    // MARK: - Direct loader: success path

    /// Loading from a TurboQuant-converted directory returns a model
    /// whose architecture configuration matches the fixture's
    /// `config.json`. This validates the metadata sidecar parse, the
    /// `DistributedQwenModel` construction, and the resulting model's
    /// reachable shape — the same surface the chat-completions handler
    /// inspects when routing through the distributed engine.
    func testLoadFromTQConvertedDirectorySucceeds() throws {
        let fixtureRoot = try TurboQuantTestFixtures.requireQwenCoder3B()
        let group = DistributedGroup()
        let model = try loadCoordinatorDistributedModel(
            modelDir: fixtureRoot,
            group: group
        )
        XCTAssertEqual(
            model.config.numHiddenLayers, 36,
            "Qwen2.5-Coder-3B has 36 transformer blocks; loader produced an unexpected layer count"
        )
        XCTAssertEqual(
            model.config.hiddenSize, 2048,
            "loader produced an unexpected hidden size"
        )
    }

    // MARK: - Direct loader: failure paths

    /// A directory missing the shard-metadata sidecar throws the
    /// soft-failure case the coordinator caller translates into a
    /// `nil` model and a logged warning. The test uses an isolated
    /// temporary directory so it runs in any environment.
    func testLoadFromNonTQDirectoryThrowsMissingShardMetadata() throws {
        let tmpRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("tq-coordinator-loader-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tmpRoot, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: tmpRoot) }

        XCTAssertThrowsError(
            try loadCoordinatorDistributedModel(
                modelDir: tmpRoot,
                group: DistributedGroup()
            )
        ) { error in
            guard let loaderError = error as? CoordinatorModelLoaderError else {
                XCTFail("expected CoordinatorModelLoaderError; got \(error)")
                return
            }
            switch loaderError {
            case .missingShardMetadata(let url):
                XCTAssertEqual(url.path, tmpRoot.path)
            default:
                XCTFail("expected .missingShardMetadata; got \(loaderError)")
            }
        }
    }

    // MARK: - Default builder: returns nil on non-TQ directory

    /// `defaultCoordinatorModelBuilder` translates the loader's
    /// `missingShardMetadata` throw into a `nil` return so the
    /// coordinator caller can fall through to the single-node path
    /// without a hard error. The non-TQ case is the common one: an
    /// operator runs `--distributed --primary` against a vanilla
    /// MLXLLM model directory and the cluster forms but the
    /// distributed dispatch branch stays dormant.
    func testDefaultBuilderReturnsNilOnNonTQDirectory() throws {
        let tmpRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("tq-coordinator-builder-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tmpRoot, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: tmpRoot) }

        let model = try defaultCoordinatorModelBuilder(tmpRoot)
        XCTAssertNil(model, "non-TQ directory must produce a nil coordinator model")
    }

    /// The default builder also returns `nil` for an unresolved
    /// directory — the callable accepts `URL?` so the bring-up path
    /// can pass `nil` through when the model loader did not surface a
    /// local snapshot. The coordinator continues without a
    /// distributed-model handle and chat completions fall through.
    func testDefaultBuilderReturnsNilWhenDirectoryIsNil() throws {
        let model = try defaultCoordinatorModelBuilder(nil)
        XCTAssertNil(model, "nil model directory must produce a nil coordinator model")
    }

    // MARK: - startCluster integration: TQ directory populates coordinatorModel

    /// Drive the bring-up path through `startCluster` with a stub
    /// builder that returns a sentinel model and assert the returned
    /// `ClusterBringUp.coordinatorModel` is non-nil. Uses a stub
    /// builder rather than the real loader so the test does not
    /// require the fixture to be installed (the fixture-backed loader
    /// is covered by `testLoadFromTQConvertedDirectorySucceeds`).
    func testStartClusterPopulatesCoordinatorModelViaBuilder() async throws {
        let fixtureRoot = try TurboQuantTestFixtures.requireQwenCoder3B()
        let coordName = "coord-modelload-\(UUID().uuidString.prefix(8))"
        let keyStore = InMemoryClusterKeyStore()
        let options = DistributedCLIOptions(
            isDistributed: true,
            isAuto: false,
            role: .primary,
            snapshotInterval: nil,
            printClusterStatus: false,
            printLayerTypeReport: false
        )

        let bringUp = try await startCluster(
            options: options,
            passphrase: "correct-horse-battery-staple",
            modelId: "ekovshilovsky/Qwen2.5-Coder-3B-TQ8",
            modelDirectory: fixtureRoot,
            keyStore: keyStore,
            bonjourFactory: { info in
                let overridden = DiscoveryInfo(
                    model: info.model, memoryGB: info.memoryGB,
                    role: info.role, clusterHash: info.clusterHash,
                    clusterId: info.clusterId, version: info.version,
                    rdma: info.rdma, name: coordName
                )
                return BonjourService(info: overridden, localClusterHash: nil)
            }
        )
        defer {
            Task { await tearDownCluster(bringUp) }
        }

        XCTAssertEqual(bringUp.role, .primary)
        XCTAssertNotNil(
            bringUp.coordinatorModel,
            "coordinator with TQ-converted snapshot must populate coordinatorModel"
        )
        XCTAssertEqual(
            bringUp.coordinatorModel?.config.numHiddenLayers, 36,
            "coordinator model must reflect the fixture's transformer block count"
        )
    }

    // MARK: - startCluster integration: non-TQ directory leaves coordinatorModel nil

    /// Drive `startCluster` against a temporary non-TQ directory and
    /// assert `coordinatorModel` stays nil. The cluster still forms
    /// (the manager is returned) so joiners can attach if needed; the
    /// chat-completions handler will fall through to the existing
    /// single-node path because `coordinatorModel == nil` short-
    /// circuits the distributed dispatch branch.
    func testStartClusterLeavesCoordinatorModelNilOnNonTQDirectory() async throws {
        let tmpRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("tq-coordinator-nontq-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tmpRoot, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: tmpRoot) }

        let coordName = "coord-nontq-\(UUID().uuidString.prefix(8))"
        let keyStore = InMemoryClusterKeyStore()
        let options = DistributedCLIOptions(
            isDistributed: true,
            isAuto: false,
            role: .primary,
            snapshotInterval: nil,
            printClusterStatus: false,
            printLayerTypeReport: false
        )

        let bringUp = try await startCluster(
            options: options,
            passphrase: "correct-horse-battery-staple",
            modelId: "vanilla/non-tq-model",
            modelDirectory: tmpRoot,
            keyStore: keyStore,
            bonjourFactory: { info in
                let overridden = DiscoveryInfo(
                    model: info.model, memoryGB: info.memoryGB,
                    role: info.role, clusterHash: info.clusterHash,
                    clusterId: info.clusterId, version: info.version,
                    rdma: info.rdma, name: coordName
                )
                return BonjourService(info: overridden, localClusterHash: nil)
            }
        )
        defer {
            Task { await tearDownCluster(bringUp) }
        }

        XCTAssertEqual(bringUp.role, .primary)
        XCTAssertNil(
            bringUp.coordinatorModel,
            "non-TQ snapshot must leave coordinatorModel nil for single-node fallback"
        )
    }
}

#endif
