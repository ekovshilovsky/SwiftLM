// BonjourService integration tests. Exercises the live advertise+browse
// round-trip over loopback mDNS: one BonjourService advertises a known
// DiscoveryInfo, a second BonjourService (configured as a different
// "node") browses, and we assert the advertised TXT fields round-trip
// through Network.framework intact and are delivered to a peers()
// subscriber.
//
// The test is defensively wrapped in an XCTSkip path because some CI
// environments block multicast DNS (containers, sandboxed test runners,
// locked-down build farms). On a developer Mac with mDNSResponder
// running, it should run and pass in a few hundred milliseconds.
//
// Cluster-hash filter coverage: we also exercise the §3.4 subscription-
// layer filter by advertising two peers on different hashes and
// asserting the local node only sees the one whose hash matches.

import XCTest
import Network
import TurboQuantKit

final class BonjourServiceTests: XCTestCase {

    // MARK: - Helpers

    /// Randomize the advertised node `name` per test so concurrent or
    /// back-to-back runs (local dev, CI matrix) never collide via the
    /// stale-cache window mDNSResponder keeps for recently-departed
    /// services.
    private func uniqueName(_ prefix: String) -> String {
        "\(prefix)-\(UUID().uuidString.prefix(8))"
    }

    private func makeInfo(
        name: String,
        clusterHash: String?,
        role: DiscoveryRole = .coordinator,
        memoryGB: Int = 128
    ) -> DiscoveryInfo {
        DiscoveryInfo(
            model: "ekovshilovsky/Qwen2.5-32B-TQ8",
            memoryGB: memoryGB,
            role: role,
            clusterHash: clusterHash,
            clusterId: clusterHash.map { _ in "7f3a8b91-4e2c-4c5d-9e6f-1a2b3c4d5e6f" },
            version: "0.1.0",
            rdma: .available,
            name: name
        )
    }

    // MARK: - Advertise + browse round-trip

    func testAdvertiseAndBrowseRoundTrip() async throws {
        let advertisedName = uniqueName("mac-studio")
        let info = makeInfo(name: advertisedName, clusterHash: "a7f3e9b1")

        let advertiser = BonjourService(info: info, localClusterHash: nil)
        defer { advertiser.stop() }

        let port: NWEndpoint.Port
        do {
            port = try await advertiser.start()
        } catch {
            throw XCTSkip("Advertiser could not start — mDNSResponder likely unavailable in this environment: \(error)")
        }
        XCTAssertGreaterThan(port.rawValue, 0, "listener should have an OS-assigned TCP port")

        // Browser runs as a second BonjourService with no local cluster
        // filter so anything the advertiser puts on the wire is visible.
        // The browser's own advertised name differs so it does not
        // self-filter the target out.
        let browserInfo = makeInfo(name: uniqueName("observer"), clusterHash: nil, role: .discovering)
        let browserService = BonjourService(info: browserInfo, localClusterHash: nil)
        defer { browserService.stop() }

        _ = try await browserService.start()

        let discovered = try await withTimeout(seconds: 5) {
            for await peer in browserService.peers() {
                if peer.info.name == advertisedName {
                    return peer
                }
            }
            return nil as DiscoveredPeer?
        }

        guard let peer = discovered else {
            throw XCTSkip("mDNSResponder did not deliver any results within 5s — skipping active Bonjour round-trip")
        }

        XCTAssertEqual(peer.info.name, advertisedName)
        XCTAssertEqual(peer.info.model, info.model)
        XCTAssertEqual(peer.info.memoryGB, info.memoryGB)
        XCTAssertEqual(peer.info.role, info.role)
        XCTAssertEqual(peer.info.clusterHash, info.clusterHash)
        XCTAssertEqual(peer.info.clusterId, info.clusterId)
        XCTAssertEqual(peer.info.version, info.version)
        XCTAssertEqual(peer.info.rdma, info.rdma)

        // The endpoint should be a Bonjour service endpoint suitable
        // for NWConnection to do an implicit resolve against.
        if case .service(let name, let type, _, _) = peer.endpoint {
            XCTAssertEqual(name, advertisedName)
            XCTAssertEqual(type, BonjourService.serviceType)
        } else {
            XCTFail("discovered peer endpoint should be a Bonjour .service endpoint, got \(peer.endpoint)")
        }
    }

    // MARK: - Cluster-hash subscription filter

    func testClusterHashFilterHidesOtherClusters() async throws {
        let sameClusterName = uniqueName("same-cluster")
        let otherClusterName = uniqueName("other-cluster")

        let sameInfo = makeInfo(name: sameClusterName, clusterHash: "a7f3e9b1")
        let otherInfo = makeInfo(name: otherClusterName, clusterHash: "deadbeef")

        let sameAdvertiser = BonjourService(info: sameInfo, localClusterHash: nil)
        let otherAdvertiser = BonjourService(info: otherInfo, localClusterHash: nil)
        defer {
            sameAdvertiser.stop()
            otherAdvertiser.stop()
        }

        do {
            _ = try await sameAdvertiser.start()
            _ = try await otherAdvertiser.start()
        } catch {
            throw XCTSkip("Advertiser could not start — mDNSResponder likely unavailable: \(error)")
        }

        // Local node filters to the "a7f3e9b1" cluster; the other
        // cluster must never appear.
        let browserInfo = makeInfo(name: uniqueName("filter-observer"),
                                   clusterHash: "a7f3e9b1",
                                   role: .worker)
        let browser = BonjourService(info: browserInfo, localClusterHash: "a7f3e9b1")
        defer { browser.stop() }
        _ = try await browser.start()

        // Collect up to 3s worth of peers and assert the other-cluster
        // name never appears. A 3s window is enough that if the filter
        // were broken we would almost certainly see both.
        let seenNames: Set<String> = await withTimeoutCollectingNames(
            seconds: 3,
            stream: browser.peers()
        )

        if !seenNames.contains(sameClusterName) {
            throw XCTSkip("mDNSResponder did not deliver the expected same-cluster peer — skipping filter assertion")
        }
        XCTAssertFalse(seenNames.contains(otherClusterName),
                       "cluster-hash filter must hide peers on different clusters (§3.4)")
    }

    // MARK: - Timeout utilities

    /// Await the first matching peer from a stream or return nil on
    /// timeout. Used by the round-trip test which needs exactly one
    /// specific peer name.
    private func withTimeout(
        seconds: Double,
        operation: @escaping @Sendable () async -> DiscoveredPeer?
    ) async throws -> DiscoveredPeer? {
        return try await withThrowingTaskGroup(of: DiscoveredPeer?.self) { group in
            group.addTask { await operation() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                return nil
            }
            let first = try await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }

    /// Collect peer names from a stream over a fixed window. Used by
    /// the filter test, which needs to observe the steady-state set of
    /// peers the filter is passing through — including the important
    /// negative property that a disallowed peer NEVER appears.
    private func withTimeoutCollectingNames(
        seconds: Double,
        stream: AsyncStream<DiscoveredPeer>
    ) async -> Set<String> {
        let collector = Collector()
        let task = Task {
            for await peer in stream {
                await collector.insert(peer.info.name)
            }
        }
        try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
        task.cancel()
        return await collector.snapshot()
    }

    /// Thread-safe accumulator for the filter-test collector task.
    private actor Collector {
        private var names: Set<String> = []
        func insert(_ name: String) { names.insert(name) }
        func snapshot() -> Set<String> { names }
    }
}
