// TopologyReporter formatting tests. Each expected string is written
// out by hand based on the specified output contract — no test body
// captures the output of `format(_:)` and asserts against it, since
// that would only prove the function is deterministic, not correct.
//
// Covers:
//   - Minimal two-node happy path with direct link and RDMA available.
//   - Three-node partial mesh with one missing cable surfacing a
//     single ACTION NEEDED block.
//   - Disabled-RDMA node emitting the recovery-mode instructions.
//   - Clean-state cluster (no missing links, no disabled RDMA)
//     producing no trailing action/RDMA sections.
//   - Layer-range display honoring the inclusive/exclusive convention
//     with multi-digit layer indices.

import XCTest
import TurboQuantKit

final class TopologyReporterTests: XCTestCase {

    // MARK: - Two-node happy path

    func testTwoNodesFullyConnected() {
        let topology = TopologyReporter.ClusterTopology(
            nodes: [
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-1",
                    memoryGB: 128,
                    rank: 0,
                    layerStart: 0,
                    layerEnd: 16,
                    rdma: .available
                ),
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-2",
                    memoryGB: 128,
                    rank: 1,
                    layerStart: 16,
                    layerEnd: 32,
                    rdma: .available
                ),
            ],
            links: [
                TopologyReporter.LinkInfo(
                    nodeA: "mac-studio-1",
                    nodeB: "mac-studio-2",
                    latencyUs: 30,
                    isDirect: true,
                    rdmaStatus: .available
                ),
            ],
            strategy: "layer-split",
            topologyType: .fullyConnected
        )

        let expected = """
        [TurboQuant Cluster]
          Topology: 2 nodes, fully_connected
          Strategy: layer-split

          Connectivity:
            mac-studio-1 <-> mac-studio-2   available 30us

          Layer assignment:
            mac-studio-1 (128GB): layers 0-15
            mac-studio-2 (128GB): layers 16-31
        """

        XCTAssertEqual(TopologyReporter.format(topology), expected)
    }

    // MARK: - Missing Thunderbolt cable

    func testThreeNodesWithMissingLinkSurfacesSingleAction() {
        let topology = TopologyReporter.ClusterTopology(
            nodes: [
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-1",
                    memoryGB: 128,
                    rank: 0,
                    layerStart: 0,
                    layerEnd: 10,
                    rdma: .available
                ),
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-2",
                    memoryGB: 128,
                    rank: 1,
                    layerStart: 10,
                    layerEnd: 20,
                    rdma: .available
                ),
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-3",
                    memoryGB: 128,
                    rank: 2,
                    layerStart: 20,
                    layerEnd: 30,
                    rdma: .available
                ),
            ],
            links: [
                TopologyReporter.LinkInfo(
                    nodeA: "mac-studio-1",
                    nodeB: "mac-studio-2",
                    latencyUs: 28,
                    isDirect: true,
                    rdmaStatus: .available
                ),
                TopologyReporter.LinkInfo(
                    nodeA: "mac-studio-2",
                    nodeB: "mac-studio-3",
                    latencyUs: 32,
                    isDirect: true,
                    rdmaStatus: .available
                ),
                TopologyReporter.LinkInfo(
                    nodeA: "mac-studio-1",
                    nodeB: "mac-studio-3",
                    latencyUs: nil,
                    isDirect: false,
                    rdmaStatus: .unsupported
                ),
            ],
            strategy: "layer-split",
            topologyType: .partialMesh
        )

        let expected = """
        [TurboQuant Cluster]
          Topology: 3 nodes, partial_mesh
          Strategy: layer-split

          Connectivity:
            mac-studio-1 <-> mac-studio-2   available 28us
            mac-studio-2 <-> mac-studio-3   available 32us
            mac-studio-1 <-> mac-studio-3   No direct link

          Layer assignment:
            mac-studio-1 (128GB): layers 0-9
            mac-studio-2 (128GB): layers 10-19
            mac-studio-3 (128GB): layers 20-29

          ACTION NEEDED: Connect Thunderbolt cable mac-studio-1 <-> mac-studio-3
        """

        XCTAssertEqual(TopologyReporter.format(topology), expected)
    }

    // MARK: - Disabled RDMA action block

    func testDisabledRdmaProducesRecoveryInstructionsForThatNodeOnly() {
        let topology = TopologyReporter.ClusterTopology(
            nodes: [
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-1",
                    memoryGB: 128,
                    rank: 0,
                    layerStart: 0,
                    layerEnd: 16,
                    rdma: .available
                ),
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-2",
                    memoryGB: 128,
                    rank: 1,
                    layerStart: 16,
                    layerEnd: 32,
                    rdma: .disabled
                ),
            ],
            links: [
                TopologyReporter.LinkInfo(
                    nodeA: "mac-studio-1",
                    nodeB: "mac-studio-2",
                    latencyUs: 45,
                    isDirect: true,
                    rdmaStatus: .disabled
                ),
            ],
            strategy: "layer-split",
            topologyType: .fullyConnected
        )

        let expected = """
        [TurboQuant Cluster]
          Topology: 2 nodes, fully_connected
          Strategy: layer-split

          Connectivity:
            mac-studio-1 <-> mac-studio-2   disabled 45us

          Layer assignment:
            mac-studio-1 (128GB): layers 0-15
            mac-studio-2 (128GB): layers 16-31

          RDMA NOT ENABLED: mac-studio-2
            Boot into macOS Recovery -> Utilities -> Terminal -> rdma_ctl enable
        """

        XCTAssertEqual(TopologyReporter.format(topology), expected)
    }

    // MARK: - Unsupported RDMA is not actionable

    func testUnsupportedRdmaDoesNotProduceActionBlock() {
        let topology = TopologyReporter.ClusterTopology(
            nodes: [
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-1",
                    memoryGB: 128,
                    rank: 0,
                    layerStart: 0,
                    layerEnd: 16,
                    rdma: .unsupported
                ),
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-2",
                    memoryGB: 128,
                    rank: 1,
                    layerStart: 16,
                    layerEnd: 32,
                    rdma: .unsupported
                ),
            ],
            links: [
                TopologyReporter.LinkInfo(
                    nodeA: "mac-studio-1",
                    nodeB: "mac-studio-2",
                    latencyUs: 90,
                    isDirect: true,
                    rdmaStatus: .unsupported
                ),
            ],
            strategy: "layer-split",
            topologyType: .fullyConnected
        )

        let expected = """
        [TurboQuant Cluster]
          Topology: 2 nodes, fully_connected
          Strategy: layer-split

          Connectivity:
            mac-studio-1 <-> mac-studio-2   unsupported 90us

          Layer assignment:
            mac-studio-1 (128GB): layers 0-15
            mac-studio-2 (128GB): layers 16-31
        """

        XCTAssertEqual(TopologyReporter.format(topology), expected)
    }

    // MARK: - Multi-digit layer ranges

    func testMultiDigitLayerRangesFormatCorrectly() {
        let topology = TopologyReporter.ClusterTopology(
            nodes: [
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-1",
                    memoryGB: 256,
                    rank: 0,
                    layerStart: 0,
                    layerEnd: 100,
                    rdma: .available
                ),
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-2",
                    memoryGB: 256,
                    rank: 1,
                    layerStart: 100,
                    layerEnd: 200,
                    rdma: .available
                ),
            ],
            links: [
                TopologyReporter.LinkInfo(
                    nodeA: "mac-studio-1",
                    nodeB: "mac-studio-2",
                    latencyUs: 25,
                    isDirect: true,
                    rdmaStatus: .available
                ),
            ],
            strategy: "layer-split",
            topologyType: .fullyConnected
        )

        let expected = """
        [TurboQuant Cluster]
          Topology: 2 nodes, fully_connected
          Strategy: layer-split

          Connectivity:
            mac-studio-1 <-> mac-studio-2   available 25us

          Layer assignment:
            mac-studio-1 (256GB): layers 0-99
            mac-studio-2 (256GB): layers 100-199
        """

        XCTAssertEqual(TopologyReporter.format(topology), expected)
    }

    // MARK: - Combined: missing cable AND disabled RDMA

    func testMissingCableAndDisabledRdmaBothSurfaceInOrder() {
        let topology = TopologyReporter.ClusterTopology(
            nodes: [
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-1",
                    memoryGB: 128,
                    rank: 0,
                    layerStart: 0,
                    layerEnd: 10,
                    rdma: .available
                ),
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-2",
                    memoryGB: 128,
                    rank: 1,
                    layerStart: 10,
                    layerEnd: 20,
                    rdma: .disabled
                ),
                TopologyReporter.NodeSummary(
                    hostname: "mac-studio-3",
                    memoryGB: 128,
                    rank: 2,
                    layerStart: 20,
                    layerEnd: 30,
                    rdma: .available
                ),
            ],
            links: [
                TopologyReporter.LinkInfo(
                    nodeA: "mac-studio-1",
                    nodeB: "mac-studio-2",
                    latencyUs: 40,
                    isDirect: true,
                    rdmaStatus: .disabled
                ),
                TopologyReporter.LinkInfo(
                    nodeA: "mac-studio-2",
                    nodeB: "mac-studio-3",
                    latencyUs: 40,
                    isDirect: true,
                    rdmaStatus: .disabled
                ),
                TopologyReporter.LinkInfo(
                    nodeA: "mac-studio-1",
                    nodeB: "mac-studio-3",
                    latencyUs: nil,
                    isDirect: false,
                    rdmaStatus: .unsupported
                ),
            ],
            strategy: "layer-split",
            topologyType: .partialMesh
        )

        let expected = """
        [TurboQuant Cluster]
          Topology: 3 nodes, partial_mesh
          Strategy: layer-split

          Connectivity:
            mac-studio-1 <-> mac-studio-2   disabled 40us
            mac-studio-2 <-> mac-studio-3   disabled 40us
            mac-studio-1 <-> mac-studio-3   No direct link

          Layer assignment:
            mac-studio-1 (128GB): layers 0-9
            mac-studio-2 (128GB): layers 10-19
            mac-studio-3 (128GB): layers 20-29

          ACTION NEEDED: Connect Thunderbolt cable mac-studio-1 <-> mac-studio-3

          RDMA NOT ENABLED: mac-studio-2
            Boot into macOS Recovery -> Utilities -> Terminal -> rdma_ctl enable
        """

        XCTAssertEqual(TopologyReporter.format(topology), expected)
    }
}
