import XCTest
import TurboQuantKit

final class ClusterPlanBuilderTests: XCTestCase {

    func testInitWithValidDimsSucceeds() {
        // Realistic model dimensions (Qwen3-class) must yield a usable builder.
        let builder = ClusterPlanBuilder(numLayers: 64, numHeads: 28, headDim: 128)
        XCTAssertNotNil(builder)
    }

    func testInitWithInvalidDimsReturnsNil() {
        // Every dimension is independently validated by the C layer so the
        // builder never comes back holding a planner that cannot produce a
        // meaningful assignment.
        XCTAssertNil(ClusterPlanBuilder(numLayers: 0, numHeads: 28, headDim: 128))
        XCTAssertNil(ClusterPlanBuilder(numLayers: -1, numHeads: 28, headDim: 128))
        XCTAssertNil(ClusterPlanBuilder(numLayers: 64, numHeads: 0, headDim: 128))
        XCTAssertNil(ClusterPlanBuilder(numLayers: 64, numHeads: 28, headDim: 0))
    }

    func testAddNodeIncrementsNodeCount() {
        guard let builder = ClusterPlanBuilder(numLayers: 64, numHeads: 28, headDim: 128) else {
            XCTFail("Builder init failed with valid dims")
            return
        }
        XCTAssertEqual(builder.nodeCount, 0)
        XCTAssertTrue(builder.addNode(hostname: "host-a.local", memoryGB: 128.0))
        XCTAssertEqual(builder.nodeCount, 1)
        XCTAssertTrue(builder.addNode(hostname: "host-b.local", memoryGB: 64.0))
        XCTAssertEqual(builder.nodeCount, 2)
    }

    func testAddNodeRejectsInvalidInputs() {
        guard let builder = ClusterPlanBuilder(numLayers: 64, numHeads: 28, headDim: 128) else {
            XCTFail("Builder init failed with valid dims")
            return
        }
        XCTAssertFalse(builder.addNode(hostname: "host.local", memoryGB: 0.0))
        XCTAssertFalse(builder.addNode(hostname: "host.local", memoryGB: -16.0))
        XCTAssertEqual(builder.nodeCount, 0)
    }

    func testLayerRangeBeforePlanReturnsNil() {
        guard let builder = ClusterPlanBuilder(numLayers: 64, numHeads: 28, headDim: 128) else {
            XCTFail("Builder init failed with valid dims")
            return
        }
        XCTAssertTrue(builder.addNode(hostname: "host-a.local", memoryGB: 128.0))
        XCTAssertNil(builder.layerRange(forRank: 0))
    }

    func testPlanContiguousCoverageAndProportionality() {
        guard let builder = ClusterPlanBuilder(numLayers: 64, numHeads: 28, headDim: 128) else {
            XCTFail("Builder init failed with valid dims")
            return
        }
        XCTAssertTrue(builder.addNode(hostname: "host-a.local", memoryGB: 128.0))
        XCTAssertTrue(builder.addNode(hostname: "host-b.local", memoryGB: 64.0))
        XCTAssertTrue(builder.addNode(hostname: "host-c.local", memoryGB: 32.0))

        XCTAssertTrue(builder.plan())

        guard let r0 = builder.layerRange(forRank: 0),
              let r1 = builder.layerRange(forRank: 1),
              let r2 = builder.layerRange(forRank: 2) else {
            XCTFail("Expected three layer ranges after plan()")
            return
        }

        // Contiguous coverage: the plan must span all 64 layers with
        // adjacent ranges touching end-to-start and no gap at either edge.
        XCTAssertEqual(r0.lowerBound, 0)
        XCTAssertEqual(r2.upperBound, 64)
        XCTAssertEqual(r0.upperBound, r1.lowerBound)
        XCTAssertEqual(r1.upperBound, r2.lowerBound)

        // Memory-proportional assignment: strictly-decreasing memory inputs
        // must produce strictly-decreasing layer spans. This is the core
        // contract of plan_memory_aware, expressed as an invariant rather
        // than pinned specific layer counts.
        XCTAssertGreaterThan(r0.count, r1.count)
        XCTAssertGreaterThan(r1.count, r2.count)
    }

    func testLayerRangeOutOfRangeReturnsNil() {
        guard let builder = ClusterPlanBuilder(numLayers: 64, numHeads: 28, headDim: 128) else {
            XCTFail("Builder init failed with valid dims")
            return
        }
        XCTAssertTrue(builder.addNode(hostname: "host-a.local", memoryGB: 128.0))
        XCTAssertTrue(builder.addNode(hostname: "host-b.local", memoryGB: 64.0))
        XCTAssertTrue(builder.plan())

        XCTAssertNil(builder.layerRange(forRank: 2))
        XCTAssertNil(builder.layerRange(forRank: -1))
    }

    func testPlanTwiceReturnsFalse() {
        guard let builder = ClusterPlanBuilder(numLayers: 64, numHeads: 28, headDim: 128) else {
            XCTFail("Builder init failed with valid dims")
            return
        }
        XCTAssertTrue(builder.addNode(hostname: "host-a.local", memoryGB: 128.0))
        XCTAssertTrue(builder.addNode(hostname: "host-b.local", memoryGB: 64.0))
        XCTAssertTrue(builder.plan())
        XCTAssertFalse(builder.plan())
    }
}
