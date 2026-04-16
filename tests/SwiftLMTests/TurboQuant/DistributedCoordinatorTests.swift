import XCTest
import TurboQuantKit

final class DistributedCoordinatorTests: XCTestCase {

    func testInitLocalReturnsSingleNodeCoordinator() {
        // With TurboQuantC linked, tq_distributed_init_local() creates a
        // single-node coordinator with rank=0 and worldSize=1.
        let coord = DistributedCoordinator.initializeLocal()
        XCTAssertNotNil(coord, "Local coordinator should initialize for single-node mode")
    }

    func testInitWithInvalidHostfileReturnsNil() {
        // A hostfile path that does not exist must produce nil rather than
        // crashing, regardless of whether TurboQuantC is linked. Callers must
        // be able to detect initialization failure and abort startup with a
        // meaningful error rather than proceeding with an unconfigured cluster.
        let coord = DistributedCoordinator.initialize(hostfile: "/nonexistent/hostfile.json")
        XCTAssertNil(coord)
    }
}
