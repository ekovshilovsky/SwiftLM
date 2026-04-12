import XCTest
import TurboQuantKit

final class DistributedCoordinatorTests: XCTestCase {

    func testInitLocalReturnsNilWithoutCLibrary() {
        // TurboQuantC is not linked in the standalone SwiftLM SPM build.
        // The #else branch in initializeLocal() returns nil unconditionally,
        // which is the expected fallback for single-node inference without the
        // C library present. The server startup path must handle nil gracefully.
        let coord = DistributedCoordinator.initializeLocal()
        XCTAssertNil(coord)
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
