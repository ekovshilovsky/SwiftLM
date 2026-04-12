import XCTest
@testable import SwiftLM

final class DistributedCoordinatorTests: XCTestCase {

    func testInitLocalReturnsCoordinator() {
        // Stub returns nil until C API connected
        let coord = DistributedCoordinator.initializeLocal()
        // Phase 4: change to XCTAssertNotNil
        XCTAssertNil(coord)
    }

    func testInitWithInvalidHostfileReturnsNil() {
        let coord = DistributedCoordinator.initialize(hostfile: "/nonexistent/hostfile.json")
        XCTAssertNil(coord)
    }
}
