// InMemoryClusterKeyStore tests. The store itself is debug-only (wrapped
// in `#if DEBUG` so it does not ship in release binaries) — this test
// file is guarded the same way so a release-configuration test build
// still compiles. Tests normally run in debug; this guard is defense
// against unusual build setups.

#if DEBUG

import XCTest
import TurboQuantKit

final class InMemoryClusterKeyStoreTests: XCTestCase {

    private func sampleRecord(_ b: UInt8) -> ClusterRecord {
        ClusterRecord(
            clusterId: Data(repeating: b, count: 16),
            key: Data(repeating: b &+ 1, count: 32)
        )
    }

    func testRoundTripAndOverwriteAndDelete() throws {
        let store = InMemoryClusterKeyStore()
        XCTAssertNil(try store.load())

        let a = sampleRecord(0x11)
        try store.save(a)
        XCTAssertEqual(try store.load(), a)

        let b = sampleRecord(0x22)
        try store.save(b)
        XCTAssertEqual(try store.load(), b)

        try store.delete()
        XCTAssertNil(try store.load())
    }

    func testInitialRecordIsReturnedOnFirstLoad() throws {
        // Some callers prefer to seed the store instead of calling save
        // after construction; the `initial:` convenience on the init
        // handles that use case.
        let seeded = sampleRecord(0x42)
        let store = InMemoryClusterKeyStore(initial: seeded)
        XCTAssertEqual(try store.load(), seeded)
    }
}

#endif // DEBUG
