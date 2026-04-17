// Tests for DistributedCLIOptions parsing and validation. The
// executable's ArgumentParser bindings live in Server.swift and are not
// directly testable (SPM doesn't support @testable on executables), so
// the logic under test here is the value-type validation that Server.swift
// calls once it has parsed the raw flags.

import XCTest
import TurboQuantKit

final class DistributedCLIOptionsTests: XCTestCase {

    // MARK: - Valid combinations

    func testDefaultIsValid() throws {
        try DistributedCLIOptions().validate()
    }

    func testDistributedAloneIsValid() throws {
        try DistributedCLIOptions(isDistributed: true).validate()
    }

    func testDistributedWithAutoIsValid() throws {
        try DistributedCLIOptions(isDistributed: true, isAuto: true).validate()
    }

    func testDistributedWithRoleIsValid() throws {
        try DistributedCLIOptions(isDistributed: true, role: .primary).validate()
        try DistributedCLIOptions(isDistributed: true, role: .secondary).validate()
    }

    func testDistributedWithSnapshotIntervalIsValid() throws {
        try DistributedCLIOptions(isDistributed: true, snapshotInterval: 500).validate()
    }

    func testClusterStatusIsValidWithoutDistributed() throws {
        // --cluster-status inspects local persisted state only, so it
        // must be usable without --distributed (e.g. to see whether a
        // cluster record exists from a prior join).
        try DistributedCLIOptions(printClusterStatus: true).validate()
    }

    // MARK: - Invalid combinations

    func testAutoRequiresDistributed() {
        XCTAssertThrowsError(try DistributedCLIOptions(isAuto: true).validate()) {
            XCTAssertEqual($0 as? DistributedCLIOptionsError, .autoRequiresDistributed)
        }
    }

    func testRoleRequiresDistributed() {
        XCTAssertThrowsError(try DistributedCLIOptions(role: .primary).validate()) {
            XCTAssertEqual($0 as? DistributedCLIOptionsError, .roleRequiresDistributed)
        }
    }

    func testSnapshotIntervalRequiresDistributed() {
        XCTAssertThrowsError(
            try DistributedCLIOptions(snapshotInterval: 1000).validate()
        ) {
            XCTAssertEqual($0 as? DistributedCLIOptionsError,
                           .snapshotIntervalRequiresDistributed)
        }
    }

    func testSnapshotIntervalMustBePositive() {
        for bad in [0, -1, -500] {
            XCTAssertThrowsError(
                try DistributedCLIOptions(
                    isDistributed: true, snapshotInterval: bad
                ).validate()
            ) {
                XCTAssertEqual($0 as? DistributedCLIOptionsError,
                               .snapshotIntervalOutOfRange(bad))
            }
        }
    }

    // MARK: - Role parsing

    func testParseRoleReturnsNilForNil() throws {
        XCTAssertNil(try DistributedCLIOptions.parseRole(nil))
    }

    func testParseRoleKnownValues() throws {
        XCTAssertEqual(try DistributedCLIOptions.parseRole("primary"), .primary)
        XCTAssertEqual(try DistributedCLIOptions.parseRole("secondary"), .secondary)
    }

    func testParseRoleRejectsUnknown() {
        XCTAssertThrowsError(
            try DistributedCLIOptions.parseRole("captain")
        ) {
            XCTAssertEqual($0 as? DistributedCLIOptionsError,
                           .invalidRole("captain"))
        }
    }

    func testParseRoleIsCaseSensitive() {
        // The CLI advertises lowercase values; accepting "Primary" would
        // hide typos. Strict matching surfaces them.
        XCTAssertThrowsError(
            try DistributedCLIOptions.parseRole("Primary")
        )
    }
}
