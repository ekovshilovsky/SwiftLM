// CLI options for the Bonjour-based distributed mode. Lives in
// TurboQuantKit (not in Server.swift) so the parsing + validation logic
// can be unit-tested — Swift Package Manager does not support
// @testable import on executable targets, so anything testable must
// sit in a library target.

import Foundation

/// Node role override. Cluster formation auto-detects a role from
/// RDMA / link capability; this flag lets the user force a specific
/// role when the auto-detection is wrong or undesirable.
public enum DistributedNodeRole: String, Sendable {
    case primary
    case secondary
}

/// Parsed CLI options for distributed-mode behavior. Produced from the
/// executable's ArgumentParser flags and validated once before the
/// server runs any distributed work.
public struct DistributedCLIOptions: Sendable, Equatable {
    /// Whether the user requested distributed mode at all. All other
    /// options are dependent on this.
    public let isDistributed: Bool
    /// Auto-join any known cluster or create a new one without
    /// interactive prompting.
    public let isAuto: Bool
    /// Role override, or nil to auto-detect at join time.
    public let role: DistributedNodeRole?
    /// Background-snapshot cadence in tokens. Applies to Secondary
    /// nodes that may disconnect. Nil means use the default (1000).
    public let snapshotInterval: Int?
    /// Print cluster topology and exit. Mutually compatible with
    /// `isDistributed=false` because it only reads local state.
    public let printClusterStatus: Bool
    /// Print the loaded model's layer type breakdown and exit.
    public let printLayerTypeReport: Bool

    public init(isDistributed: Bool = false,
                isAuto: Bool = false,
                role: DistributedNodeRole? = nil,
                snapshotInterval: Int? = nil,
                printClusterStatus: Bool = false,
                printLayerTypeReport: Bool = false) {
        self.isDistributed = isDistributed
        self.isAuto = isAuto
        self.role = role
        self.snapshotInterval = snapshotInterval
        self.printClusterStatus = printClusterStatus
        self.printLayerTypeReport = printLayerTypeReport
    }
}

public enum DistributedCLIOptionsError: Error, Equatable, CustomStringConvertible {
    case autoRequiresDistributed
    case roleRequiresDistributed
    case snapshotIntervalRequiresDistributed
    case snapshotIntervalOutOfRange(Int)
    case invalidRole(String)

    public var description: String {
        switch self {
        case .autoRequiresDistributed:
            return "--auto requires --distributed"
        case .roleRequiresDistributed:
            return "--role requires --distributed"
        case .snapshotIntervalRequiresDistributed:
            return "--snapshot-interval requires --distributed"
        case .snapshotIntervalOutOfRange(let value):
            return "--snapshot-interval must be a positive integer (got \(value))"
        case .invalidRole(let value):
            return "--role must be 'primary' or 'secondary' (got '\(value)')"
        }
    }
}

extension DistributedCLIOptions {
    /// Validate the combination of parsed flags. Throws on the first
    /// problem found so the executable can surface a clear error before
    /// doing any work.
    public func validate() throws {
        if isAuto && !isDistributed {
            throw DistributedCLIOptionsError.autoRequiresDistributed
        }
        if role != nil && !isDistributed {
            throw DistributedCLIOptionsError.roleRequiresDistributed
        }
        if let interval = snapshotInterval {
            if !isDistributed {
                throw DistributedCLIOptionsError.snapshotIntervalRequiresDistributed
            }
            if interval <= 0 {
                throw DistributedCLIOptionsError.snapshotIntervalOutOfRange(interval)
            }
        }
    }

    /// Parse the --role CLI string into the enum, with a precise error
    /// for typos. Nil maps to nil (no override).
    public static func parseRole(_ raw: String?) throws -> DistributedNodeRole? {
        guard let raw else { return nil }
        guard let role = DistributedNodeRole(rawValue: raw) else {
            throw DistributedCLIOptionsError.invalidRole(raw)
        }
        return role
    }
}
