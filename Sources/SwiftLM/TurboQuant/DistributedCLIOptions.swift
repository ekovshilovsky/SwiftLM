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
    /// Direct cluster passphrase via `--passphrase <value>`. Mutually
    /// exclusive with `passphraseFile` and `passphraseCommand`.
    public let passphrase: String?
    /// Path to a file containing the cluster passphrase via
    /// `--passphrase-file <path>`. Mutually exclusive with `passphrase`
    /// and `passphraseCommand`.
    public let passphraseFile: String?
    /// Shell command to execute and capture stdout for the cluster
    /// passphrase via `--passphrase-command <cmd>`. Mutually exclusive
    /// with `passphrase` and `passphraseFile`.
    public let passphraseCommand: String?

    public init(isDistributed: Bool = false,
                isAuto: Bool = false,
                role: DistributedNodeRole? = nil,
                snapshotInterval: Int? = nil,
                printClusterStatus: Bool = false,
                printLayerTypeReport: Bool = false,
                passphrase: String? = nil,
                passphraseFile: String? = nil,
                passphraseCommand: String? = nil) {
        self.isDistributed = isDistributed
        self.isAuto = isAuto
        self.role = role
        self.snapshotInterval = snapshotInterval
        self.printClusterStatus = printClusterStatus
        self.printLayerTypeReport = printLayerTypeReport
        self.passphrase = passphrase
        self.passphraseFile = passphraseFile
        self.passphraseCommand = passphraseCommand
    }
}

public enum DistributedCLIOptionsError: Error, Equatable, CustomStringConvertible {
    case autoRequiresDistributed
    case roleRequiresDistributed
    case snapshotIntervalRequiresDistributed
    case snapshotIntervalOutOfRange(Int)
    case invalidRole(String)
    case passphraseFlagRequiresDistributed(String)
    case multiplePassphraseSources([String])

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
        case .passphraseFlagRequiresDistributed(let flag):
            return "\(flag) requires --distributed"
        case .multiplePassphraseSources(let names):
            return "multiple passphrase sources specified: \(names.joined(separator: ", ")). Use exactly one of --passphrase, --passphrase-file, --passphrase-command."
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
        // Passphrase flags only make sense alongside --distributed
        // because they configure cluster authentication. Reject early
        // to avoid leaving an operator wondering why the value is
        // ignored.
        if !isDistributed {
            if passphrase != nil {
                throw DistributedCLIOptionsError
                    .passphraseFlagRequiresDistributed("--passphrase")
            }
            if passphraseFile != nil {
                throw DistributedCLIOptionsError
                    .passphraseFlagRequiresDistributed("--passphrase-file")
            }
            if passphraseCommand != nil {
                throw DistributedCLIOptionsError
                    .passphraseFlagRequiresDistributed("--passphrase-command")
            }
        }
        // The three passphrase sources are mutually exclusive. Catching
        // the conflict here lets the resolver assume at most one source
        // is set, simplifying its decision tree.
        var specified: [String] = []
        if passphrase != nil { specified.append("--passphrase") }
        if passphraseFile != nil { specified.append("--passphrase-file") }
        if passphraseCommand != nil { specified.append("--passphrase-command") }
        if specified.count > 1 {
            throw DistributedCLIOptionsError.multiplePassphraseSources(specified)
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
