// Pure value-to-text formatter for cluster topology status. Takes a
// fully-populated ClusterTopology value and returns a human-readable
// report suitable for terminal output. Performs no I/O and does not
// read from the network, filesystem, or process environment — callers
// assemble the topology value from runtime state and hand it in.

import Foundation

/// Formats cluster topology information as human-readable terminal output
/// with actionable recommendations for missing Thunderbolt links and
/// RDMA configuration steps.
public struct TopologyReporter {

    /// Full cluster topology snapshot passed to `format(_:)`.
    public struct ClusterTopology {
        public let nodes: [NodeSummary]
        public let links: [LinkInfo]
        public let strategy: String
        /// Free-form topology label surfaced to the user, e.g. `"fully_connected"`,
        /// `"chain"`, or `"partial_mesh"`. Not validated here; callers pass
        /// whatever shape name makes sense for the detected layout.
        public let topologyType: String

        public init(
            nodes: [NodeSummary],
            links: [LinkInfo],
            strategy: String,
            topologyType: String
        ) {
            self.nodes = nodes
            self.links = links
            self.strategy = strategy
            self.topologyType = topologyType
        }
    }

    /// One node's contribution to the cluster report.
    public struct NodeSummary {
        public let hostname: String
        public let memoryGB: Int
        public let rank: Int
        /// Inclusive start of the layer range assigned to this node.
        public let layerStart: Int
        /// Exclusive end of the layer range. A node owning layers 0..<8
        /// reports `layerStart = 0, layerEnd = 8` and renders as `layers 0-7`.
        public let layerEnd: Int
        public let rdma: RdmaCapability

        public init(
            hostname: String,
            memoryGB: Int,
            rank: Int,
            layerStart: Int,
            layerEnd: Int,
            rdma: RdmaCapability
        ) {
            self.hostname = hostname
            self.memoryGB = memoryGB
            self.rank = rank
            self.layerStart = layerStart
            self.layerEnd = layerEnd
            self.rdma = rdma
        }
    }

    /// A measured (or absent) point-to-point link between two nodes.
    /// `isDirect == false` is how the reporter emits an actionable
    /// "connect a Thunderbolt cable" recommendation; latency is ignored
    /// for indirect links since routing through a third node yields a
    /// number that would mislead the operator.
    public struct LinkInfo {
        public let nodeA: String
        public let nodeB: String
        public let latencyUs: Double
        public let isDirect: Bool
        public let rdmaStatus: RdmaCapability

        public init(
            nodeA: String,
            nodeB: String,
            latencyUs: Double,
            isDirect: Bool,
            rdmaStatus: RdmaCapability
        ) {
            self.nodeA = nodeA
            self.nodeB = nodeB
            self.latencyUs = latencyUs
            self.isDirect = isDirect
            self.rdmaStatus = rdmaStatus
        }
    }

    /// Render the full cluster status report as a single string with
    /// newline-separated lines. The output has no trailing newline so
    /// callers choose whether to append one.
    public static func format(_ topology: ClusterTopology) -> String {
        var lines: [String] = []
        lines.append("[TurboQuant Cluster]")
        lines.append("  Topology: \(topology.nodes.count) nodes, \(topology.topologyType)")
        lines.append("  Strategy: \(topology.strategy)")
        lines.append("")
        lines.append("  Connectivity:")
        for link in topology.links {
            let status: String
            if link.isDirect {
                status = "\(link.rdmaStatus.rawValue) \(String(format: "%.0f", link.latencyUs))us"
            } else {
                status = "No direct link"
            }
            lines.append("    \(link.nodeA) <-> \(link.nodeB)   \(status)")
        }
        lines.append("")
        lines.append("  Layer assignment:")
        for node in topology.nodes {
            // layerEnd is exclusive; the display range is inclusive on both
            // sides, hence the -1 on the upper bound.
            lines.append("    \(node.hostname) (\(node.memoryGB)GB): layers \(node.layerStart)-\(node.layerEnd - 1)")
        }

        let missingLinks = topology.links.filter { !$0.isDirect }
        if !missingLinks.isEmpty {
            lines.append("")
            for link in missingLinks {
                lines.append("  ACTION NEEDED: Connect Thunderbolt cable \(link.nodeA) <-> \(link.nodeB)")
            }
        }

        // `disabled` means hardware is present but recovery-mode activation
        // has not been run; `unsupported` means the hardware cannot do RDMA
        // at all. Only the fixable case gets an action block.
        let rdmaDisabled = topology.nodes.filter { $0.rdma == .disabled }
        if !rdmaDisabled.isEmpty {
            lines.append("")
            for node in rdmaDisabled {
                lines.append("  RDMA NOT ENABLED: \(node.hostname)")
                lines.append("    Boot into macOS Recovery -> Utilities -> Terminal -> rdma_ctl enable")
            }
        }

        return lines.joined(separator: "\n")
    }
}
