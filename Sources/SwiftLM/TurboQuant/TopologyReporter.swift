// Pure value-to-text formatter for cluster topology status. Takes a
// fully-populated ClusterTopology value and returns a human-readable
// report suitable for terminal output. Performs no I/O and does not
// read from the network, filesystem, or process environment — callers
// assemble the topology value from runtime state and hand it in.

import Foundation

/// Shape of the cluster's physical connectivity graph. Raw values are
/// the terminal-rendered form; Swift-level naming is camelCased per
/// language convention.
public enum ClusterTopologyType: String, Sendable {
    /// Every pair of nodes has a direct point-to-point link.
    case fullyConnected = "fully_connected"
    /// Nodes arranged in a line: A–B–C–D. Common on small clusters
    /// limited by Thunderbolt port count.
    case chain = "chain"
    /// Some pairs direct, some routed through an intermediary. Usually
    /// actionable via an additional cable.
    case partialMesh = "partial_mesh"
}

/// Formats cluster topology information as human-readable terminal output
/// with actionable recommendations for missing Thunderbolt links and
/// RDMA configuration steps.
public struct TopologyReporter {

    /// Full cluster topology snapshot passed to `format(_:)`.
    public struct ClusterTopology {
        public let nodes: [NodeSummary]
        public let links: [LinkInfo]
        public let strategy: String
        public let topologyType: ClusterTopologyType

        public init(
            nodes: [NodeSummary],
            links: [LinkInfo],
            strategy: String,
            topologyType: ClusterTopologyType
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

    /// A point-to-point link between two nodes, direct or indirect.
    /// `latencyUs` is optional: nil means the latency has not been
    /// measured or is not applicable (e.g., an indirect link routed
    /// through a third node, where a measured number would mislead
    /// the operator).
    public struct LinkInfo {
        public let nodeA: String
        public let nodeB: String
        public let latencyUs: Double?
        public let isDirect: Bool
        public let rdmaStatus: RdmaCapability

        public init(
            nodeA: String,
            nodeB: String,
            latencyUs: Double?,
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
    ///
    /// For a link with `isDirect == false`, the reporter prints
    /// "No direct link" and emits an "ACTION NEEDED: Connect
    /// Thunderbolt cable" recommendation below; any `latencyUs` on
    /// such a link is ignored because a number measured via routing
    /// through a third node would mislead the operator.
    public static func format(_ topology: ClusterTopology) -> String {
        var lines: [String] = []
        lines.append("[TurboQuant Cluster]")
        lines.append("  Topology: \(topology.nodes.count) nodes, \(topology.topologyType.rawValue)")
        lines.append("  Strategy: \(topology.strategy)")
        lines.append("")
        lines.append("  Connectivity:")
        for link in topology.links {
            let status: String
            if link.isDirect {
                if let latency = link.latencyUs {
                    status = "\(link.rdmaStatus.rawValue) \(String(format: "%.0f", latency))us"
                } else {
                    status = "\(link.rdmaStatus.rawValue) (unmeasured)"
                }
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
