// Routing-decision regression suite for the chat-completions
// dispatcher. The full HTTP handler lives in the SwiftLM executable
// target and depends on a built model container, the prompt cache,
// and the Hummingbird response builders — none of which are
// reachable from the unit test target. The decision the handler
// branches on, however, is a pure function exposed by
// `TurboQuantKit` (`decideChatCompletionRoute`), so the regression
// coverage anchors on that helper.
//
// The three brief-mandated tests assert the truth table the dispatch
// logic has to satisfy:
//
//   1. Joiner-role bring-up always returns 503 (never reaches the
//      single-node path, never hits the distributed engine).
//   2. Coordinator bring-up without a `coordinatorModel` falls
//      through to the single-node path even when joiners are
//      attached. The distributed branch is wired in code but
//      unreachable until the model loader landing follow-up provides
//      a `DistributedQwenModel`.
//   3. The no-cluster path (no `ClusterBringUp` in scope) is the
//      single-node path — guards against the routing helper
//      regressing the default behaviour.

#if DEBUG

import Foundation
import XCTest
@testable import TurboQuantKit

final class DistributedChatCompletionsTests: XCTestCase {

    // MARK: - Fixture

    /// Build a `ClusterManager` with the in-memory key store and a
    /// throw-away `BonjourService`. The manager is never started — we
    /// exercise only its public `attachedJoinerCount()` getter, which
    /// returns zero for a freshly-constructed manager. Tests that
    /// need a non-zero joiner count construct a stub bring-up with
    /// the count baked into the helper invocation directly, since
    /// driving real handshakes is out of scope for the routing-rule
    /// regression and is covered separately by the
    /// `DistributedBringUp` and `ClusterManagerSession` suites.
    private func makeIdleManager(role: DiscoveryRole) -> ClusterManager {
        let info = DiscoveryInfo(
            model: "test-model",
            memoryGB: 16,
            role: role,
            clusterHash: nil,
            clusterId: nil,
            version: "0.1.0",
            rdma: .unsupported,
            name: "test-host"
        )
        let bonjour = BonjourService(info: info, localClusterHash: nil)
        return ClusterManager(
            keyStore: InMemoryClusterKeyStore(),
            bonjour: bonjour,
            localHostname: "test-host",
            model: "test-model",
            memoryGB: 16,
            version: "0.1.0",
            rdma: .unsupported
        )
    }

    private func makeBringUp(
        role: DistributedNodeRole,
        coordinatorModel: DistributedQwenModel? = nil
    ) -> ClusterBringUp {
        let manager = makeIdleManager(role: role == .primary ? .coordinator : .worker)
        return ClusterBringUp(
            manager: manager,
            role: role,
            workerTask: nil,
            coordinatorModel: coordinatorModel
        )
    }

    // MARK: - Test 1: joiner role short-circuits with 503

    /// A `.secondary` bring-up must surface `.joinerUnavailable`
    /// regardless of whether joiners are attached or whether a
    /// coordinator-side model is present. Joiners do not sample, so
    /// the chat-completions handler maps this to a 503 response with
    /// a `joiner_role` error code.
    func testJoinerRoleReturns503() {
        let bringUp = makeBringUp(role: .secondary)
        let decision = decideChatCompletionRoute(
            clusterBringUp: bringUp,
            attachedJoinerCount: 0
        )
        XCTAssertEqual(decision, .joinerUnavailable,
                       "joiner-role bring-up must short-circuit to the 503 branch")

        // The attached-count and coordinator-model presence must not
        // influence the joiner branch — the role is the load-bearing
        // signal. Re-running with a non-zero count and (hypothetical)
        // model still has to surface the same decision.
        let bringUpWithCount = makeBringUp(role: .secondary, coordinatorModel: nil)
        let decisionWithCount = decideChatCompletionRoute(
            clusterBringUp: bringUpWithCount,
            attachedJoinerCount: 4
        )
        XCTAssertEqual(decisionWithCount, .joinerUnavailable,
                       "joiner-role bring-up must short-circuit even when peers are present")
    }

    // MARK: - Test 2: coordinator without model falls through

    /// `.primary` bring-up with attached joiners but no
    /// `coordinatorModel` must fall through to the single-node path.
    /// This is the v1 production state — the coordinator-side model
    /// loader integration lands separately, and until it does the
    /// distributed branch is unreachable in the deployed binary
    /// while staying wired in source for follow-up integration.
    func testCoordinatorWithoutModelFallsThroughToSingleNode() {
        let bringUp = makeBringUp(role: .primary, coordinatorModel: nil)
        let decision = decideChatCompletionRoute(
            clusterBringUp: bringUp,
            attachedJoinerCount: 2
        )
        XCTAssertEqual(decision, .singleNode,
                       "coordinator without coordinatorModel must use the single-node path even when joiners are attached")

        // No joiners attached + no model is also single-node — covers
        // the freshly-bootstrapped coordinator that has not yet seen
        // any peers complete the handshake.
        let bringUpNoPeers = makeBringUp(role: .primary, coordinatorModel: nil)
        let decisionNoPeers = decideChatCompletionRoute(
            clusterBringUp: bringUpNoPeers,
            attachedJoinerCount: 0
        )
        XCTAssertEqual(decisionNoPeers, .singleNode,
                       "coordinator with no peers must use the single-node path")
    }

    // MARK: - Test 3: no cluster bring-up is unchanged

    /// Servers started without `--distributed` carry no
    /// `ClusterBringUp` handle. The decision must surface `.singleNode`
    /// so the existing handler runs untouched. This is the binding
    /// regression check — the no-cluster path must remain
    /// byte-identical to the pre-routing implementation.
    func testNonDistributedSingleNodeUnchanged() {
        let decision = decideChatCompletionRoute(
            clusterBringUp: nil,
            attachedJoinerCount: 0
        )
        XCTAssertEqual(decision, .singleNode,
                       "no cluster bring-up must take the single-node path")

        // The attached-count input is irrelevant when the bring-up is
        // nil. The helper must ignore the count rather than spuriously
        // routing to the distributed branch on a non-zero value.
        let decisionStaleCount = decideChatCompletionRoute(
            clusterBringUp: nil,
            attachedJoinerCount: 99
        )
        XCTAssertEqual(decisionStaleCount, .singleNode,
                       "non-distributed servers must ignore the attached-count input")
    }

    // MARK: - Test 4: cluster manager attached-joiner count starts at zero

    /// The chat-completions handler reads
    /// `ClusterManager.attachedJoinerCount()` to gate the distributed
    /// branch. A freshly-constructed manager has not seen any
    /// handshakes, so the count must start at zero. Guards against a
    /// regression in the manager's accessor that would route through
    /// the distributed engine before any joiners attach.
    func testFreshClusterManagerReportsZeroJoiners() async {
        let manager = makeIdleManager(role: .coordinator)
        let count = await manager.attachedJoinerCount()
        XCTAssertEqual(count, 0,
                       "a manager that has not run createCluster/joinCluster must report zero attached joiners")
    }
}

#endif // DEBUG
