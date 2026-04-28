// Routing-decision helper for the `/v1/chat/completions` endpoint
// when distributed cluster bring-up is in scope. The HTTP handler in
// the executable target captures the bring-up handle and consults
// `decideChatCompletionRoute` before tokenising the request, then
// branches to one of three implementations: a 503 short-circuit for
// joiner nodes, the distributed engine for an attached coordinator,
// or the existing single-node path for everything else.
//
// The decision is split out from the handler for two reasons:
//
//   1. The decision is a pure function of three inputs (presence /
//      absence of bring-up, role, joiner count, presence / absence of
//      a coordinator-side `DistributedQwenModel`). Splitting it makes
//      the truth table explicit and unit-testable without standing
//      up the HTTP server.
//
//   2. The unit test target depends on `TurboQuantKit` only; the
//      executable target's `handleChatCompletion` is not reachable
//      from tests. Keeping the decision here lets the regression
//      suite cover the routing rules without the heavy dependency
//      chain.

import Foundation

/// One of three branches the chat-completions handler can take when
/// a `ClusterBringUp` handle is in scope. The distinction between
/// `joinerUnavailable` and `singleNode` is load-bearing: joiners must
/// surface a 503 with an actionable message, whereas the single-node
/// path is the default for non-distributed servers and for
/// coordinators that are not yet equipped to drive the distributed
/// engine.
public enum ChatCompletionRouteDecision: Sendable, Equatable {

    /// This process is running as a cluster joiner. Joiners forward
    /// activations to the coordinator over the cluster's collective
    /// transport but do not sample tokens, so they cannot serve
    /// chat-completion requests. The HTTP handler responds with 503
    /// and a `joiner_role` error code.
    case joinerUnavailable

    /// This process is the cluster coordinator with at least one
    /// joiner attached and a coordinator-side `DistributedQwenModel`
    /// available to drive the engine. The HTTP handler routes through
    /// `ClusterManager.beginInferenceSession`.
    case distributed

    /// Default branch: no cluster bring-up, or a coordinator with no
    /// joiners attached, or a coordinator without a TurboQuant-loaded
    /// model. The HTTP handler runs the existing single-node
    /// generation path.
    case singleNode
}

/// Decide which chat-completions branch to take given the state of
/// the cluster bring-up. The function is pure — it takes already-
/// resolved values rather than entering the manager actor — so the
/// caller can read `attachedJoinerCount()` once at request time and
/// pass the snapshot in.
///
/// - Parameters:
///   - clusterBringUp: The bring-up handle captured by the HTTP
///     server. `nil` covers servers started without `--distributed`.
///   - attachedJoinerCount: Snapshot of
///     `clusterBringUp.manager.attachedJoinerCount()` taken at
///     request time. Ignored when `clusterBringUp` is `nil`.
public func decideChatCompletionRoute(
    clusterBringUp: ClusterBringUp?,
    attachedJoinerCount: Int
) -> ChatCompletionRouteDecision {
    guard let bringUp = clusterBringUp else {
        return .singleNode
    }
    switch bringUp.role {
    case .secondary:
        return .joinerUnavailable
    case .primary:
        // The distributed branch requires both a non-empty joiner
        // list (no point fanning out to nobody) and a
        // coordinator-side model the engine can drive. The
        // coordinator-model wiring is staged behind a follow-up that
        // produces a `DistributedQwenModel` from the model loader;
        // until that lands the field is `nil` in production and the
        // coordinator transparently falls back to the single-node
        // path.
        if attachedJoinerCount > 0 && bringUp.coordinatorModel != nil {
            return .distributed
        }
        return .singleNode
    }
}
