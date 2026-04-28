// Top-level facade that composes the distributed model, the per-rank
// joiner control channels, and the per-request coordinator session
// into a single object the HTTP request handler can call against.
//
// Lifecycle. The engine owns one `DistributedQwenModel` instance for
// the entire process lifetime: the model is large (multiple GB of
// weights resident on the Metal heap) and it has no per-request
// state, so reusing it across every concurrent or sequential request
// is the only sane strategy. The engine constructs a fresh
// `CoordinatorInferenceSession` per request because each session
// owns its own KV cache, sampler, and broadcast bookkeeping.
//
// Streaming. `generate(_:)` returns an `AsyncThrowingStream<TokenEvent>`
// rather than an `AsyncStream<TokenEvent>` so transport-level errors
// (a torn-down joiner channel, a forward-pass crash) and contract
// violations (cancellation propagating from the consumer) surface
// naturally on the consuming `for try await` loop. The session itself
// also represents the same conditions as `FinishReason.error(_:)`
// events; the engine keeps both surfaces alive so the HTTP layer
// can choose between forwarding the error event verbatim (preferred
// for SSE clients that want to render a final "error: ..." frame)
// and rethrowing on the stream (preferred for callers that drive a
// retry loop).
//
// Cancellation. The engine wires the session task into the stream's
// `onTermination` callback. When the consumer cancels its iteration
// (HTTP client disconnects, request handler times out), the stream
// terminates, the callback fires, and the session task is cancelled.
// The session's internal cancellation handling broadcasts
// `sessionEnd` so joiners release their per-session state cleanly.

import Foundation
import MLX
import MLXLMCommon

/// Process-level facade for the distributed inference path. One
/// instance per process; HTTP handlers call `generate(_:)` per
/// request to obtain a stream of token events.
public actor DistributedInferenceEngine {

    // MARK: - Stored state

    /// The distributed model, loaded once for the process lifetime.
    /// Owned by the engine because the model has no per-request state
    /// and reloading multi-GB weights per request is not viable.
    private let model: DistributedQwenModel

    /// Joiner control channels. Identical to what each per-request
    /// session is constructed with; held on the engine because the
    /// channel set is process-scoped (one entry per cluster member,
    /// established at cluster handshake time).
    private let joiners: [InferenceControlChannel]

    /// Sampler factory. Invoked once per request with the request's
    /// `SamplingParams`. Held as a closure rather than a stored
    /// sampler so each session gets its own sampler instance with a
    /// fresh RNG seeded from the request's seed — sharing one sampler
    /// across requests would entangle their RNG streams and break the
    /// per-request reproducibility contract.
    private let samplerFactory: @Sendable (CoordinatorInferenceSession.InferenceRequest) -> any DistributedSampler

    // MARK: - Init

    /// Construct an engine. The model and channels are captured by
    /// reference; the factory closure is invoked per `generate` call.
    /// The default factory routes greedy requests through
    /// `GreedySampler` and stochastic requests through
    /// `TemperatureSampler` seeded from `request.seed`.
    public init(
        model: DistributedQwenModel,
        joiners: [InferenceControlChannel],
        samplerFactory: @escaping @Sendable (CoordinatorInferenceSession.InferenceRequest) -> any DistributedSampler
            = DistributedInferenceEngine.defaultSamplerFactory
    ) {
        self.model = model
        self.joiners = joiners
        self.samplerFactory = samplerFactory
    }

    /// Default sampler factory. Returns `GreedySampler` for
    /// `temperature == 0` (the conventional argmax shorthand) and
    /// `TemperatureSampler` for everything else, seeded from the
    /// request's `seed` so two requests with identical inputs and
    /// seeds produce identical token streams.
    public static let defaultSamplerFactory:
        @Sendable (CoordinatorInferenceSession.InferenceRequest) -> any DistributedSampler
        = { request in
            if request.sampling.temperature == 0 {
                return GreedySampler()
            }
            return TemperatureSampler(seed: request.seed)
        }

    // MARK: - Generate

    /// Run one inference request to completion. The returned stream
    /// yields one `.delta(token:)` per generated token followed by
    /// exactly one `.finish(reason:)` and then terminates. Errors on
    /// the underlying session task surface as the stream throwing on
    /// the consumer's `for try await` loop.
    ///
    /// The session task is cancelled if the consumer cancels its
    /// iteration (HTTP client disconnect, request-handler timeout)
    /// via the stream's `onTermination` callback.
    public func generate(
        _ request: CoordinatorInferenceSession.InferenceRequest
    ) -> AsyncThrowingStream<CoordinatorInferenceSession.TokenEvent, Error> {
        // Build the per-request collaborators. Capturing `model` once
        // here keeps the forward closure free of `self` — the engine
        // actor stays out of the model's hot path.
        let model = self.model
        let joiners = self.joiners
        let sampler = samplerFactory(request)

        let session = CoordinatorInferenceSession(
            forward: { ids, cache in model(ids, cache: cache) },
            makeCache: { model.makeCache() },
            joiners: joiners,
            sampler: sampler
        )

        return AsyncThrowingStream { continuation in
            // Bridge the session's `AsyncStream<TokenEvent>` continuation
            // shape to the engine's `AsyncThrowingStream<TokenEvent>`.
            // The session writes deltas / finish events into its own
            // continuation; this task awaits `serve`, then drains any
            // events still buffered on the inner stream before
            // terminating the outer throwing stream. Draining inside
            // the same task (rather than a parallel forwarder) avoids
            // a race where the outer stream closes before the
            // forwarder has flushed the trailing `.finish` event.
            let (sessionStream, sessionContinuation) = AsyncStream
                .makeStream(of: CoordinatorInferenceSession.TokenEvent.self)

            let serveTask = Task {
                // Run `serve` concurrently with the inner-stream
                // drain so deltas yielded mid-decode flow through
                // the outer continuation as soon as they are
                // produced rather than after `serve` returns.
                let serveResult: Task<Void, Error> = Task {
                    try await session.serve(request, stream: sessionContinuation)
                }
                for await event in sessionStream {
                    continuation.yield(event)
                }
                // The inner stream closed: `serve` is done writing.
                // Await its result so any error is surfaced on the
                // outer stream.
                do {
                    try await serveResult.value
                    continuation.finish()
                } catch is CancellationError {
                    // The session emitted `.finish(.cancelled)`
                    // before throwing; the consumer has already seen
                    // it through the drain loop above. Close the
                    // outer stream without an error so a cancelled
                    // iteration ends rather than throws.
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                // Consumer iteration ended (natural completion or
                // explicit cancellation). Cancel the serve task so
                // the actor's CancellationError branch broadcasts
                // `sessionEnd` to joiners and releases per-session
                // resources. On a natural completion the task has
                // already returned and the cancel is a no-op.
                serveTask.cancel()
            }
        }
    }
}
