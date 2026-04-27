// Marker protocol for embedding-layer types that participate in the
// tensor-parallel layout. Conforming types expose a uniform `make`
// factory that takes the full vocabulary table plus a sharding
// strategy, returning a layer whose internal storage matches the
// strategy. The current implementation ships only the `.replicated` strategy, fulfilled
// by `ReplicatedEmbedding` (token-id lookup) and `ReplicatedLMHead`
// (logits projection); both keep the full V x H table on every rank
// and require no collectives.
//
// The protocol is shaped so a future vocab-parallel implementation
// can slot in as a peer conforming type — model-construction code
// only knows the protocol surface and the strategy enum, so adding
// new strategies does not touch downstream wiring.

import MLX
import MLXNN

public enum EmbeddingShardStrategy: Sendable {
    case replicated
    // case vocabParallel  // future optimisation
}

public protocol ShardedEmbeddingLayer: Module, UnaryLayer {
    static func make(
        vocabSize: Int,
        hiddenSize: Int,
        weight: MLXArray,
        strategy: EmbeddingShardStrategy,
        group: DistributedGroup
    ) -> Self
}
