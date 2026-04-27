// Replicated embedding layer for tensor-parallel inference. Each rank
// holds the full V x H token-embedding table; token-id lookup is a
// local `MLX.take` and issues no collectives. Conforms to
// `ShardedEmbeddingLayer` so model construction can swap in a
// vocab-parallel implementation later without touching call sites.

import MLX
import MLXNN

public final class ReplicatedEmbedding: Module, ShardedEmbeddingLayer {
    public let weight: MLXArray
    public let group: DistributedGroup

    private init(weight: MLXArray, group: DistributedGroup) {
        self.weight = weight
        self.group = group
        super.init()
    }

    public static func make(
        vocabSize: Int, hiddenSize: Int, weight: MLXArray,
        strategy: EmbeddingShardStrategy, group: DistributedGroup
    ) -> ReplicatedEmbedding {
        precondition(strategy == .replicated, "ReplicatedEmbedding only supports .replicated")
        precondition(weight.shape == [vocabSize, hiddenSize], "weight shape mismatch")
        return ReplicatedEmbedding(weight: weight, group: group)
    }

    public func callAsFunction(_ tokenIds: MLXArray) -> MLXArray {
        return MLX.take(weight, tokenIds, axis: 0)
    }
}
