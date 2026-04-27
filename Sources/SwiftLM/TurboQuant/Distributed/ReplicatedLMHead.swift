// Replicated language-model head for tensor-parallel inference. Each
// rank holds the full V x H output-projection table (typically tied
// to the embedding); the logits projection is a local matmul against
// the transposed table and issues no collectives. Conforms to
// `ShardedEmbeddingLayer` so model construction can swap in a
// vocab-parallel implementation later without touching call sites.

import MLX
import MLXNN

public final class ReplicatedLMHead: Module, ShardedEmbeddingLayer {
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
    ) -> ReplicatedLMHead {
        precondition(strategy == .replicated, "ReplicatedLMHead only supports .replicated")
        precondition(weight.shape == [vocabSize, hiddenSize], "weight shape mismatch")
        return ReplicatedLMHead(weight: weight, group: group)
    }

    public func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        return MLX.matmul(hidden, weight.transposed(-1, -2))
    }
}
