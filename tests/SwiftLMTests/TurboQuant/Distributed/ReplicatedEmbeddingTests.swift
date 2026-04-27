import XCTest
import MLX
@testable import TurboQuantKit

final class ReplicatedEmbeddingTests: XCTestCase {

    func testEmbeddingLookupMatchesDirectIndexing() {
        let V = 128
        let H = 32
        let table = MLXRandom.normal([V, H])
        let tokenIds = MLXArray([5, 17, 42])

        let group = DistributedGroup()
        let layer = ReplicatedEmbedding.make(
            vocabSize: V, hiddenSize: H, weight: table,
            strategy: .replicated, group: group
        )
        let out = layer.callAsFunction(tokenIds)  // (3, H)

        let expected = MLX.stacked([
            table[5], table[17], table[42]
        ], axis: 0)
        let diff = (out - expected).abs().max().item(Float.self)
        XCTAssertLessThan(diff, 1e-5)
    }

    func testLMHeadMatchesMatmul() {
        let V = 64
        let H = 16
        let batch = 2
        let table = MLXRandom.normal([V, H])
        let hidden = MLXRandom.normal([batch, H])

        let group = DistributedGroup()
        let head = ReplicatedLMHead.make(
            vocabSize: V, hiddenSize: H, weight: table,
            strategy: .replicated, group: group
        )
        let logits = head.callAsFunction(hidden)  // (batch, V)
        let expected = MLX.matmul(hidden, table.transposed(1, 0))
        let diff = (logits - expected).abs().max().item(Float.self)
        XCTAssertLessThan(diff, 1e-4)
    }
}
