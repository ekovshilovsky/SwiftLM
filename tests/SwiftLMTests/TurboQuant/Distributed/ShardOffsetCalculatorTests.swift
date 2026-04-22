import XCTest
@testable import TurboQuantKit

final class ShardOffsetCalculatorTests: XCTestCase {

    /// For a 2-rank column-parallel split of a (3072, 2048) tensor with
    /// a byteLength of 6_291_456, rank 0 gets the first half (rows
    /// 0-1535, 3_145_728 bytes starting at the tensor's byte_offset)
    /// and rank 1 gets the second half (rows 1536-3071, 3_145_728 bytes
    /// starting at byte_offset + 3_145_728).
    func testColumnParallelTwoRankHalving() throws {
        let entry = ShardTensorEntry(
            shape: [3072, 2048], dtype: "tq8",
            file: "m.safetensors",
            byteOffset: 1000, byteLength: 6_291_456,
            shardAxis: 0, shardStrategy: .columnParallel,
            codebookKey: nil, rotationKey: nil, expertIndex: nil
        )
        let calc0 = ShardOffsetCalculator(rank: 0, worldSize: 2)
        let r0 = try calc0.sliceByteRange(for: entry)
        XCTAssertEqual(r0.offset, 1000)
        XCTAssertEqual(r0.length, 3_145_728)
        XCTAssertEqual(r0.rankOutputDim, 1536)
        XCTAssertEqual(r0.strategy, .columnParallel)

        let calc1 = ShardOffsetCalculator(rank: 1, worldSize: 2)
        let r1 = try calc1.sliceByteRange(for: entry)
        XCTAssertEqual(r1.offset, 1000 + 3_145_728)
        XCTAssertEqual(r1.length, 3_145_728)
        XCTAssertEqual(r1.rankOutputDim, 1536)
    }

    func testReplicatedReturnsFullRange() throws {
        let entry = ShardTensorEntry(
            shape: [128], dtype: "fp16", file: "m.safetensors",
            byteOffset: 5000, byteLength: 256,
            shardAxis: nil, shardStrategy: .replicated,
            codebookKey: nil, rotationKey: nil, expertIndex: nil
        )
        let calc = ShardOffsetCalculator(rank: 0, worldSize: 2)
        let r = try calc.sliceByteRange(for: entry)
        XCTAssertEqual(r.offset, 5000)
        XCTAssertEqual(r.length, 256)
        XCTAssertNil(r.rankOutputDim)
        XCTAssertNil(r.rankInputDim)
        XCTAssertEqual(r.strategy, .replicated)
    }

    func testColumnParallelRejectsNonDivisibleOutputDim() {
        let entry = ShardTensorEntry(
            shape: [3073, 2048], dtype: "tq8", file: "m.safetensors",
            byteOffset: 0, byteLength: 1,
            shardAxis: 0, shardStrategy: .columnParallel,
            codebookKey: nil, rotationKey: nil, expertIndex: nil
        )
        let calc = ShardOffsetCalculator(rank: 0, worldSize: 2)
        XCTAssertThrowsError(try calc.sliceByteRange(for: entry))
    }

    func testRowParallelReturnsFullBoundsWithRankInputDim() throws {
        let entry = ShardTensorEntry(
            shape: [2048, 3072], dtype: "tq8", file: "m.safetensors",
            byteOffset: 0, byteLength: 6_291_456,
            shardAxis: 1, shardStrategy: .rowParallel,
            codebookKey: nil, rotationKey: nil, expertIndex: nil
        )
        // Row-parallel slices are non-contiguous in the file — we get
        // one column-range per row. sliceByteRange returns the bounds
        // of the full tensor plus the rank's input-dim size so the
        // caller can drive stride-aware reads via
        // ShardAwareSafetensorsReader.
        let calc = ShardOffsetCalculator(rank: 0, worldSize: 2)
        let r = try calc.sliceByteRange(for: entry)
        XCTAssertEqual(r.strategy, .rowParallel)
        XCTAssertEqual(r.rankInputDim, 1536)
        XCTAssertEqual(r.rankOutputDim, 2048)
        XCTAssertEqual(r.offset, 0)
        XCTAssertEqual(r.length, 6_291_456)
    }

    func testColumnParallelRank1OfFourComputesCorrectOffset() throws {
        // Four-way split of a (4096, 1024) tensor with byteLength
        // 8_388_608. Rank 2 (third quarter) should start at
        // byteLength * 2/4 = 4_194_304 past the base offset.
        let entry = ShardTensorEntry(
            shape: [4096, 1024], dtype: "tq8", file: "m.safetensors",
            byteOffset: 100, byteLength: 8_388_608,
            shardAxis: 0, shardStrategy: .columnParallel,
            codebookKey: nil, rotationKey: nil, expertIndex: nil
        )
        let calc = ShardOffsetCalculator(rank: 2, worldSize: 4)
        let r = try calc.sliceByteRange(for: entry)
        XCTAssertEqual(r.offset, 100 + 4_194_304)
        XCTAssertEqual(r.length, 2_097_152)
        XCTAssertEqual(r.rankOutputDim, 1024)
    }
}
