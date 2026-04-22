import XCTest
@testable import TurboQuantKit

final class ShardAwareSafetensorsReaderTests: XCTestCase {

    /// Write a synthetic 8x4 byte matrix (row-major) into a temp file
    /// and verify that a column-parallel contiguous read returns the
    /// first half rows intact (rows 0-3, 16 bytes).
    func testColumnParallelContiguousReadMatchesOriginalSlice() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("shardreader-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: tmp) }

        // 8 rows of 4 bytes each, row r filled with value r.
        var bytes: [UInt8] = []
        for r in 0..<8 {
            for _ in 0..<4 { bytes.append(UInt8(r)) }
        }
        try Data(bytes).write(to: tmp)

        let reader = ShardAwareSafetensorsReader(fileURL: tmp)
        let range = ShardOffsetCalculator.SliceRange(
            offset: 0, length: 16, strategy: .columnParallel,
            rankOutputDim: 4, rankInputDim: 4
        )
        let data = try reader.readContiguousRange(range)
        XCTAssertEqual(data.count, 16)
        // First 4 bytes should be all 0 (row 0), last 4 should be all 3 (row 3).
        XCTAssertEqual(Array(data.prefix(4)), [0, 0, 0, 0])
        XCTAssertEqual(Array(data.suffix(4)), [3, 3, 3, 3])
    }

    /// A range that extends past the end of the file must throw, not
    /// silently return fewer bytes than requested.
    func testReadPastEndOfFileFails() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("shardreader-\(UUID().uuidString).bin")
        try Data(repeating: 0, count: 10).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let reader = ShardAwareSafetensorsReader(fileURL: tmp)
        let range = ShardOffsetCalculator.SliceRange(
            offset: 0, length: 100, strategy: .columnParallel,
            rankOutputDim: nil, rankInputDim: nil
        )
        XCTAssertThrowsError(try reader.readContiguousRange(range))
    }

    /// Column-parallel read of the second half (rows 4-7) must
    /// start at offset 16 and return 16 bytes.
    func testColumnParallelSecondRankSlice() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("shardreader-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: tmp) }
        var bytes: [UInt8] = []
        for r in 0..<8 {
            for _ in 0..<4 { bytes.append(UInt8(r)) }
        }
        try Data(bytes).write(to: tmp)

        let reader = ShardAwareSafetensorsReader(fileURL: tmp)
        let range = ShardOffsetCalculator.SliceRange(
            offset: 16, length: 16, strategy: .columnParallel,
            rankOutputDim: 4, rankInputDim: 4
        )
        let data = try reader.readContiguousRange(range)
        XCTAssertEqual(data.count, 16)
        XCTAssertEqual(Array(data.prefix(4)), [4, 4, 4, 4])
        XCTAssertEqual(Array(data.suffix(4)), [7, 7, 7, 7])
    }

    /// Row-parallel stride-aware read: given a (4 rows, 8 columns)
    /// uint8 matrix, rank 0 of 2 (columns 0-3) and rank 1 of 2 (columns
    /// 4-7) should together reconstruct the original.
    func testRowParallelStrideAwareRead() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("shardreader-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: tmp) }

        // 4 rows x 8 cols, col c in row r has value (r*8 + c).
        var bytes: [UInt8] = []
        for r in 0..<4 {
            for c in 0..<8 {
                bytes.append(UInt8(r * 8 + c))
            }
        }
        try Data(bytes).write(to: tmp)

        let reader = ShardAwareSafetensorsReader(fileURL: tmp)
        // Rank 0 reads columns 0-3 of each row.
        let rank0 = try reader.readRowParallelSlice(
            fullTensorOffset: 0,
            fullTensorLength: 32,
            rowCount: 4,
            fullColumnsPerRow: 8,
            bytesPerElement: 1,
            rankColumnStart: 0,
            rankColumnCount: 4
        )
        XCTAssertEqual(rank0.count, 16)
        // Row 0 cols 0-3: [0,1,2,3].
        XCTAssertEqual(Array(rank0[0..<4]), [0, 1, 2, 3])
        // Row 3 cols 0-3: [24,25,26,27].
        XCTAssertEqual(Array(rank0[12..<16]), [24, 25, 26, 27])

        // Rank 1 reads columns 4-7 of each row.
        let rank1 = try reader.readRowParallelSlice(
            fullTensorOffset: 0,
            fullTensorLength: 32,
            rowCount: 4,
            fullColumnsPerRow: 8,
            bytesPerElement: 1,
            rankColumnStart: 4,
            rankColumnCount: 4
        )
        XCTAssertEqual(rank1.count, 16)
        // Row 0 cols 4-7: [4,5,6,7].
        XCTAssertEqual(Array(rank1[0..<4]), [4, 5, 6, 7])
        // Row 3 cols 4-7: [28,29,30,31].
        XCTAssertEqual(Array(rank1[12..<16]), [28, 29, 30, 31])
    }
}
