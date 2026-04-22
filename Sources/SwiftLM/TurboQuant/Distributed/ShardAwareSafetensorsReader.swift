// Byte-level reader over a safetensors data file. Concerned only
// with raw range extraction; higher-level tensor parsing lives in
// the shard-loading pipeline that composes this reader with the
// ShardMetadata + ShardOffsetCalculator.

import Foundation

public struct ShardAwareSafetensorsReader: Sendable {
    public let fileURL: URL

    public init(fileURL: URL) {
        self.fileURL = fileURL
    }

    public enum Error: Swift.Error, Equatable {
        case rangeExceedsFileSize(offset: Int, length: Int, fileSize: Int)
        case readFailed(String)
    }

    /// Read the contiguous byte range described by `range` and return
    /// it as Data. Used for column-parallel and replicated strategies
    /// where the rank's bytes form a single contiguous slab on disk.
    public func readContiguousRange(
        _ range: ShardOffsetCalculator.SliceRange
    ) throws -> Data {
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }

        let size = try handle.seekToEnd()
        let wantedEnd = UInt64(range.offset + range.length)
        if wantedEnd > size {
            throw Error.rangeExceedsFileSize(
                offset: range.offset,
                length: range.length,
                fileSize: Int(size)
            )
        }

        try handle.seek(toOffset: UInt64(range.offset))
        guard let data = try handle.read(upToCount: range.length),
              data.count == range.length
        else {
            throw Error.readFailed("short read")
        }
        return data
    }

    /// For row-parallel slicing, the reader strides through the file
    /// and extracts only this rank's column range per row. Takes the
    /// full tensor bounds + element size + rank's column range, returns
    /// the concatenated rank-local bytes (row-major).
    public func readRowParallelSlice(
        fullTensorOffset: Int,
        fullTensorLength: Int,
        rowCount: Int,
        fullColumnsPerRow: Int,
        bytesPerElement: Int,
        rankColumnStart: Int,
        rankColumnCount: Int
    ) throws -> Data {
        let rowBytes = fullColumnsPerRow * bytesPerElement
        let rankRowBytes = rankColumnCount * bytesPerElement
        precondition(
            rowBytes * rowCount == fullTensorLength,
            "row_bytes * rowCount must equal fullTensorLength"
        )
        precondition(
            rankColumnStart + rankColumnCount <= fullColumnsPerRow,
            "rank column range exceeds tensor width"
        )

        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }

        var out = Data(capacity: rankRowBytes * rowCount)
        for row in 0..<rowCount {
            let rowStart = fullTensorOffset
                + row * rowBytes
                + rankColumnStart * bytesPerElement
            try handle.seek(toOffset: UInt64(rowStart))
            guard let chunk = try handle.read(upToCount: rankRowBytes),
                  chunk.count == rankRowBytes
            else {
                throw Error.readFailed("short read in row \(row)")
            }
            out.append(chunk)
        }
        return out
    }
}
