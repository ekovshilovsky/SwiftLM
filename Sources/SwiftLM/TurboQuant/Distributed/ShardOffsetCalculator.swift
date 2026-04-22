// Compute per-rank byte ranges into a safetensors file given a
// tensor's shard metadata and the (rank, worldSize) pair. The caller
// hands the resulting SliceRange to ShardAwareSafetensorsReader to
// actually read the bytes. This module is pure arithmetic over the
// ShardTensorEntry shape; no I/O.

import Foundation

public struct ShardOffsetCalculator {
    public let rank: Int
    public let worldSize: Int

    public init(rank: Int, worldSize: Int) {
        precondition(rank >= 0 && rank < worldSize, "rank out of range")
        precondition(worldSize > 0, "worldSize must be positive")
        self.rank = rank
        self.worldSize = worldSize
    }

    /// Byte range into the containing safetensors file covering this
    /// rank's slice of the tensor. For column-parallel and replicated
    /// strategies the range is contiguous on disk; for row-parallel
    /// the range encompasses the full tensor and the caller must
    /// issue stride-aware reads using rankInputDim.
    public struct SliceRange: Sendable, Equatable {
        public let offset: Int
        public let length: Int
        public let strategy: ShardStrategy
        public let rankOutputDim: Int?
        public let rankInputDim: Int?
    }

    public enum Error: Swift.Error, Equatable {
        case nonDivisibleShardSize(axisSize: Int, worldSize: Int)
        case missingShardAxis
        case unexpectedShardAxis(expected: Int, got: Int?)
    }

    public func sliceByteRange(for entry: ShardTensorEntry) throws -> SliceRange {
        switch entry.shardStrategy {
        case .replicated:
            return SliceRange(
                offset: entry.byteOffset,
                length: entry.byteLength,
                strategy: .replicated,
                rankOutputDim: nil,
                rankInputDim: nil
            )

        case .columnParallel:
            guard let axis = entry.shardAxis else {
                throw Error.missingShardAxis
            }
            guard axis == 0 else {
                throw Error.unexpectedShardAxis(expected: 0, got: axis)
            }
            let fullOut = entry.shape[0]
            guard fullOut % worldSize == 0 else {
                throw Error.nonDivisibleShardSize(axisSize: fullOut, worldSize: worldSize)
            }
            let rankOut = fullOut / worldSize
            // Rows are contiguous; per-row byte count = total / row
            // count; rank's byte count = rankOut * bytes-per-row.
            let bytesPerOutputRow = entry.byteLength / fullOut
            let rankLength = rankOut * bytesPerOutputRow
            let rankOffset = entry.byteOffset + rank * rankLength
            let rankIn = entry.shape.count > 1 ? entry.shape[1] : nil
            return SliceRange(
                offset: rankOffset,
                length: rankLength,
                strategy: .columnParallel,
                rankOutputDim: rankOut,
                rankInputDim: rankIn
            )

        case .rowParallel:
            guard let axis = entry.shardAxis else {
                throw Error.missingShardAxis
            }
            guard axis == 1 else {
                throw Error.unexpectedShardAxis(expected: 1, got: axis)
            }
            let fullIn = entry.shape[1]
            guard fullIn % worldSize == 0 else {
                throw Error.nonDivisibleShardSize(axisSize: fullIn, worldSize: worldSize)
            }
            // Row-parallel slices are non-contiguous in the file; the
            // caller issues stride-aware reads via
            // ShardAwareSafetensorsReader. Return full bounds so the
            // reader knows the data region and rankInputDim so it
            // knows how many columns this rank keeps per row.
            return SliceRange(
                offset: entry.byteOffset,
                length: entry.byteLength,
                strategy: .rowParallel,
                rankOutputDim: entry.shape[0],
                rankInputDim: fullIn / worldSize
            )

        case .expertParallel:
            // Expert-parallel tensors are one per expert and are
            // loaded whole per-expert; the caller filters by
            // expert_index against the entry's metadata. Return full
            // bounds — the loader walks the tensor list and keeps
            // only the experts this rank owns.
            return SliceRange(
                offset: entry.byteOffset,
                length: entry.byteLength,
                strategy: .expertParallel,
                rankOutputDim: nil,
                rankInputDim: nil
            )
        }
    }
}
