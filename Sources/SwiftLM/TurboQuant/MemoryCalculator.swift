import Foundation

/// Calculates memory budget for a given model + context configuration.
/// Used at startup to validate the requested configuration fits in memory.
struct MemoryCalculator {

    struct Budget {
        let weightMemoryBytes: UInt64
        let kvCacheMemoryBytes: UInt64
        let decodeWindowMemoryBytes: UInt64
        let overheadBytes: UInt64
        let totalBytes: UInt64
        let availableBytes: UInt64

        var fits: Bool { totalBytes <= availableBytes }
        var utilizationPercent: Double {
            Double(totalBytes) / Double(availableBytes) * 100.0
        }

        /// Formatted budget breakdown for display.
        var description: String {
            let fmt = { (label: String, bytes: UInt64) -> String in
                String(format: "  %-40s %6.1f GB", label, Double(bytes) / 1_073_741_824.0)
            }
            let lines = [
                fmt("Model weights:", weightMemoryBytes),
                fmt("KV cache (compressed):", kvCacheMemoryBytes),
                fmt("Decode window (fp16):", decodeWindowMemoryBytes),
                fmt("Activations + overhead:", overheadBytes),
                String(repeating: "-", count: 52),
                fmt("Total estimated:", totalBytes),
                fmt("Available:", availableBytes),
                "",
                fits ? "Status: FITS (\(String(format: "%.0f", 100.0 - utilizationPercent))% headroom)"
                     : "Status: DOES NOT FIT (need \(String(format: "%.1f", Double(totalBytes - availableBytes) / 1_073_741_824.0)) GB more)"
            ]
            return lines.joined(separator: "\n")
        }
    }

    /// Calculate memory budget for the given configuration.
    static func calculate(
        modelParameterCount: UInt64,
        bitsPerWeight: Int,
        contextLength: Int,
        numLayers: Int,
        numHeads: Int,
        headDim: Int,
        kvBits: Int,
        decodeWindowTokens: Int
    ) -> Budget {
        let weightBytes = modelParameterCount * UInt64(bitsPerWeight) / 8

        // KV cache: context * layers * 2 (K+V) * heads * dim * bits/8
        let kvBytes = UInt64(contextLength) * UInt64(numLayers) * 2
            * UInt64(numHeads) * UInt64(headDim) * UInt64(kvBits) / 8

        // Decode window: tokens * layers * 2 * heads * dim * 2 bytes (fp16)
        let windowBytes = UInt64(decodeWindowTokens) * UInt64(numLayers) * 2
            * UInt64(numHeads) * UInt64(headDim) * 2

        let overhead: UInt64 = 5 * 1_073_741_824 // 5 GB estimated

        let total = weightBytes + kvBytes + windowBytes + overhead

        let available = ProcessInfo.processInfo.physicalMemory

        return Budget(
            weightMemoryBytes: weightBytes,
            kvCacheMemoryBytes: kvBytes,
            decodeWindowMemoryBytes: windowBytes,
            overheadBytes: overhead,
            totalBytes: total,
            availableBytes: available
        )
    }
}
