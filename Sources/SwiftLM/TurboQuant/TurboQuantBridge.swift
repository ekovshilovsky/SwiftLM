import Foundation

#if canImport(TurboQuantC)
import TurboQuantC
#endif

/// Swift bridge to libturboquant_mlx C API.
/// Wraps opaque C handles with Swift memory management and type safety.
/// When the TurboQuantC SPM package is not linked, all factory methods
/// return nil, allowing the rest of SwiftLM to compile unconditionally.
final class TurboQuantBridge {

    /// Load a TurboQuant-compressed model from the specified path.
    /// Returns nil if the path is invalid or the model cannot be loaded.
    static func loadModel(path: String) -> TurboQuantModel? {
        return TurboQuantModel(path: path)
    }

    /// Create a TQ-compressed KV cache with the given parameters.
    static func createKVCache(
        numLayers: Int,
        numHeads: Int,
        headDim: Int,
        kvBits: Int = 3,
        maxContext: Int = 1_048_576,
        decodeWindow: Int = 131_072
    ) -> TurboQuantKVCache? {
        return TurboQuantKVCache(
            numLayers: numLayers,
            numHeads: numHeads,
            headDim: headDim,
            kvBits: kvBits,
            maxContext: maxContext,
            decodeWindow: decodeWindow
        )
    }

    /// Query the linked TurboQuant library version string.
    /// Returns nil when TurboQuantC is not available.
    static func libraryVersion() -> String? {
        #if canImport(TurboQuantC)
        guard let cStr = tq_version() else { return nil }
        return String(cString: cStr)
        #else
        return nil
        #endif
    }
}

/// Swift wrapper for a loaded TurboQuant model.
/// Owns the underlying C handle and releases it on deinitialization.
final class TurboQuantModel {
    #if canImport(TurboQuantC)
    private let handle: tq_model_t

    init?(path: String) {
        guard let h = tq_model_load(path) else { return nil }
        self.handle = h
    }

    deinit {
        tq_model_free(handle)
    }

    /// Run a forward pass with the given input array.
    /// Caller is responsible for interpreting the returned opaque pointer
    /// (typically an MLX array) and freeing it via `TurboQuantBridge.freeArray`.
    func forward(input: UnsafeRawPointer) -> UnsafeMutableRawPointer? {
        return tq_model_forward(handle, input)
    }
    #else
    init?(path: String) { return nil }
    #endif
}

/// Swift wrapper for a TQ-compressed KV cache.
/// Manages the lifecycle of the underlying C handle and provides
/// type-safe access to cache operations.
final class TurboQuantKVCache {
    #if canImport(TurboQuantC)
    private let handle: tq_kv_cache_t

    init?(
        numLayers: Int,
        numHeads: Int,
        headDim: Int,
        kvBits: Int,
        maxContext: Int,
        decodeWindow: Int
    ) {
        guard let h = tq_kv_cache_create(
            Int32(numLayers),
            Int32(numHeads),
            Int32(headDim),
            Int32(kvBits),
            Int32(maxContext),
            Int32(decodeWindow)
        ) else { return nil }
        self.handle = h
    }

    deinit {
        tq_kv_cache_free(handle)
    }

    /// Append key-value entries for a given transformer layer.
    func append(layer: Int, keys: UnsafeRawPointer, values: UnsafeRawPointer) {
        tq_kv_cache_append(handle, Int32(layer), keys, values)
    }

    /// Retrieve fp16 keys from the decode window for the specified range.
    func getKeysFP16(layer: Int, start: Int, end: Int) -> UnsafeMutableRawPointer? {
        return tq_kv_cache_get_keys_fp16(handle, Int32(layer), Int32(start), Int32(end))
    }

    /// Retrieve fp16 values from the decode window for the specified range.
    func getValuesFP16(layer: Int, start: Int, end: Int) -> UnsafeMutableRawPointer? {
        return tq_kv_cache_get_values_fp16(handle, Int32(layer), Int32(start), Int32(end))
    }

    /// Run fused TQ attention over the compressed region of the cache.
    func attention(
        layer: Int,
        queries: UnsafeRawPointer,
        compressedStart: Int,
        compressedEnd: Int
    ) -> UnsafeMutableRawPointer? {
        return tq_kv_cache_attention(
            handle, Int32(layer), queries,
            Int32(compressedStart), Int32(compressedEnd)
        )
    }

    /// Current sequence length stored in the cache.
    var sequenceLength: Int {
        return Int(tq_kv_cache_seq_length(handle))
    }

    /// Free an array returned by forward, getKeys, getValues, or attention.
    static func freeArray(_ ptr: UnsafeMutableRawPointer) {
        tq_array_free(ptr)
    }
    #else
    init?(
        numLayers: Int,
        numHeads: Int,
        headDim: Int,
        kvBits: Int,
        maxContext: Int,
        decodeWindow: Int
    ) {
        return nil
    }
    #endif
}
