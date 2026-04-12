import Foundation

/// Detects TurboQuant metadata in model files and routes to the appropriate loader.
/// Standard (non-TQ) models pass through to the existing loading path unchanged.
struct TurboQuantModelLoader {

    /// Check if a model directory contains TurboQuant-compressed weights.
    /// Reads the safetensors metadata for "quantization_method": "turboquant".
    static func isTurboQuantModel(at path: String) -> Bool {
        // Phase 4: read config.json or safetensors header, check for TQ metadata
        return false
    }

    /// Load a TurboQuant model via the C API bridge.
    /// Returns nil if the model is not TQ-compressed or loading fails.
    static func load(from path: String) -> TurboQuantModel? {
        guard isTurboQuantModel(at: path) else {
            return nil
        }
        return TurboQuantBridge.loadModel(path: path)
    }

    /// Validate TQ metadata version compatibility.
    /// Returns an error message if incompatible, nil if compatible.
    static func validateMetadata(at path: String) -> String? {
        // Phase 4: check tq_version field matches "1"
        return nil
    }
}
