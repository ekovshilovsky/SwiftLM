import Foundation

/// Detects TurboQuant metadata in model files and routes to the appropriate loader.
/// Standard (non-TQ) models pass through to the existing loading path unchanged.
///
/// Detection is performed via pure Foundation JSON parsing of config.json,
/// which means no C library dependency is required for model identification.
struct TurboQuantModelLoader {

    /// Check if a model directory contains TurboQuant-compressed weights.
    /// Reads config.json and inspects the quantization_config for
    /// "quantization_method": "turboquant".
    static func isTurboQuantModel(at path: String) -> Bool {
        let configURL = URL(fileURLWithPath: path)
            .appendingPathComponent("config.json")

        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let quantConfig = json["quantization_config"] as? [String: Any],
              let method = quantConfig["quantization_method"] as? String
        else {
            return false
        }

        return method == "turboquant"
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
    /// Returns an error message if the model's TQ format version is
    /// unsupported, or nil if the model is compatible.
    static func validateMetadata(at path: String) -> String? {
        let configURL = URL(fileURLWithPath: path)
            .appendingPathComponent("config.json")

        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let quantConfig = json["quantization_config"] as? [String: Any]
        else {
            return "Unable to read quantization_config from config.json"
        }

        guard let version = quantConfig["tq_version"] as? String else {
            return "Missing tq_version in quantization_config"
        }

        guard version == "1" else {
            return "Unsupported TurboQuant format version '\(version)' (expected '1')"
        }

        return nil
    }

    /// Read model architecture parameters from config.json for memory budget estimation.
    /// Returns nil if the config cannot be parsed or required fields are missing.
    static func readModelConfig(at path: String) -> ModelParams? {
        let configURL = URL(fileURLWithPath: path)
            .appendingPathComponent("config.json")

        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        // Extract architecture parameters used for memory budget calculation
        guard let numLayers = json["num_hidden_layers"] as? Int,
              let numHeads = json["num_key_value_heads"] as? Int ?? json["num_attention_heads"] as? Int,
              let headDim = json["head_dim"] as? Int ?? {
                  // Derive head_dim from hidden_size / num_attention_heads if not explicit
                  guard let hiddenSize = json["hidden_size"] as? Int,
                        let attnHeads = json["num_attention_heads"] as? Int
                  else { return nil }
                  return hiddenSize / attnHeads
              }()
        else {
            return nil
        }

        let quantConfig = json["quantization_config"] as? [String: Any]
        let bitsPerWeight = quantConfig?["bits"] as? Int ?? 4

        // Estimate total parameter count from model size on disk if available,
        // otherwise fall back to a rough calculation from hidden dimensions
        let hiddenSize = json["hidden_size"] as? Int ?? (numHeads * headDim)
        let intermediateSize = json["intermediate_size"] as? Int ?? (hiddenSize * 4)
        let vocabSize = json["vocab_size"] as? Int ?? 32000
        let estimatedParams = UInt64(numLayers) * UInt64(12 * hiddenSize * hiddenSize + 8 * hiddenSize * intermediateSize) / UInt64(hiddenSize)
            + UInt64(vocabSize * hiddenSize)

        return ModelParams(
            numLayers: numLayers,
            numHeads: numHeads,
            headDim: headDim,
            bitsPerWeight: bitsPerWeight,
            estimatedParameterCount: estimatedParams
        )
    }

    /// Subset of model architecture parameters needed for memory budgeting.
    struct ModelParams {
        let numLayers: Int
        let numHeads: Int
        let headDim: Int
        let bitsPerWeight: Int
        let estimatedParameterCount: UInt64
    }
}
