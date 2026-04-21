import Foundation

let json = """
{
    "audio_config": {
        "_name_or_path": "",
        "architectures": null,
        "model_type": "gemma4_audio",
        "output_proj_dims": 1536
    },
    "vision_config": {
        "_name_or_path": "",
        "architectures": null,
        "num_hidden_layers": 16,
        "hidden_size": 768
    }
}
"""

struct Gemma4VisionConfigurationProxy: Codable {
    public let hiddenLayers: Int?
    public let intermediateSize: Int?
    public let attentionHeads: Int?
    public let patchSize: Int?
    public let hiddenSize: Int?
    
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case patchSize = "patch_size"
    }
}

struct Gemma4AudioConfigurationProxy: Codable {
    public let modelType: String?
    public let hiddenSize: Int?
    public let numHiddenLayers: Int?
    public let numAttentionHeads: Int?
    public let outputProjDims: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case outputProjDims = "output_proj_dims"
    }
}

struct RootConfig: Codable {
    var visionConfig: Gemma4VisionConfigurationProxy?
    var audioConfig: Gemma4AudioConfigurationProxy?

    enum CodingKeys: String, CodingKey {
        case visionConfig = "vision_config"
        case audioConfig = "audio_config"
    }

    init(from decoder: Decoder) throws {
        let topContainer = try decoder.container(keyedBy: CodingKeys.self)
        print("topContainer OK")
        self.visionConfig = try topContainer.decodeIfPresent(Gemma4VisionConfigurationProxy.self, forKey: .visionConfig)
        print("visionConfig OK")
        self.audioConfig = try topContainer.decodeIfPresent(Gemma4AudioConfigurationProxy.self, forKey: .audioConfig)
        print("audioConfig OK")
    }
}

do {
    let decoder = JSONDecoder()
    let data = json.data(using: .utf8)!
    let config = try decoder.decode(RootConfig.self, from: data)
    print("visionConfig.hiddenSize: \(String(describing: config.visionConfig?.hiddenSize))")
    print("audioConfig.outputProjDims: \(String(describing: config.audioConfig?.outputProjDims))")
} catch {
    print("ERROR: \(error)")
}
