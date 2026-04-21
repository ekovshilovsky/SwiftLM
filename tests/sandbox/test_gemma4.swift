import Foundation
import MLX
import MLXRandom
import MLXLMCommon
import MLXInferenceCore

@main
struct TestGemma4 {
    static func main() async throws {
        MLX.GPU.set(cacheLimit: 64 * 1024 * 1024 * 1024)
        print("Loading Model...")
        let modelDir = URL(fileURLWithPath: "/Users/simba/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/81dbfa344421b8cce28ecfcda7d639fbdeab2509").resolvingSymlinksInPath()
        
        let config = try await ModelConfiguration(directory: modelDir)
        let agentContext = ModelContext(configuration: config, modelDirectory: modelDir)
        let factory = ModelFactory()
        let modelWrapper = try await factory.load(modelContext: agentContext)
        
        let prompt = "Hey! What is the capital of France?"
        print("Prompt: \(prompt)")
        let tokens = try await modelWrapper.tokenize(prompt: prompt)
        print("Tokenized: \(tokens)")
        
        let generateParams = GenerateParameters(temperature: 0.0) // greedy!
        
        print("Generating...")
        let result = try await modelWrapper.generate(
            promptTokens: tokens,
            parameters: generateParams
        ) { progress in
            switch progress {
            case .token(let t, let s):
                print(s, terminator: "")
                fflush(stdout)
                return .more
            default:
                return .more
            }
        }
        
        print("\n\nTokens:", result.tokens)
    }
}
