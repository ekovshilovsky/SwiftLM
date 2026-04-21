import Foundation
import MLX
import MLXLLM

let modelPath = "/Users/simba/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-8bit/snapshots/18f3418f2da5426ec6e4967b4c96bdd2d0002ee4"
do {
    let weights = try MLX.load(url: URL(fileURLWithPath: modelPath + "/model-00001-of-00004.safetensors"))
    print("Keys starting with per_layer:")
    for k in weights.keys where k.contains("per_layer") {
        print(k)
    }
} catch {
    print("Error:", error)
}
