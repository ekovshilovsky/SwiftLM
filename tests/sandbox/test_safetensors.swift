import Foundation
import MLX
import MLXNN

// Find all safetensors files
let fileManager = FileManager.default
let hubPath = NSString(string: "~/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots").expandingTildeInPath

guard let snapshotDirs = try? fileManager.contentsOfDirectory(atPath: hubPath), let latestSnapshot = snapshotDirs.first else {
    print("No snapshots found")
    exit(1)
}

let fullPath = URL(fileURLWithPath: hubPath).appendingPathComponent(latestSnapshot)
let files = (try? fileManager.contentsOfDirectory(atPath: fullPath.path)) ?? []
let safetensorsFiles = files.filter { $0.hasSuffix(".safetensors") }

for sf in safetensorsFiles {
    let fullSFPath = fullPath.appendingPathComponent(sf)
    let arrays = try? MLX.loadArrays(url: fullSFPath)
    if let arrays = arrays {
        for (key, _) in arrays {
            if key.contains("layer_projection") || key.contains("per_layer") {
                print(key)
            }
        }
    }
}
