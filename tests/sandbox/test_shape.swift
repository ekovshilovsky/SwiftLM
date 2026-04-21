import Foundation
import MLX

let textEmbeds = MLXArray.zeros([1, 10, 4])
let imageIndices = MLXArray([2, 3, 4])
let imageFeatures = MLXArray.ones([1, 3, 4]) * 5.0

var result = textEmbeds
result[0..., imageIndices, 0...] = imageFeatures

eval(result)
print(result)
