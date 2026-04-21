import MLX

var result = MLX.zeros([1, 10, 4])
var indices = MLXArray([2, 3, 4])
var features = MLX.ones([1, 3, 4]) * 5.0

result[0..., indices, 0...] = features

eval(result)
print(result)
