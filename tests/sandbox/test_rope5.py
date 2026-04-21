import mlx.core as mx
import numpy as np

freqs_np = (1.0 / (10000 ** (np.arange(0, 16, 2)/16.0))).astype(np.float32)

x_np = np.arange(16).reshape(1, 1, 1, 16).astype(np.float32)
x_mx = mx.array(x_np)

out_base = mx.fast.rope(x_mx, 16, traditional=False, base=10000.0, scale=1.0, offset=1)[0, 0, 0]
out_freqs = mx.fast.rope(x_mx, 16, traditional=False, base=None, scale=1.0, offset=1, freqs=mx.array(freqs_np))[0, 0, 0]

print("Base :", out_base)
print("Freqs :", out_freqs)
