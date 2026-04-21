import mlx.core as mx
import numpy as np

freqs_np = (1.0 / (10000 ** (np.arange(0, 16, 2)/16.0))).astype(np.float32)

x_np = np.arange(16).reshape(1, 1, 1, 16).astype(np.float32)
x_mx = mx.array(x_np)

out_base_t = mx.fast.rope(x_mx, 16, traditional=True, base=10000.0, scale=1.0, offset=1)[0, 0, 0]
out_freqs_8_f = mx.fast.rope(x_mx, 16, traditional=False, base=None, scale=1.0, offset=1, freqs=mx.array(freqs_np))[0, 0, 0]
out_freqs_8_t = mx.fast.rope(x_mx, 16, traditional=True, base=None, scale=1.0, offset=1, freqs=mx.array(freqs_np))[0, 0, 0]

print("Base T      :", out_base_t)
print("Freqs 8 F   :", out_freqs_8_f)
print("Freqs 8 T   :", out_freqs_8_t)
