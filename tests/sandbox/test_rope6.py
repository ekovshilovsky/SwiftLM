import mlx.core as mx
import numpy as np

freqs_np = (1.0 / (10000 ** (np.arange(0, 16, 2)/16.0))).astype(np.float32)
freqs_np_16 = np.concatenate([freqs_np, freqs_np])

x_np = np.arange(16).reshape(1, 1, 1, 16).astype(np.float32)
x_mx = mx.array(x_np)

out_base = mx.fast.rope(x_mx, 16, traditional=False, base=10000.0, scale=1.0, offset=1)[0, 0, 0]
out_freqs_8 = mx.fast.rope(x_mx, 16, traditional=False, base=None, scale=1.0, offset=1, freqs=mx.array(freqs_np))[0, 0, 0]
# wait, if I pass freqs of shape 16, what happens?
try:
    out_freqs_16 = mx.fast.rope(x_mx, 16, traditional=False, base=None, scale=1.0, offset=1, freqs=mx.array(freqs_np_16))[0, 0, 0]
except Exception as e:
    out_freqs_16 = str(e)

print("Base      :", out_base)
print("Freqs 8   :", out_freqs_8)
print("Freqs 16  :", out_freqs_16)
