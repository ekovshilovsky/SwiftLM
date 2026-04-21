import mlx.core as mx
import numpy as np

def pt_rope(x, freqs):
    inv_freq = freqs
    pos = np.array([1])
    f = np.outer(pos, inv_freq)
    emb = np.concatenate([f, f], axis=-1)
    cos = np.cos(emb)
    sin = np.sin(emb)
    
    half = x.shape[-1] // 2
    x_rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + x_rot * sin

x_np = np.arange(16).reshape(1, 1, 1, 16).astype(np.float32)
freqs_np = (1.0 / (10000 ** (np.arange(0, 16, 2)/16.0))).astype(np.float32)

x_mx = mx.array(x_np)
freqs_mx = mx.array(freqs_np)

out_pt = pt_rope(x_np, freqs_np)[0, 0, 0]

out1 = mx.fast.rope(x_mx, 16, traditional=False, base=None, scale=1.0, offset=1, freqs=freqs_mx)
out2 = mx.fast.rope(x_mx, 16, traditional=True, base=None, scale=1.0, offset=1, freqs=freqs_mx)

print("PT         :", out_pt)
print("MX F freqs :", out1[0, 0, 0])
print("MX T freqs :", out2[0, 0, 0])
