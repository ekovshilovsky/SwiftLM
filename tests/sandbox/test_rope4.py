import mlx.core as mx
import numpy as np

x_np = np.arange(16).reshape(1, 1, 1, 16).astype(np.float32)
freqs_np = (1.0 / (10000 ** (np.arange(0, 16, 2)/16.0))).astype(np.float32)

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

def mx_rope_trad(x, freqs):
    # Simulate mlx fast rope with traditional=True
    half = x.shape[-1] // 2
    out = np.zeros_like(x)
    for i in range(half):
        f = freqs[i]
        c = np.cos(f * 1)
        s = np.sin(f * 1)
        out[..., i] = x[..., i] * c - x[..., i + half] * s
        out[..., i + half] = x[..., i + half] * c + x[..., i] * s
    return out

out_pt = pt_rope(x_np, freqs_np)[0, 0, 0]
out_mx = mx_rope_trad(x_np, freqs_np)[0, 0, 0]
out_mx_core = mx.fast.rope(mx.array(x_np), 16, traditional=True, base=None, scale=1.0, offset=1, freqs=mx.array(freqs_np))[0, 0, 0]

print("PT        :", out_pt)
print("MX Sim    :", out_mx)
print("MX Core   :", out_mx_core)
