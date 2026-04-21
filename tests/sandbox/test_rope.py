import mlx.core as mx
import numpy as np

def pt_rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)

x_np = np.arange(16).reshape(1, 1, 1, 16).astype(np.float32)
x_mx = mx.array(x_np)

out_mx_trad = mx.fast.rope(x_mx, 16, traditional=True, base=10000.0, scale=1.0, offset=0)
out_mx_inter = mx.fast.rope(x_mx, 16, traditional=False, base=10000.0, scale=1.0, offset=0)

# Simulate RoPE rotation with theta=10000 on the first token (pos=0)
# wait, for pos=0, angle is 0, so cos=1, sin=0. 
# out should be identically x_mx!
# Let's use offset=1 (pos=1) so sin != 0!
out_mx_trad = mx.fast.rope(x_mx, 16, traditional=True, base=10000.0, scale=1.0, offset=1)
out_mx_inter = mx.fast.rope(x_mx, 16, traditional=False, base=10000.0, scale=1.0, offset=1)

print("NP rotate_half pattern (if sin=1, cos=0):", pt_rotate_half(x_np)[0, 0, 0])
print("MX trad=True:", out_mx_trad[0, 0, 0])
print("MX trad=False:", out_mx_inter[0, 0, 0])
