import mlx.core as mx
import glob
import os

model_path = os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/")
snapshots = glob.glob(model_path + "*")
if snapshots:
    snap_path = snapshots[0]
    for sf in glob.glob(snap_path + "/*.safetensors"):
        weights = mx.load(sf)
        for k, v in weights.items():
            if "lm_head." in k or "embed_tokens" in k:
                print(f"{k}: {v.shape}")
