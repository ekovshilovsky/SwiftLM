# GLM-5.1 MoE Port - WIP Workspace State

This repository (SwiftLM) is currently heavily coupled with local instances of its backend dependencies to natively support the **baa-ai/GLM-5.1-RAM-270GB-MLX** model. 

As of April 2026, the local workspace is structured in **three active W.I.P repositories** that must be maintained together on the `glm5.1` branch for the `SwiftLM` engine to boot correctly.

## 1. SwiftLM (Root)
* **Path**: `/Users/simba/SwiftLM`
* **Branch**: `glm5.1`
* **Purpose**: Inference API, terminal logging, and profiling benchmark orchestrator.
* **Status**: 
    - Forced `Package.swift` to point to the local `../mlx-swift` and `./mlx-swift-lm` paths instead of resolving GitHub remotes to allow cross-repo C++ debugging.
    - Updated `profile_runner.py` to pipe standard output for runtime overcommit monitoring.

## 2. mlx-swift-lm (Local Sub-Repository)
* **Path**: `/Users/simba/SwiftLM/mlx-swift-lm`
* **Branch**: `glm5.1`
* **Purpose**: Defines MLX graph execution (Architecture mapping & Hugging Face Safetensors index bridging).
* **Status**: 
    - Defines `GLMMoeDSA` architecture and dense MLP `SwitchGLU` layers.
    - Uses aggressive parameter pruning during `sanitize()` to gracefully drop `self_attn.indexer` keys and execute the model using standard `Multi Latent Attention` (MLA).

## 3. mlx-swift (C++ / Metal Backend Runtime)
* **Path**: `/Users/simba/SwiftLM/mlx-swift`
* **Branch**: `glm5.1`
* **Purpose**: MLX C-Bindings, kernel dispatch, and zero-copy NVMe SSD streaming.
* **Status**: 
    - Unlocked unbounded memory chunk streaming in `LoadSSDExpert`. Removed the hardcoded `[SSDStreamer] Load length exceeds Pinned Buffer capacity` condition inside `ssd_streamer.mm` to allow streaming aggregated massive tensor blocks directly into Metal-configured Unified Memory.

---
### Warning for Deployment
Before deploying or returning `SwiftLM` to production (`main`):
1. Wait for `SharpAI/mlx-swift` to successfully merge the `glm5.1` branch updates.
2. Wait for `SharpAI/mlx-swift-lm` to merge the GLM-5.1 specific parameter logic.
3. Update `Package.swift` here in the root to repoint to the latest GitHub release tags instead of local `../mlx-swift` paths. 
