# TurboQuant Integration Architecture

## Overview

This fork extends SwiftLM with TurboQuant weight compression and multi-Mac
distributed inference while preserving all upstream functionality.

## Integration Design

### Model Loading Path

```
Load safetensors
  -> Read metadata
  -> Check for "quantization_method": "turboquant"
    -> YES: TurboQuantModelLoader -> libturboquant_mlx C API -> TurboQuantLinear layers
    -> NO:  Existing QuantizedLinear path (unchanged)
```

### C API Bridge

Swift calls into `libturboquant_mlx.dylib` via the C API defined in
`turboquant_c.h`. The bridge is in `TurboQuantBridge.swift`.

### Distributed Architecture

```
SwiftLM binary
  -> Parse --distributed --hostfile flags
  -> TQDistributedCoordinator.init()
  -> Shard model across nodes
  -> Each node loads its shard
  -> HTTP server runs on coordinator node only
  -> Inference: activations flow between nodes via JACCL/Ring
```

### New CLI Flags

| Flag | Default | Purpose |
|---|---|---|
| `--distributed` | false | Enable multi-node inference |
| `--hostfile <path>` | none | Cluster topology JSON |
| `--backend <jaccl\|ring\|auto>` | auto | Distributed backend |
| `--shard-strategy <pipeline\|tensor\|auto>` | auto | Parallelism type |
| `--max-context <n>` | model default | Maximum context length |
| `--decode-window <n>` | 131072 | FP16 decode window tokens |
| `--kv-bits <n>` | 3 | KV cache quantization bits |
| `--chunk-size <n>` | 512 | Prefill chunk size |

### Memory Budget Check

At startup, before loading any weights:
1. Calculate total memory required (weights + KV cache + decode buffer + overhead)
2. Compare against available unified memory
3. If exceeds: refuse to start with clear error message showing the budget breakdown
4. If tight (>90%): warn but allow
