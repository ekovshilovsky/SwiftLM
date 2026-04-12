# SwiftLM TurboQuant Fork — Agent Context

## Overview

Fork of SharpAI/SwiftLM extended with TurboQuant weight decompression
and multi-Mac distributed inference via JACCL/RDMA.

## Branch Strategy

- `main` — syncs with upstream `SharpAI/SwiftLM`
- `turboquant-integration` — all TQ work goes here

## Build Commands

```bash
swift build -c release
swift test
```

## What We Add

All new code in `Sources/SwiftLM/TurboQuant/`:
- `TurboQuantBridge.swift` — Swift wrapper around libturboquant_mlx C API
- `TurboQuantModelLoader.swift` — Detect TQ metadata, route to TQ loader
- `DistributedCoordinator.swift` — Multi-Mac setup via hostfile
- `MemoryCalculator.swift` — Memory budget validation at startup

## Upstream Touch Points (minimize these)

| File | Change |
|---|---|
| `Package.swift` | Add TurboQuantC dependency |
| `Sources/SwiftLM/Server.swift` | Add CLI flags, memory check at startup |
| `Sources/MLXInferenceCore/InferenceEngine.swift` | TQ model detection branch |

## Rules

- Never modify upstream code outside the defined touch points
- All TQ-specific logic in `Sources/SwiftLM/TurboQuant/`
- Keep upstream regression tests passing at all times
- Goal: eventual PR back to SharpAI/SwiftLM

## Cleanroom Policy

Same as turboquant-mlx-core:
- NEVER reference arozanov/turboquant-mlx or TheTom/turboquant_plus code
- Safe MIT references listed in turboquant-mlx-core CLAUDE.md

## Testing

- `Tests/SwiftLMTests/TurboQuant/` — TQ-specific unit tests
- `Tests/SwiftLMTests/TurboQuant/Integration/UpstreamRegressionTests.swift` — critical for PR
