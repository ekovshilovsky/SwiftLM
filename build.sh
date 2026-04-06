#!/bin/bash
set -e

echo "=> Initializing submodules..."
git submodule update --init --recursive

echo "=> Building SwiftLM (release)..."
swift build -c release

# --- Copy the pre-built default.metallib next to the binary ---
# NOTE: default.metallib is a PRE-BUILT artifact tracked in the mlx-swift
# submodule via `git add -f`. It CANNOT be compiled locally because the MLX
# Metal kernel sources (bf16_math.h) conflict with newer macOS Metal SDK
# versions. The metallib must be version-matched to the mlx-swift C++ code
# that was compiled into the SwiftLM binary. Do NOT substitute it with the
# Python mlx-metal pip package — that causes GPU kernel ABI corruption.

echo "=> Copying default.metallib..."
METALLIB_SRC="LocalPackages/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/default.metallib"
METALLIB_DEST=".build/arm64-apple-macosx/release/"

if [ -f "$METALLIB_SRC" ]; then
    mkdir -p "$METALLIB_DEST"
    cp "$METALLIB_SRC" "$METALLIB_DEST"
    echo "✅ Copied default.metallib to $METALLIB_DEST"
else
    echo "⚠️  default.metallib not found at $METALLIB_SRC"
    echo "   This file must be tracked in git. Run:"
    echo "     git add -f $METALLIB_SRC && git commit -m 'Track default.metallib'"
    exit 1
fi

echo "=> Build complete!"
