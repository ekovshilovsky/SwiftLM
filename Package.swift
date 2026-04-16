// swift-tools-version: 5.9
import PackageDescription

// Path to the CMake-built turboquant-mlx-core shared library.
// The C++ core (libturboquant_mlx.dylib) is built via CMake in the sibling
// repo; SPM imports the header-only TurboQuantC module for compile-time
// availability and links the pre-built dylib for runtime symbol resolution.
let turboquantBuildDir = "\(Context.packageDirectory)/../turboquant-mlx-core/build"

let package = Package(
    name: "SwiftLM",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MLXInferenceCore", targets: ["MLXInferenceCore"]),
        .executable(name: "SwiftLM", targets: ["SwiftLM"]),
        .executable(name: "SwiftBuddy", targets: ["SwiftBuddy"])
    ],
    dependencies: [
        // TurboQuant C API headers and module map (sibling repo, built via CMake)
        .package(path: "../turboquant-mlx-core"),
        // Local Apple MLX Swift fork for C++ extensions
        .package(url: "https://github.com/SharpAI/mlx-swift.git", branch: "main"),
        // Apple's LLM library built on MLX Swift (SharpAI fork — with GPU/CPU layer partitioning)
        .package(url: "https://github.com/SharpAI/mlx-swift-lm.git", branch: "main"),
        // HuggingFace tokenizers + model download
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.2.0")),
        // Lightweight HTTP server (Apple-backed Swift server project)
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.0.0"),
        // Async argument parser (for CLI flags: --model, --port)
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        // SwiftSoup for HTML parsing
        .package(url: "https://github.com/scinfu/SwiftSoup.git", from: "2.7.0"),
    ],
    targets: [
        // ── TurboQuant library (shared between SwiftLM and tests) ────────
        // Isolated library target so unit tests can import these types
        // without @testable on the SwiftLM executable, which SPM does not
        // support. No external dependencies: pure Foundation + conditional
        // TurboQuantC import for bridge availability.
        .target(
            name: "TurboQuantKit",
            dependencies: [
                .product(name: "TurboQuantC", package: "turboquant-mlx-core"),
            ],
            path: "Sources/SwiftLM/TurboQuant"
        ),
        // ── CLI HTTP server (macOS only) ──────────────────────────────
        .executableTarget(
            name: "SwiftLM",
            dependencies: [
                "TurboQuantKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/SwiftLM",
            exclude: ["TurboQuant"],
            linkerSettings: [
                .unsafeFlags([
                    "-L\(turboquantBuildDir)",
                    "-lturboquant_mlx",
                    "-Xlinker", "-rpath", "-Xlinker", turboquantBuildDir,
                ]),
            ]
        ),
        // ── macOS GUI App (SwiftBuddy) ──────────────────────────────
        .executableTarget(
            name: "SwiftBuddy",
            dependencies: [
                "MLXInferenceCore",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "SwiftSoup", package: "SwiftSoup"),
            ],
            path: "SwiftBuddy/SwiftBuddy"
        ),
        // ── Shared inference library for SwiftLM Chat (iOS + macOS) ──
        .target(
            name: "MLXInferenceCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXInferenceCore",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        // ── Automated Test Harness ──────────────────────────────────
        .testTarget(
            name: "SwiftBuddyTests",
            dependencies: ["SwiftBuddy", "MLXInferenceCore"]
        ),
        // ── TurboQuant Unit Tests ────────────────────────────────────
        // Depends on TurboQuantKit so types can be imported without @testable
        // on the SwiftLM executable. Tests cover: model detection, metadata
        // validation, bridge fallback paths (no TurboQuantC linked), memory
        // budget calculations, and distributed coordinator nil-safety behavior.
        .testTarget(
            name: "SwiftLMTests",
            dependencies: ["TurboQuantKit"],
            path: "tests/SwiftLMTests",
            sources: [
                "TurboQuant/TurboQuantBridgeTests.swift",
                "TurboQuant/TurboQuantModelLoaderTests.swift",
                "TurboQuant/DistributedCoordinatorTests.swift",
                "TurboQuant/MemoryCalculatorTests.swift",
                "TurboQuant/Integration/TurboQuantServingTests.swift",
                "TurboQuant/Integration/UpstreamRegressionTests.swift",
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-L\(turboquantBuildDir)",
                    "-lturboquant_mlx",
                    "-Xlinker", "-rpath", "-Xlinker", turboquantBuildDir,
                ]),
            ]
        )
    ]
)
