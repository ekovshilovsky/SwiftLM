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
        .package(path: "./mlx-swift"),
        // Apple's LLM library built on MLX Swift (SharpAI fork — with GPU/CPU layer partitioning)
        .package(path: "./mlx-swift-lm"),
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
        // ── Vendored Argon2 reference implementation ─────────────────
        // Source: github.com/P-H-C/phc-winner-argon2 @ f57e61e19229
        // (2021-06-25, CC0/Apache-2.0 dual-licensed). Vendored directly so
        // cluster authentication does not depend on any third-party wrapper.
        // We include the portable reference round function (ref.c) rather
        // than opt.c; Blake2 headers + blake2b.c live under blake2/.
        .target(
            name: "CArgon2",
            path: "Sources/CArgon2",
            exclude: ["LICENSE"],
            sources: [
                "argon2.c",
                "core.c",
                "ref.c",
                "thread.c",
                "encoding.c",
                "blake2/blake2b.c",
            ],
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("."),
                .headerSearchPath("blake2"),
                // Silence unused-parameter warnings from the upstream code;
                // we do not want to patch the vendored source.
                .unsafeFlags(["-Wno-unused-parameter", "-Wno-unused-function"]),
            ]
        ),
        // ── TurboQuant library (shared between SwiftLM and tests) ────────
        // Isolated library target so unit tests can import these types
        // without @testable on the SwiftLM executable, which SPM does not
        // support. No external dependencies: pure Foundation + conditional
        // TurboQuantC import for bridge availability.
        .target(
            name: "TurboQuantKit",
            dependencies: [
                .product(name: "TurboQuantC", package: "turboquant-mlx-core"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                // KVCache / KVCacheSimple protocol surface used by the
                // cache-aware forward paths in TurboQuantSingleRankModel
                // and DistributedQwenModel for incremental decode.
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                "CArgon2",
            ],
            path: "Sources/SwiftLM/TurboQuant"
        ),
        // ── CLI HTTP server (macOS only) ──────────────────────────────
        .executableTarget(
            name: "SwiftLM",
            dependencies: [
                "TurboQuantKit",
                "MLXInferenceCore",
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
        // ── STFT Audio Profiling Testing Script (macOS only) ───────────
        .executableTarget(
            name: "SwiftLMTestSTFT",
            dependencies: [
                "MLXInferenceCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/SwiftLMTestSTFT",
            exclude: ["ground_truth.py"]
        ),

        // ── macOS GUI App (SwiftBuddy) ──────────────────────────────
        .executableTarget(
            name: "SwiftBuddy",
            dependencies: [
                "MLXInferenceCore",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "SwiftSoup", package: "SwiftSoup"),
            ],
            path: "SwiftBuddy/SwiftBuddy",
            exclude: [
                "Assets.xcassets",
                "SwiftBuddy.entitlements",
                "Personas/Lumina.json"
            ]
        ),
        // ── Shared inference library for SwiftLM Chat (iOS + macOS) ──
        .target(
            name: "MLXInferenceCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
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
        // budget calculations, cluster authentication and handshake, and
        // distributed coordinator nil-safety behavior.
        //
        // The `KeychainClusterKeyStore` tests substitute a recording
        // `SecItemClient` fake so they run under plain `swift test`
        // without the keychain-access-groups entitlement (which is an
        // Apple-restricted entitlement and cannot be carried by an
        // ad-hoc-signed test binary — only real Developer-ID-signed
        // hosts can). See tests/SwiftLMTests/TurboQuant/README.md for
        // the recipe to verify the live path on a signed dev machine.
        .testTarget(
            name: "SwiftLMTests",
            dependencies: ["TurboQuantKit"],
            path: "tests/SwiftLMTests",
            sources: [
                "TurboQuant/TestFixtureLocator.swift",
                "TurboQuant/TurboQuantBridgeTests.swift",
                "TurboQuant/TurboQuantModelLoaderTests.swift",
                "TurboQuant/TurboQuantShardedLinearTests.swift",
                "TurboQuant/DistributedCoordinatorTests.swift",
                "TurboQuant/ClusterPlanBuilderTests.swift",
                "TurboQuant/MemoryCalculatorTests.swift",
                "TurboQuant/BonjourDiscoveryTests.swift",
                "TurboQuant/BonjourServiceTests.swift",
                "TurboQuant/ClusterAuthTests.swift",
                "TurboQuant/ClusterHandshakeTests.swift",
                "TurboQuant/ClusterManagerTests.swift",
                "TurboQuant/ClusterKeyStoreTests.swift",
                "TurboQuant/FileClusterKeyStoreTests.swift",
                "TurboQuant/InMemoryClusterKeyStoreTests.swift",
                "TurboQuant/TestPipe.swift",
                "TurboQuant/DistributedCLIOptionsTests.swift",
                "TurboQuant/TopologyReporterTests.swift",
                "TurboQuant/Distributed/ShardAwareSafetensorsReaderTests.swift",
                "TurboQuant/Distributed/ShardMetadataTests.swift",
                "TurboQuant/Distributed/ShardOffsetCalculatorTests.swift",
                "TurboQuant/Distributed/TurboQuantAllToShardedLinearTests.swift",
                "TurboQuant/Distributed/TurboQuantShardedLinearEndToEndTests.swift",
                "TurboQuant/Distributed/TurboQuantShardedToAllLinearTests.swift",
                "TurboQuant/Distributed/ReplicatedEmbeddingTests.swift",
                "TurboQuant/Distributed/DistributedQwenModelTests.swift",
                "TurboQuant/Distributed/DistributedQwenForwardPassTests.swift",
                "TurboQuant/Distributed/DecodeEquivalenceTests.swift",
                "TurboQuant/TurboQuantSingleRankModelTests.swift",
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
