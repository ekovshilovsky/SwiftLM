// Regression tests that guard behaviors the upstream SwiftLM PR
// depends on remaining intact. If any of these fail, the fork has
// drifted from upstream in a way the maintainer would notice on
// first review.
//
// Hermetic tests (run on every `swift test` invocation):
//   - testExistingCLIFlagsStillAccepted — asserts each flag the
//     upstream README documents is still present in --help output.
//   - testStandardModelLoadingStillWorks — runs `--info` against a
//     synthesized minimal config.json (no weights required) and
//     asserts the partition plan banner renders. Validates the
//     config-parse → ModelProfiler → partition-plan path.
//
// Model-exercising tests (opt-in via SWIFTLM_TEST_MODEL=<path>):
//   - testExistingAPIEndpointsUnchanged — spawns the server against
//     a local model and asserts /health, /v1/models, and
//     /v1/chat/completions response shapes.
//   - testExistingKVCacheStillWorks — multi-turn conversation
//     through the same chat endpoint to exercise the prompt KV
//     cache construction path.
//
// The model-gated tests XCTSkip when SWIFTLM_TEST_MODEL is unset so
// the default contributor workflow stays green without requiring any
// external model download. See tests/SwiftLMTests/TurboQuant/README.md
// for recipes on running the full path locally.

import XCTest
import Foundation

final class UpstreamRegressionTests: XCTestCase {

    // MARK: - Binary + env lookup

    /// Resolve the SwiftLM executable. `swift test` does not build
    /// executable targets unless they are a dependency of the test
    /// target, so these tests skip if the binary is not sitting in
    /// either the debug or release build directory. Contributors who
    /// want full coverage run `swift build --product SwiftLM` before
    /// `swift test`.
    private static let swiftlmBinary: URL? = {
        let fm = FileManager.default

        // Walk up from the test bundle: xctest bundle lives at
        // .build/<arch>/<config>/SwiftLMPackageTests.xctest/Contents/MacOS/.
        // The SwiftLM executable for the same config is at
        // .build/<arch>/<config>/SwiftLM.
        var dir = Bundle.main.bundleURL
        for _ in 0..<6 {
            dir.deleteLastPathComponent()
            let candidate = dir.appendingPathComponent("SwiftLM")
            if fm.isExecutableFile(atPath: candidate.path) {
                return candidate
            }
        }

        // Fallback: .build/release/SwiftLM relative to the package dir.
        // PWD during `swift test` is the package root.
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        for suffix in [".build/release/SwiftLM", ".build/debug/SwiftLM"] {
            let candidate = cwd.appendingPathComponent(suffix)
            if fm.isExecutableFile(atPath: candidate.path) {
                return candidate
            }
        }
        return nil
    }()

    private static var testModelPath: String? {
        ProcessInfo.processInfo.environment["SWIFTLM_TEST_MODEL"]
    }

    /// Shared port for server-spawning tests. Fixed rather than
    /// randomized so a test failure leaves an obvious lsof target.
    /// Tests within this class run serially in XCTest so there is no
    /// contention between them.
    private static let testPort = 5427

    private func skipUnlessBinaryBuilt() throws {
        guard Self.swiftlmBinary != nil else {
            throw XCTSkip(
                "SwiftLM executable not found. Run " +
                "`swift build --product SwiftLM` before `swift test` " +
                "to cover this regression.")
        }
    }

    private func skipUnlessModelAvailable() throws {
        try skipUnlessBinaryBuilt()
        guard Self.testModelPath != nil else {
            throw XCTSkip(
                "Set SWIFTLM_TEST_MODEL=<path to a local model dir> " +
                "to run this regression against a live server. See " +
                "tests/SwiftLMTests/TurboQuant/README.md.")
        }
    }

    // MARK: - Subprocess helpers

    private struct CLIResult {
        let exitCode: Int32
        let stdout: String
        let stderr: String
    }

    private func runCLI(_ args: [String], timeout: TimeInterval = 30) throws -> CLIResult {
        guard let binary = Self.swiftlmBinary else {
            XCTFail("binary missing — caller must skipUnlessBinaryBuilt() first")
            return CLIResult(exitCode: -1, stdout: "", stderr: "")
        }
        let proc = Process()
        proc.executableURL = binary
        proc.arguments = args
        let outPipe = Pipe(), errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        try proc.run()

        // Drain pipes in background so the subprocess isn't blocked
        // on a full pipe buffer when it emits a large --help or --info
        // banner.
        var stdoutData = Data(), stderrData = Data()
        let group = DispatchGroup()
        group.enter()
        DispatchQueue.global().async {
            stdoutData = outPipe.fileHandleForReading.readDataToEndOfFile()
            group.leave()
        }
        group.enter()
        DispatchQueue.global().async {
            stderrData = errPipe.fileHandleForReading.readDataToEndOfFile()
            group.leave()
        }

        let deadline = DispatchTime.now() + timeout
        let waitResult = group.wait(timeout: deadline)
        if waitResult == .timedOut {
            proc.terminate()
            XCTFail("subprocess timed out after \(timeout)s: \(args.joined(separator: " "))")
        }
        proc.waitUntilExit()

        return CLIResult(
            exitCode: proc.terminationStatus,
            stdout: String(data: stdoutData, encoding: .utf8) ?? "",
            stderr: String(data: stderrData, encoding: .utf8) ?? ""
        )
    }

    /// Launch SwiftLM as a server subprocess and return the handle.
    /// Caller must call `.terminate()` in a defer block.
    private func launchServer(model: String, port: Int) throws -> Process {
        guard let binary = Self.swiftlmBinary else {
            throw XCTSkip("binary not built")
        }
        let proc = Process()
        proc.executableURL = binary
        proc.arguments = ["--model", model, "--port", "\(port)", "--max-tokens", "32"]
        proc.standardOutput = Pipe()
        proc.standardError = Pipe()
        try proc.run()
        return proc
    }

    private func waitForHealth(port: Int, timeout: TimeInterval = 180) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        var attempts = 0
        while Date() < deadline {
            attempts += 1
            if let health = try? await getJSON("http://127.0.0.1:\(port)/health"),
               health["status"] as? String == "ok" {
                return
            }
            try await Task.sleep(nanoseconds: 500_000_000)
        }
        XCTFail("server on port \(port) did not become healthy within \(timeout)s (\(attempts) probes)")
    }

    // MARK: - HTTP helpers

    private func getJSON(_ url: String) async throws -> [String: Any] {
        let (data, response) = try await URLSession.shared.data(from: URL(string: url)!)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw NSError(domain: "UpstreamRegressionTests", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "HTTP \(response)"])
        }
        return try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    }

    private func postJSON(_ url: String, body: [String: Any]) async throws -> [String: Any] {
        var request = URLRequest(url: URL(string: url)!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let snippet = String(data: data.prefix(200), encoding: .utf8) ?? ""
            throw NSError(domain: "UpstreamRegressionTests", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "HTTP \(response): \(snippet)"])
        }
        return try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    }

    // MARK: - Hermetic regressions

    func testExistingCLIFlagsStillAccepted() throws {
        try skipUnlessBinaryBuilt()

        let result = try runCLI(["--help"])
        XCTAssertEqual(result.exitCode, 0,
                       "SwiftLM --help must exit 0; stderr:\n\(result.stderr)")

        // Flags the upstream README documents. Removing or renaming
        // any of these breaks an existing contract the maintainer
        // would catch on first read of the PR.
        let expectedFlags = [
            "--model", "--port", "--host", "--max-tokens",
            "--ctx-size", "--temp", "--top-p",
            "--parallel", "--thinking", "--vision",
            "--mem-limit", "--api-key", "--info",
            "--gpu-layers", "--cors",
        ]
        for flag in expectedFlags {
            XCTAssertTrue(
                result.stdout.contains(flag),
                "upstream flag \(flag) missing from --help output"
            )
        }
    }

    func testStandardModelLoadingStillWorks() throws {
        try skipUnlessBinaryBuilt()

        // Synthesize a minimal Qwen2 config that the upstream
        // ModelProfiler can plan against. No weights needed — --info
        // exits before any safetensors read.
        let tmp = try Self.makeTempConfigDir(configJSON: """
        {
          "architectures": ["Qwen2ForCausalLM"],
          "hidden_size": 256,
          "intermediate_size": 512,
          "num_attention_heads": 4,
          "num_hidden_layers": 2,
          "num_key_value_heads": 2,
          "vocab_size": 1000,
          "max_position_embeddings": 2048,
          "model_type": "qwen2",
          "torch_dtype": "float16",
          "tie_word_embeddings": true
        }
        """)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let result = try runCLI(["--model", tmp.path, "--info"], timeout: 30)

        XCTAssertEqual(result.exitCode, 0,
                       "--info must exit 0 against a valid config; stderr:\n\(result.stderr)")
        XCTAssertTrue(result.stdout.contains("Partition Plan"),
                      "--info output must contain the Partition Plan section")
        XCTAssertTrue(
            result.stdout.contains("FULL GPU") ||
            result.stdout.contains("SPLIT"),
            "--info must classify a partition strategy"
        )
        XCTAssertTrue(result.stdout.contains("GPU layers"),
                      "--info must report GPU layer assignment")
    }

    // MARK: - Model-exercising regressions

    func testExistingAPIEndpointsUnchanged() async throws {
        try skipUnlessModelAvailable()
        guard let model = Self.testModelPath else { return }

        let port = Self.testPort
        let server = try launchServer(model: model, port: port)
        defer { server.terminate(); server.waitUntilExit() }

        try await waitForHealth(port: port)

        // /health
        let health = try await getJSON("http://127.0.0.1:\(port)/health")
        XCTAssertEqual(health["status"] as? String, "ok")
        XCTAssertNotNil(health["model"], "/health must report the loaded model")
        XCTAssertNotNil(health["partition"], "/health must report the partition plan")

        // /v1/models
        let models = try await getJSON("http://127.0.0.1:\(port)/v1/models")
        XCTAssertEqual(models["object"] as? String, "list",
                       "/v1/models must return OpenAI-compatible list shape")
        let data = models["data"] as? [[String: Any]] ?? []
        XCTAssertGreaterThan(data.count, 0, "/v1/models must list at least one model")
        XCTAssertNotNil(data.first?["id"])
        XCTAssertEqual(data.first?["object"] as? String, "model")

        // /v1/chat/completions (non-streaming)
        let chat = try await postJSON(
            "http://127.0.0.1:\(port)/v1/chat/completions",
            body: [
                "model": "x",
                "messages": [["role": "user", "content": "Reply: ok"]],
                "max_tokens": 8,
                "temperature": 0,
            ]
        )
        XCTAssertEqual(chat["object"] as? String, "chat.completion")
        let choices = chat["choices"] as? [[String: Any]] ?? []
        XCTAssertGreaterThan(choices.count, 0)
        let message = choices.first?["message"] as? [String: Any]
        XCTAssertEqual(message?["role"] as? String, "assistant")
        let content = message?["content"] as? String ?? ""
        XCTAssertFalse(content.isEmpty, "assistant content must not be empty")

        let usage = chat["usage"] as? [String: Any] ?? [:]
        XCTAssertGreaterThan(usage["prompt_tokens"] as? Int ?? 0, 0)
        XCTAssertGreaterThan(usage["completion_tokens"] as? Int ?? 0, 0)
    }

    func testExistingKVCacheStillWorks() async throws {
        try skipUnlessModelAvailable()
        guard let model = Self.testModelPath else { return }

        let port = Self.testPort
        let server = try launchServer(model: model, port: port)
        defer { server.terminate(); server.waitUntilExit() }

        try await waitForHealth(port: port)

        // Multi-message conversation exercises the prompt KV cache
        // construction: the server must materialize cache state for
        // several prior turns before generating the next response.
        // If the KV cache path were regressed the server would crash,
        // hang, or produce empty content for the second turn.
        let chat = try await postJSON(
            "http://127.0.0.1:\(port)/v1/chat/completions",
            body: [
                "model": "x",
                "messages": [
                    ["role": "user",      "content": "Remember this: BANANA"],
                    ["role": "assistant", "content": "Got it."],
                    ["role": "user",      "content": "Say the word you just heard."],
                ],
                "max_tokens": 16,
                "temperature": 0,
            ]
        )
        XCTAssertEqual(chat["object"] as? String, "chat.completion")
        let choices = chat["choices"] as? [[String: Any]] ?? []
        XCTAssertGreaterThan(choices.count, 0)
        let message = choices.first?["message"] as? [String: Any]
        XCTAssertEqual(message?["role"] as? String, "assistant")
        let content = message?["content"] as? String ?? ""
        XCTAssertFalse(content.isEmpty,
                       "multi-turn assistant response must not be empty")

        // The prompt token count must reflect all three prior
        // messages having been processed — a regressed cache would
        // either fail to accumulate or reject the prompt before
        // reaching this assertion.
        let usage = chat["usage"] as? [String: Any] ?? [:]
        let promptTokens = usage["prompt_tokens"] as? Int ?? 0
        XCTAssertGreaterThan(promptTokens, 10,
                             "prompt_tokens must reflect the full three-turn prompt")
    }

    // MARK: - Fixture helpers

    private static func makeTempConfigDir(configJSON: String) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("swiftlm-regression-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try configJSON.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )
        return dir
    }
}
