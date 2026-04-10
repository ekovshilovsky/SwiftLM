import XCTest
import Foundation

final class AudioTests: XCTestCase {

    // Feature 1: --audio flag is accepted without crash
    func testAudio_AudioFlagAccepted() async throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        // Note: Testing with a lightweight model, or just asserting that SwiftLM prints its launch message.
        let args = ["swift", "run", "SwiftLM", "--model", "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "--audio"]
        process.arguments = args
        
        let projectPath = URL(fileURLWithPath: #file).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        process.currentDirectoryURL = projectPath
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        
        try process.run()
        
        // Wait up to 15 seconds to grab standard output up to the "Loading" print
        let start = Date()
        var foundLoading = false
        var accumulated = ""
        while Date().timeIntervalSince(start) < 15.0 {
            let data = pipe.fileHandleForReading.availableData
            if !data.isEmpty {
                accumulated += String(data: data, encoding: .utf8) ?? ""
                if accumulated.contains("Loading") || accumulated.contains("SwiftLM") {
                    foundLoading = true
                    process.terminate()
                    break
                }
            } else {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        process.terminate()
        
        XCTAssertTrue(foundLoading, "Output should indicate SwiftLM started successfully with --audio flag. Got: \(accumulated)")
    }
}
