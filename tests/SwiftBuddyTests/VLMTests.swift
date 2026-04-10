import XCTest
import Foundation

final class VLMTests: XCTestCase {
    
    // Feature 1: --vision flag loads VLM instead of LLM
    func testVLM_VisionFlagLoadsVLMFactory() async throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        let args = ["swift", "run", "SwiftLM", "--model", "mlx-community/Qwen2-VL-2B-Instruct-4bit", "--vision"]
        process.arguments = args
        
        let projectPath = URL(fileURLWithPath: #file).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        process.currentDirectoryURL = projectPath
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        
        try process.run()
        
        // Wait up to 15 seconds to grab standard output up to the "Loading" print
        let start = Date()
        var found = false
        var accumulated = ""
        while Date().timeIntervalSince(start) < 15.0 {
            let data = pipe.fileHandleForReading.availableData
            if !data.isEmpty {
                accumulated += String(data: data, encoding: .utf8) ?? ""
                if accumulated.contains("Loading VLM") {
                    found = true
                    process.terminate()
                    break
                }
            } else {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        process.terminate()
        
        XCTAssertTrue(found, "Output should indicate VLM is loading. Got: \(accumulated)")
    }
}
