import XCTest
import TurboQuantKit

final class TurboQuantBridgeTests: XCTestCase {

    // MARK: - Library Version

    func testLibraryVersionReturnsValidString() {
        let version = TurboQuantBridge.libraryVersion()
        XCTAssertNotNil(version, "Library version should be available when TurboQuantC is linked")
        XCTAssertEqual(version, "0.1.0", "Expected library version 0.1.0")
    }

    // MARK: - Model Loading

    func testLoadModelReturnsNilForInvalidPath() {
        let model = TurboQuantBridge.loadModel(path: "/nonexistent/path")
        XCTAssertNil(model, "Loading from a nonexistent path should return nil")
    }

    func testLoadModelReturnsNilForFilePath() {
        // A file (not a directory) should also return nil
        let tempFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("tqtest-\(UUID().uuidString).txt")
        FileManager.default.createFile(atPath: tempFile.path, contents: Data())
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let model = TurboQuantBridge.loadModel(path: tempFile.path)
        XCTAssertNil(model, "Loading from a file (not directory) should return nil")
    }

    // MARK: - KV Cache

    func testCreateKVCacheReturnsInstance() {
        let cache = TurboQuantBridge.createKVCache(
            numLayers: 28, numHeads: 8, headDim: 128
        )
        XCTAssertNotNil(cache, "KV cache creation should succeed with valid parameters")
    }

    func testCreateKVCacheWithCustomBits() {
        let cache = TurboQuantBridge.createKVCache(
            numLayers: 36, numHeads: 16, headDim: 128,
            kvBits: 4, maxContext: 32768, decodeWindow: 8192
        )
        XCTAssertNotNil(cache, "KV cache with custom parameters should initialize")
    }

    // MARK: - Dequantization

    func testDequantModelFailsForInvalidPaths() {
        let success = TurboQuantBridge.dequantModel(
            sourcePath: "/nonexistent/model",
            outputPath: "/tmp/tqtest-dequant-output"
        )
        XCTAssertFalse(success, "Dequant should fail for nonexistent source path")
    }
}
