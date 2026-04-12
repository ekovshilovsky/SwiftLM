import XCTest
@testable import SwiftLM

final class TurboQuantBridgeTests: XCTestCase {

    func testLoadModelReturnsNilForInvalidPath() {
        let model = TurboQuantBridge.loadModel(path: "/nonexistent/path")
        XCTAssertNil(model)
    }

    func testCreateKVCacheReturnsInstance() {
        // Stub returns nil until C API is connected
        let cache = TurboQuantBridge.createKVCache(
            numLayers: 28, numHeads: 8, headDim: 128
        )
        // Phase 4: change to XCTAssertNotNil when implementation lands
        XCTAssertNil(cache)
    }
}
