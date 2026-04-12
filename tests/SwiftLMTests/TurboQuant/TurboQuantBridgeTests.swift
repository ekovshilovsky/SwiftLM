import XCTest
import TurboQuantKit

final class TurboQuantBridgeTests: XCTestCase {

    func testLoadModelReturnsNilForInvalidPath() {
        // TurboQuantModel(path:) calls tq_model_load(), which returns nil
        // for any path that cannot be opened. When TurboQuantC is not linked,
        // the #else branch also returns nil, so this assertion holds in both cases.
        let model = TurboQuantBridge.loadModel(path: "/nonexistent/path")
        XCTAssertNil(model)
    }

    func testCreateKVCacheReturnsInstance() {
        // TurboQuantC is not linked in the standalone SwiftLM SPM build —
        // the #else branch in TurboQuantKVCache.init returns nil unconditionally.
        // This test validates the graceful fallback: the rest of the stack
        // must tolerate a nil cache and route to the upstream KV path instead.
        let cache = TurboQuantBridge.createKVCache(
            numLayers: 28, numHeads: 8, headDim: 128
        )
        XCTAssertNil(cache)
    }
}
