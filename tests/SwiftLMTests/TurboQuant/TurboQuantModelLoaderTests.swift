import XCTest
@testable import SwiftLM

final class TurboQuantModelLoaderTests: XCTestCase {

    func testNonTQModelReturnsFalse() {
        // A path without TQ metadata should not be detected as TQ
        XCTAssertFalse(TurboQuantModelLoader.isTurboQuantModel(at: "/tmp/standard-model"))
    }

    func testLoadNonTQModelReturnsNil() {
        let model = TurboQuantModelLoader.load(from: "/tmp/standard-model")
        XCTAssertNil(model)
    }

    func testValidateMetadataReturnsNilForNoFile() {
        let error = TurboQuantModelLoader.validateMetadata(at: "/nonexistent")
        // Stub returns nil (no error) for now
        XCTAssertNil(error)
    }
}
