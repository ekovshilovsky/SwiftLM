import XCTest
import Foundation
import TurboQuantKit

final class TurboQuantModelLoaderTests: XCTestCase {

    // MARK: - Helpers

    /// Creates a temporary directory containing a config.json with the given JSON payload.
    /// The caller is responsible for removing the directory after the test.
    private func makeTempModelDir(config: [String: Any]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tqtest-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: config, options: [])
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    // MARK: - isTurboQuantModel

    func testNonTQModelReturnsFalse() throws {
        // A config.json that lacks a quantization_config stanza must not be
        // identified as a TurboQuant model — standard models must be unaffected.
        let dir = try makeTempModelDir(config: [
            "model_type": "qwen2",
            "num_hidden_layers": 28
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertFalse(TurboQuantModelLoader.isTurboQuantModel(at: dir.path))
    }

    func testTQModelDetected() throws {
        // A config.json advertising quantization_method = turboquant must be
        // positively identified so the TQ loader is dispatched at startup.
        let dir = try makeTempModelDir(config: [
            "model_type": "qwen2",
            "quantization_config": [
                "quantization_method": "turboquant",
                "tq_version": "1",
                "bits": 4
            ]
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertTrue(TurboQuantModelLoader.isTurboQuantModel(at: dir.path))
    }

    func testWrongQuantMethodReturnsFalse() throws {
        // A config.json with a different quantization method (e.g. AWQ) must
        // not be routed through the TurboQuant loader path.
        let dir = try makeTempModelDir(config: [
            "model_type": "qwen2",
            "quantization_config": [
                "quantization_method": "awq",
                "bits": 4
            ]
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertFalse(TurboQuantModelLoader.isTurboQuantModel(at: dir.path))
    }

    // MARK: - validateMetadata

    func testValidateMetadataReturnsNilForValid() throws {
        // A well-formed TQ config with tq_version "1" must produce no validation
        // error, allowing startup to proceed.
        let dir = try makeTempModelDir(config: [
            "model_type": "qwen2",
            "quantization_config": [
                "quantization_method": "turboquant",
                "tq_version": "1",
                "bits": 4
            ]
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let error = TurboQuantModelLoader.validateMetadata(at: dir.path)
        XCTAssertNil(error, "Expected no validation error for a valid TQ config, got: \(error ?? "")")
    }

    func testValidateMetadataRejectsWrongVersion() throws {
        // A tq_version value other than "1" is unrecognized by this build of SwiftLM
        // and must produce a non-nil error string so startup aborts cleanly.
        let dir = try makeTempModelDir(config: [
            "model_type": "qwen2",
            "quantization_config": [
                "quantization_method": "turboquant",
                "tq_version": "99",
                "bits": 4
            ]
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let error = TurboQuantModelLoader.validateMetadata(at: dir.path)
        XCTAssertNotNil(error, "Expected a validation error for tq_version 99")
    }

    func testValidateMetadataRejectsNonexistentPath() {
        // A path with no config.json must produce a non-nil error rather than
        // returning nil (no error), which would incorrectly indicate success.
        let error = TurboQuantModelLoader.validateMetadata(at: "/nonexistent/model")
        XCTAssertNotNil(error)
    }

    // MARK: - readModelConfig

    func testReadModelConfigExtractsParameters() throws {
        // The memory budget calculator depends on correctly extracting
        // architecture parameters from config.json. Verify that the extraction
        // handles the standard Qwen2 config format used by all TQ models.
        let dir = try makeTempModelDir(config: [
            "model_type": "qwen2",
            "hidden_size": 2048,
            "num_hidden_layers": 36,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "intermediate_size": 11008,
            "vocab_size": 151936,
            "quantization_config": [
                "quantization_method": "turboquant",
                "tq_version": "1",
                "bits": 4,
                "residual_bits": 4
            ]
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let params = TurboQuantModelLoader.readModelConfig(at: dir.path)
        XCTAssertNotNil(params, "Should extract model config from a valid config.json")
        XCTAssertEqual(params?.numLayers, 36)
        XCTAssertEqual(params?.numHeads, 2)
        XCTAssertEqual(params?.headDim, 128)  // 2048 / 16 = 128
        XCTAssertEqual(params?.bitsPerWeight, 4)
    }

    func testReadModelConfigReturnsNilForMissingFile() {
        let params = TurboQuantModelLoader.readModelConfig(at: "/nonexistent/model")
        XCTAssertNil(params)
    }
}
