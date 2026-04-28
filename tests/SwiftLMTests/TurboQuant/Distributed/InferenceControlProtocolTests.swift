import XCTest
import Foundation
@testable import TurboQuantKit

/// Round-trip and shape tests for the coordinator-to-joiner control
/// channel codec. Each message variant must encode and decode back
/// to the equal value, and the on-the-wire encoding must remain
/// stable so deployed clusters do not silently break across SwiftLM
/// upgrades.
///
/// These tests run without the model fixture — pure data-shape
/// assertions over the codec.
final class InferenceControlProtocolTests: XCTestCase {

    // MARK: - SamplingParams round-trip

    func testSamplingParamsRoundTripWithAllFields() throws {
        let original = SamplingParams(
            temperature: 0.7,
            topK: 40,
            topP: 0.95,
            maxTokens: 512,
            stopTokens: [13, 100, 999]
        )
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(SamplingParams.self, from: encoded)
        XCTAssertEqual(decoded, original)
    }

    func testSamplingParamsRoundTripWithOptionalFieldsAbsent() throws {
        let original = SamplingParams(
            temperature: 0,
            maxTokens: 64
        )
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(SamplingParams.self, from: encoded)
        XCTAssertEqual(decoded, original)
        XCTAssertNil(decoded.topK)
        XCTAssertNil(decoded.topP)
        XCTAssertEqual(decoded.stopTokens, [])
    }

    // MARK: - Per-message round-trip

    func testSessionStartRoundTrip() throws {
        let sampling = SamplingParams(temperature: 1.0, topK: 50, maxTokens: 256)
        let original = InferenceControlMessage.sessionStart(
            sessionID: UUID(),
            sampling: sampling,
            seed: 0xDEADBEEFCAFE_BABE
        )
        let data = try InferenceControlCodec.encode(original)
        let decoded = try InferenceControlCodec.decode(data)
        XCTAssertEqual(decoded, original)
    }

    func testPrefillRoundTrip() throws {
        let original = InferenceControlMessage.prefill(promptTokens: [1, 2, 3, 4, 5])
        let data = try InferenceControlCodec.encode(original)
        let decoded = try InferenceControlCodec.decode(data)
        XCTAssertEqual(decoded, original)
    }

    func testDecodeRoundTrip() throws {
        let original = InferenceControlMessage.decode
        let data = try InferenceControlCodec.encode(original)
        let decoded = try InferenceControlCodec.decode(data)
        XCTAssertEqual(decoded, original)
    }

    func testInjectTokenRoundTrip() throws {
        let original = InferenceControlMessage.injectToken(sampledToken: 17)
        let data = try InferenceControlCodec.encode(original)
        let decoded = try InferenceControlCodec.decode(data)
        XCTAssertEqual(decoded, original)
    }

    func testSessionEndRoundTrip() throws {
        let original = InferenceControlMessage.sessionEnd
        let data = try InferenceControlCodec.encode(original)
        let decoded = try InferenceControlCodec.decode(data)
        XCTAssertEqual(decoded, original)
    }

    // MARK: - Codec stability

    /// The shared encoder uses `sortedKeys`, so encoding the same
    /// value twice must produce byte-identical output. A future
    /// regression where someone constructs a fresh encoder without
    /// the same options would surface here as a bytes mismatch.
    func testEncodingIsDeterministic() throws {
        let message = InferenceControlMessage.sessionStart(
            sessionID: UUID(uuidString: "12345678-1234-1234-1234-1234567890AB")!,
            sampling: SamplingParams(temperature: 0.5, topK: 20, topP: 0.9, maxTokens: 100),
            seed: 42
        )
        let bytes1 = try InferenceControlCodec.encode(message)
        let bytes2 = try InferenceControlCodec.encode(message)
        XCTAssertEqual(bytes1, bytes2,
                       "control-channel encoder must produce identical bytes " +
                       "for identical input")
    }

    /// Cross-message-type confusion test: every variant decodes to
    /// the right case after a round trip even when other variants
    /// share field names. Catches a regression where Codable
    /// synthesis collapses two cases sharing a payload field.
    func testNoCrossVariantConfusion() throws {
        let id = UUID()
        let messages: [InferenceControlMessage] = [
            .sessionStart(
                sessionID: id,
                sampling: SamplingParams(temperature: 0.0, maxTokens: 1),
                seed: 0
            ),
            .prefill(promptTokens: [42]),
            .decode,
            .injectToken(sampledToken: 7),
            .sessionEnd
        ]
        for original in messages {
            let data = try InferenceControlCodec.encode(original)
            let decoded = try InferenceControlCodec.decode(data)
            XCTAssertEqual(decoded, original,
                           "variant must round-trip without case confusion: " +
                           "original=\(original) decoded=\(decoded)")
        }
    }
}
