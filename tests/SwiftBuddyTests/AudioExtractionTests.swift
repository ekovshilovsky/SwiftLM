import XCTest
import MLXInferenceCore
import AVFoundation

final class AudioExtractionTests: XCTestCase {

    // Feature 2: Base64 WAV data URI extraction from API content
    func testAudio_Base64WAVExtraction() {
        // Dummy base64 string padded to multiple of 4
        let base64String = "UklGRuQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0Yc=="
        let audioPart = ChatCompletionRequest.ContentPart(
            type: "input_audio",
            inputAudio: ChatCompletionRequest.InputAudioContent(data: base64String, format: "wav")
        )
        let message = ChatCompletionRequest.Message(
            role: "user",
            content: .parts([audioPart])
        )
        
        let audioData = message.extractAudio()
        XCTAssertEqual(audioData.count, 1)
        
        if let data = audioData.first {
            XCTAssertEqual(data, Data(base64Encoded: base64String))
        } else {
            XCTFail("Expected valid data extraction")
        }
    }

    // Feature 3: WAV header parsing: extract sample rate, channels, bit depth
    func testAudio_WAVHeaderParsing() throws {
        let base64String = "UklGRuQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0Yc=="
        let data = Data(base64Encoded: base64String)!
        
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        
        // AVFoundation parses WAV headers easily
        let audioFile = try AVFoundation.AVAudioFile(forReading: url)
        let format = audioFile.fileFormat
        
        XCTAssertEqual(format.sampleRate, 8000.0)
        XCTAssertEqual(format.channelCount, 1)
        XCTAssertEqual(format.commonFormat, .pcmFormatInt16)
        
        // Ensure data is readable
        XCTAssertEqual(audioFile.length, 0) // No actual data chunks appended yet
    }
}
