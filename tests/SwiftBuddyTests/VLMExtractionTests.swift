import XCTest
import MLXInferenceCore
#if canImport(MLXVLM)
import MLXVLM
#endif

final class VLMExtractionTests: XCTestCase {

    // Feature 2: Base64 data URI image extraction from multipart content
    func testVLM_Base64ImageExtraction() {
        let base64String = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=" // 1x1 transparent PNG
        let imagePart = ChatCompletionRequest.ContentPart(
            type: "image_url",
            imageUrl: ChatCompletionRequest.ImageUrlContent(url: "data:image/png;base64,\(base64String)")
        )
        let message = ChatCompletionRequest.Message(
            role: "user",
            content: .parts([imagePart])
        )
        
        let images = message.extractImages()
        XCTAssertEqual(images.count, 1)
        
        if case let .ciImage(image) = images.first {
            XCTAssertNotNil(image)
            XCTAssertEqual(image.extent.width, 1)
            XCTAssertEqual(image.extent.height, 1)
        } else {
            XCTFail("Expected .ciImage, got \(String(describing: images.first))")
        }
    }

    // Feature 3: HTTP URL image extraction from multipart content
    func testVLM_HTTPURLImageExtraction() {
        let imagePart = ChatCompletionRequest.ContentPart(
            type: "image_url",
            imageUrl: ChatCompletionRequest.ImageUrlContent(url: "https://example.com/test.jpg")
        )
        let message = ChatCompletionRequest.Message(
            role: "user",
            content: .parts([imagePart])
        )
        
        let images = message.extractImages()
        XCTAssertEqual(images.count, 1)
        
        if case let .url(url) = images.first {
            XCTAssertEqual(url.absoluteString, "https://example.com/test.jpg")
        } else {
            XCTFail("Expected .url, got \(String(describing: images.first))")
        }
    }

    // Feature 8: Multiple images in single message are all processed
    func testVLM_MultipleImagesInMessage() {
        let base64String = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        
        let textPart = ChatCompletionRequest.ContentPart(type: "text", text: "Here are two images:")
        let imagePart1 = ChatCompletionRequest.ContentPart(
            type: "image_url",
            imageUrl: ChatCompletionRequest.ImageUrlContent(url: "data:image/png;base64,\(base64String)")
        )
        let imagePart2 = ChatCompletionRequest.ContentPart(
            type: "image_url",
            imageUrl: ChatCompletionRequest.ImageUrlContent(url: "https://example.com/test2.jpg")
        )
        
        let message = ChatCompletionRequest.Message(
            role: "user",
            content: .parts([textPart, imagePart1, imagePart2])
        )
        
        let images = message.extractImages()
        XCTAssertEqual(images.count, 2)
    }
}
