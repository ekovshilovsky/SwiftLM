import Foundation
import MLX

public class ALMTypeRegistry {
    public static let shared = ALMTypeRegistry()
    
    private var creators: [String: () -> Any] = [:]
    
    private init() {
        // Feature 8: Register Whisper
        register(creator: { WhisperModelCreator() }, for: "whisper")
    }
    
    public func register(creator: @escaping () -> (Any), for key: String) {
        creators[key] = creator
    }
    
    public func creator(for key: String) -> (() -> Any)? {
        return creators[key]
    }
}

public struct WhisperModelCreator {
    public init() {}
}
