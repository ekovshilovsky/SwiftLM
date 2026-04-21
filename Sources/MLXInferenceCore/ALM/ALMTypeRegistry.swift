import Foundation
import MLX

public actor ALMTypeRegistry {
    public static let shared = ALMTypeRegistry()
    
    private var creators: [String: @Sendable () -> Any] = ["whisper": { WhisperModelCreator() }]
    
    private init() {}
    
    public func register(creator: @escaping @Sendable () -> (Any), for key: String) {
        creators[key] = creator
    }
    
    public func creator(for key: String) -> (@Sendable () -> Any)? {
        return creators[key]
    }
}

public struct WhisperModelCreator {
    public init() {}
}
