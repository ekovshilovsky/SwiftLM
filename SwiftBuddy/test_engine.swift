import Foundation

class InferenceEngine {}

class GraphPalaceService {
    static let shared = GraphPalaceService()
    
    func buildRelationalGraph(wingName: String, using engine: InferenceEngine? = nil) async throws {
        guard let engine = engine else {
            print("[GraphPalace] Engine unavailable or not injected.")
            return
        }
        print("[GraphPalace] Engine is valid!")
    }
}

class RegistryService {
    static let shared = RegistryService()
    
    func downloadPersona(name: String, using engine: InferenceEngine? = nil) async {
        if let engine = engine {
             do {
                 try await GraphPalaceService.shared.buildRelationalGraph(wingName: name, using: engine)
             } catch {}
        } else {
             print("[RegistryService] Engine not injected.")
        }
    }
}

let engine = InferenceEngine()
let unownedEngine: InferenceEngine? = engine

Task {
    await RegistryService.shared.downloadPersona(name: "Test", using: unownedEngine)
    exit(0)
}

RunLoop.main.run()
