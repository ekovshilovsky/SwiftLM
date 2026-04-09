// ChatViewModel.swift — Bridges InferenceEngine actor to SwiftUI
import SwiftUI
import Combine
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var streamingText: String = ""
    @Published var thinkingText: String? = nil
    @Published var isGenerating: Bool = false
    @Published var config: GenerationConfig = .default
    @Published var systemPrompt: String = ""
    public var currentWing: String? = nil

    weak var engine: InferenceEngine?
    private var generationTask: Task<Void, Never>?

    // MARK: — Send

    func send(_ userText: String) async {
        guard let engine, !isGenerating else { return }

        let userMessage = ChatMessage.user(userText)
        messages.append(userMessage)

        isGenerating = true
        streamingText = ""
        thinkingText = nil
        
        // --- INVISIBLE RAG INJECTION ---
        var wakeUpText = ""
        var activeRagDirective = ""
        
        if let wing = currentWing, !wing.isEmpty {
            do {
                // 1. WAKE-UP HOOK: Offline Persona Context Injection (STATIC - preservers KV cache)
                let coreFacts = try MemoryPalaceService.shared.fetchRoomContents(wingName: wing, roomName: "CORE IDENTITY")
                let bgFacts = try MemoryPalaceService.shared.fetchRoomContents(wingName: wing, roomName: "BACKGROUND STORY")
                let toneFacts = try MemoryPalaceService.shared.fetchRoomContents(wingName: wing, roomName: "TALK TONE")
                
                var combinedIdentity = (coreFacts + bgFacts).joined(separator: "\n")
                if !toneFacts.isEmpty {
                    combinedIdentity += "\n\nCONVERSATIONAL TONE DIRECTIVE:\n" + toneFacts.joined(separator: "\n")
                }
                
                if !combinedIdentity.isEmpty {
                    wakeUpText = "SYSTEM PERSONA DIRECTIVE:\n\(combinedIdentity)\n\n"
                }
                
                // 2. ACTIVE RAG HOOK (DYNAMIC - injected onto the newest turn to prevent cache wipe)
                let facts = try MemoryPalaceService.shared.searchMemories(query: userText, wingName: wing)
                if !facts.isEmpty {
                    let factList = facts.map { "- [\($0.hallType)] \($0.text)" }.joined(separator: "\n")
                    activeRagDirective = "\n\n[RELEVANT MEMORY CONTEXT FOR THIS TURN]:\n\(factList)\n\nYou must strictly incorporate these facts if they are relevant here."
                }
            } catch {
                print("RAG Pre-Fetch Failed: \(error.localizedDescription)")
            }
        }

        // Apply System Persona to the VERY FIRST User prompt permanently inside the official state.
        let identityPayload = wakeUpText + systemPrompt
        if !identityPayload.isEmpty && messages.count == 1 {
            if let firstUserIdx = messages.firstIndex(where: { $0.role == .user }) {
                let originalText = messages[firstUserIdx].content
                messages[firstUserIdx].content = "SYSTEM DIRECTIVE & CONTEXT:\n\(identityPayload)\n\nUSER PROMPT:\n\(originalText)"
            }
        }

        // Apply dynamic memory strictly to the CURRENT prompt PERMANENTLY so we don't destroy MLX's historical Prefix KV cache on the next turn!
        if !activeRagDirective.isEmpty {
            if let lastUserIdx = messages.lastIndex(where: { $0.role == .user }) {
                messages[lastUserIdx].content += activeRagDirective
            }
        }
        
        var fullMessages = messages
        
        // Squash consecutive roles to prevent Jinja alternation crashes on strict models (e.g., Gemma)
        var collapsedMessages: [ChatMessage] = []
        for msg in fullMessages {
            if let last = collapsedMessages.last, last.role == msg.role {
                collapsedMessages[collapsedMessages.count - 1].content += "\n\n" + msg.content
            } else {
                collapsedMessages.append(msg)
            }
        }
        fullMessages = collapsedMessages

        generationTask = Task {
            var response = ""
            var thinking = ""
            var hasRawThinkTags = false

            for await token in engine.generate(messages: fullMessages, config: config) {
                guard !Task.isCancelled else { break }

                if token.isThinking {
                    thinking += token.text
                    thinkingText = thinking
                } else {
                    response += token.text
                    
                    // Fallback cleanup if the model outputs literal <think>...</think> tags
                    // and the tokenizer isn't setting the isThinking flag correctly.
                    if response.contains("<think>") {
                        hasRawThinkTags = true
                        
                        // Try to safely extract thinking content between the tags
                        if let startRange = response.range(of: "<think>"),
                           let endRange = response.range(of: "</think>") {
                            // Extract thinking
                            let rawThinking = String(response[startRange.upperBound..<endRange.lowerBound])
                            thinkingText = rawThinking
                            
                            // Remove the entire block from the visible response
                            let before = String(response[..<startRange.lowerBound])
                            let after = String(response[endRange.upperBound...])
                            streamingText = before + after
                        } else if let startRange = response.range(of: "<think>") {
                            // We have a start tag but no end tag yet, it's currently generating the thought
                            let rawThinking = String(response[startRange.upperBound...])
                            thinkingText = rawThinking
                            
                            // Only update streaming text with what came before
                            streamingText = String(response[..<startRange.lowerBound])
                        }
                    } else if !hasRawThinkTags {
                        // Standard flow: no raw tags seen yet, just stream normally
                        streamingText = response 
                    }
                }
            }

            // Commit completed message
            if !response.isEmpty {
                // Do a final cleanup just in case
                var finalVisible = response
                if let startRange = response.range(of: "<think>"),
                   let endRange = response.range(of: "</think>") {
                    let before = String(response[..<startRange.lowerBound])
                    let after = String(response[endRange.upperBound...])
                    finalVisible = before + after
                } else if let startRange = response.range(of: "<think>") {
                     finalVisible = String(response[..<startRange.lowerBound])
                }
                
                // Trim leading newlines that often follow thought blocks
                finalVisible = finalVisible.trimmingCharacters(in: .whitespacesAndNewlines)
                
                if !finalVisible.isEmpty {
                    messages.append(.assistant(finalVisible, thinkingContent: thinkingText))
                }
            }

            streamingText = ""
            thinkingText = nil
            isGenerating = false
        }

        await generationTask?.value
    }

    // MARK: — Controls

    func stopGeneration() {
        generationTask?.cancel()
        if !streamingText.isEmpty {
            messages.append(.assistant(streamingText, thinkingContent: thinkingText))
        }
        streamingText = ""
        thinkingText = nil
        isGenerating = false
    }

    func newConversation() {
        stopGeneration()
        messages = []
    }
}
