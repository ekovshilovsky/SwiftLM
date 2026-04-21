import Foundation

let path = "Sources/Server/Server.swift"
var file = try! String(contentsOfFile: path)
let target = "let promptTokens = try await tokenMemory.applyChatTemplate(messages: payload.messages)"
let replace = """
            let promptTokens = try await tokenMemory.applyChatTemplate(messages: payload.messages)
            print("🚀 RAW PROMPT TOKENS EVALUATED: \\(promptTokens)")
"""
file = file.replacingOccurrences(of: target, with: replace)
try! file.write(toFile: path, atomically: true, encoding: .utf8)
