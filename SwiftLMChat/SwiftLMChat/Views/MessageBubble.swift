// MessageBubble.swift — Chat message bubble + live streaming bubble
import SwiftUI

// MARK: — Shared adaptive colors

private extension Color {
    /// Assistant bubble background — works in both light and dark mode.
    /// Light mode: warm near-white. Dark mode: elevated dark fill.
    static var assistantBubble: Color {
        #if os(macOS)
        return Color(NSColor.controlBackgroundColor)
        #else
        return Color(UIColor.secondarySystemBackground)
        #endif
    }

    /// Subtle inner tint for thinking disclosure group.
    static var thinkingBubble: Color {
        #if os(macOS)
        return Color(NSColor.windowBackgroundColor)
        #else
        return Color(UIColor.tertiarySystemBackground)
        #endif
    }
}

// MARK: — Static Message Bubble

struct MessageBubble: View {
    let message: ChatMessage

    var isUser: Bool { message.role == .user }

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if isUser { Spacer(minLength: 60) }

            if !isUser {
                // Avatar
                Circle()
                    .fill(LinearGradient(
                        colors: [.blue, .purple],
                        startPoint: .topLeading, endPoint: .bottomTrailing
                    ))
                    .frame(width: 28, height: 28)
                    .overlay(
                        Image(systemName: "cpu")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.white)
                    )
            }

            VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
                Text(message.content)
                    .font(.system(.body, design: .default))
                    .textSelection(.enabled)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(isUser ? Color.accentColor : Color.assistantBubble)
                    .foregroundStyle(isUser ? .white : .primary)
                    .clipShape(BubbleShape(isUser: isUser))
                    .shadow(color: .black.opacity(0.06), radius: 2, x: 0, y: 1)

                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            if !isUser { Spacer(minLength: 60) }
        }
    }
}

// MARK: — Live Streaming Bubble

struct StreamingBubble: View {
    let text: String
    let thinkingText: String?
    @State private var cursorVisible = true

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            Circle()
                .fill(LinearGradient(
                    colors: [.blue, .purple],
                    startPoint: .topLeading, endPoint: .bottomTrailing
                ))
                .frame(width: 28, height: 28)
                .overlay(
                    Image(systemName: "cpu")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.white)
                )

            VStack(alignment: .leading, spacing: 6) {
                // Thinking section
                if let thinking = thinkingText, !thinking.isEmpty {
                    DisclosureGroup {
                        Text(thinking)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.thinkingBubble, in: RoundedRectangle(cornerRadius: 8))
                    } label: {
                        Label("Thinking…", systemImage: "brain")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.secondary)
                    }
                }

                // Response text
                if !text.isEmpty {
                    HStack(alignment: .bottom, spacing: 2) {
                        Text(text)
                            .font(.system(.body, design: .default))
                            .foregroundStyle(.primary)
                            .textSelection(.enabled)
                        // Blinking cursor
                        RoundedRectangle(cornerRadius: 1)
                            .frame(width: 2, height: 16)
                            .foregroundStyle(.blue)
                            .opacity(cursorVisible ? 1 : 0)
                            .animation(.easeInOut(duration: 0.5).repeatForever(), value: cursorVisible)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.assistantBubble)
                    .clipShape(BubbleShape(isUser: false))
                    .shadow(color: .black.opacity(0.06), radius: 2, x: 0, y: 1)
                } else {
                    TypingIndicator()
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(Color.assistantBubble)
                        .clipShape(BubbleShape(isUser: false))
                }
            }

            Spacer(minLength: 60)
        }
        .onAppear { cursorVisible = false }
    }
}

// MARK: — Typing Indicator

struct TypingIndicator: View {
    @State private var phase = 0

    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<3) { i in
                Circle()
                    .frame(width: 7, height: 7)
                    .foregroundStyle(.secondary)
                    .scaleEffect(phase == i ? 1.4 : 0.8)
                    .animation(
                        .easeInOut(duration: 0.45)
                            .repeatForever()
                            .delay(Double(i) * 0.15),
                        value: phase
                    )
            }
        }
        .onAppear { withAnimation { phase = 1 } }
    }
}

// MARK: — Bubble Shape

struct BubbleShape: Shape {
    let isUser: Bool
    let radius: CGFloat = 18

    func path(in rect: CGRect) -> Path {
        let tl: CGFloat = isUser ? radius : 4
        let tr: CGFloat = isUser ? 4  : radius
        let bl: CGFloat = radius
        let br: CGFloat = radius
        var path = Path()
        path.move(to: CGPoint(x: rect.minX + tl, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX - tr, y: rect.minY))
        path.addQuadCurve(to: CGPoint(x: rect.maxX, y: rect.minY + tr),
                          control: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - br))
        path.addQuadCurve(to: CGPoint(x: rect.maxX - br, y: rect.maxY),
                          control: CGPoint(x: rect.maxX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX + bl, y: rect.maxY))
        path.addQuadCurve(to: CGPoint(x: rect.minX, y: rect.maxY - bl),
                          control: CGPoint(x: rect.minX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY + tl))
        path.addQuadCurve(to: CGPoint(x: rect.minX + tl, y: rect.minY),
                          control: CGPoint(x: rect.minX, y: rect.minY))
        path.closeSubpath()
        return path
    }
}
