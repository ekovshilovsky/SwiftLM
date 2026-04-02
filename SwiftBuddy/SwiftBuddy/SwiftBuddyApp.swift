// SwiftBuddyApp.swift — App entry point (iOS + macOS)
import SwiftUI

// MARK: — Appearance Store (persists dark/light/system preference)

final class AppearanceStore: ObservableObject {
    private static let key = "swiftlm.colorScheme"   // "dark" | "light" | "system"

    @Published var preference: String {
        didSet { UserDefaults.standard.set(preference, forKey: Self.key) }
    }

    init() {
        preference = UserDefaults.standard.string(forKey: Self.key) ?? "dark"
    }

    var colorScheme: ColorScheme? {
        switch preference {
        case "dark":  return .dark
        case "light": return .light
        default:      return nil
        }
    }
}

// MARK: — App

@main
struct SwiftBuddyApp: App {
    @StateObject private var engine = InferenceEngine()
    @StateObject private var appearance = AppearanceStore()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(engine)
                .environmentObject(appearance)
                .preferredColorScheme(appearance.colorScheme)
                .accentColor(SwiftBuddyTheme.accent)
                .tint(SwiftBuddyTheme.accent)
        }
        #if os(macOS)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Model") {
                Button("Choose Model…") {
                    NotificationCenter.default.post(name: .showModelPicker, object: nil)
                }.keyboardShortcut("m", modifiers: [.command, .shift])
                Button("Unload Model") {
                    engine.unload()
                }
            }
        }
        #endif
    }
}

extension Notification.Name {
    static let showModelPicker = Notification.Name("showModelPicker")
}

