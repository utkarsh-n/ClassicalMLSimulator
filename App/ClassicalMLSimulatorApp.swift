import SwiftUI

@main
struct ClassicalMLSimulatorApp: App {
    var body: some Scene {
        WindowGroup {
            MLRootView()
                .preferredColorScheme(.dark)
                .tint(MLTheme.cyan)
        }
    }
}
