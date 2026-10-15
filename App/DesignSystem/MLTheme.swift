import SwiftUI

enum MLTheme {
    static let canvas = Color(red: 0.025, green: 0.042, blue: 0.058)
    static let panel = Color(red: 0.045, green: 0.070, blue: 0.087)
    static let cyan = Color(red: 0.34, green: 0.93, blue: 0.96)
    static let amber = Color(red: 1, green: 0.74, blue: 0.37)
    static let violet = Color(red: 0.76, green: 0.57, blue: 1)
    static let ink = Color(red: 0.91, green: 0.95, blue: 0.97)
    static let muted = Color(red: 0.60, green: 0.69, blue: 0.73)
    static func classColor(_ index: Int) -> Color {
        switch index {
        case 0: cyan
        case 1: amber
        case 2: violet
        case 3: Color(red: 0.96, green: 0.36, blue: 0.84)
        case 4: Color(red: 0.45, green: 0.9, blue: 0.58)
        case 5: Color(red: 0.45, green: 0.55, blue: 1)
        default:
            index < 0 ? muted : Color(hue: (Double(index) * 0.61803398875).truncatingRemainder(dividingBy: 1), saturation: 0.55, brightness: 0.95)
        }
    }
}

struct LabBackground: View {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    var body: some View {
        ZStack {
            MLTheme.canvas
            if !reduceTransparency {
                RadialGradient(colors: [MLTheme.cyan.opacity(0.07), .clear], center: .topTrailing, startRadius: 0, endRadius: 650)
            }
            Canvas { context, size in
                var path = Path()
                for x in stride(from: 0.0, to: size.width, by: 28) {
                    path.move(to: CGPoint(x: x, y: 0)); path.addLine(to: CGPoint(x: x, y: size.height))
                }
                for y in stride(from: 0.0, to: size.height, by: 28) {
                    path.move(to: CGPoint(x: 0, y: y)); path.addLine(to: CGPoint(x: size.width, y: y))
                }
                context.stroke(path, with: .color(.white.opacity(0.025)), lineWidth: 0.5)
            }
        }.ignoresSafeArea().allowsHitTesting(false).accessibilityHidden(true)
    }
}

struct LabPanel<Content: View>: View {
    var title: String?
    var minimumHeight: CGFloat?
    @ViewBuilder var content: Content
    @Environment(\.colorSchemeContrast) private var contrast
    init(_ title: String? = nil, minimumHeight: CGFloat? = nil, @ViewBuilder content: () -> Content) {
        self.title = title; self.minimumHeight = minimumHeight; self.content = content()
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let title { Engraving(title) }
            content
        }
        .padding(16)
        .frame(maxWidth: .infinity, minHeight: minimumHeight, alignment: .topLeading)
        .background(MLTheme.panel, in: .rect(cornerRadius: 22))
        .overlay { RoundedRectangle(cornerRadius: 22).stroke(.white.opacity(contrast == .increased ? 0.3 : 0.09), lineWidth: 1).allowsHitTesting(false) }
    }
}

struct Engraving: View {
    var text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text.uppercased()).font(.system(.caption2, design: .monospaced, weight: .medium))
            .tracking(1.5).foregroundStyle(MLTheme.muted)
    }
}

struct LabIconButton: View {
    var symbol: String; var label: String; var accent: Color = MLTheme.cyan
    var action: () -> Void
    var body: some View {
        Button(action: action) {
            Image(systemName: symbol).font(.system(size: 17, weight: .semibold))
                .foregroundStyle(accent).frame(width: 48, height: 48)
                .glassEffect(.regular.tint(accent.opacity(0.08)).interactive(), in: .circle)
        }.buttonStyle(.plain).accessibilityLabel(label)
    }
}

struct StatusBadge: View {
    var text: String; var color: Color = MLTheme.cyan
    var body: some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 5, height: 5)
            Text(text).font(.system(.caption2, design: .monospaced, weight: .medium))
        }.foregroundStyle(color).padding(.horizontal, 9).padding(.vertical, 6)
            .background(color.opacity(0.08), in: .capsule)
            .overlay { Capsule().stroke(color.opacity(0.2), lineWidth: 0.6) }
    }
}
