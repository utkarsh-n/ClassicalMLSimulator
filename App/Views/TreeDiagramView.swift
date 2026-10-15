import SwiftUI
import MLCore

/// A readable opening of the actual fitted tree; deeper nodes stay in the model.
struct TreeDiagramView: View {
    let nodes: [TreeDisplayNode]

    private var uniqueNodes: [TreeDisplayNode] {
        var seen = Set<Int>()
        return nodes.filter { seen.insert($0.id).inserted }
    }

    private var levels: [[TreeDisplayNode]] {
        let unique = uniqueNodes
        guard let firstDepth = unique.map(\.depth).min() else { return [] }
        return (firstDepth..<(firstDepth + 3)).map { depth in
            unique.filter { $0.depth == depth }.sorted { $0.id < $1.id }
        }.filter { !$0.isEmpty }
    }

    var body: some View {
        let rows = levels
        let visible = rows.flatMap { $0 }
        let hiddenCount = max(0, uniqueNodes.count - visible.count)
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("SPLIT TRACE", systemImage: "point.topleft.down.to.point.bottomright.curvepath")
                    .font(.system(.caption2, design: .monospaced, weight: .semibold))
                    .foregroundStyle(MLTheme.muted)
                Spacer()
                Text("\(visible.count) NODES")
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(MLTheme.muted)
            }
            if rows.isEmpty {
                Text("The first split appears as the tree grows.")
                    .font(.subheadline)
                    .foregroundStyle(MLTheme.muted)
                    .frame(maxWidth: .infinity, minHeight: 72)
            } else {
                GeometryReader { geometry in
                    let positions = positions(for: rows, size: geometry.size)
                    ZStack(alignment: .topLeading) {
                        Canvas { context, _ in
                            for node in visible {
                                guard let parent = node.parentID,
                                      let from = positions[parent], let to = positions[node.id] else { continue }
                                var link = Path()
                                link.move(to: CGPoint(x: from.x, y: from.y + 22))
                                link.addCurve(to: CGPoint(x: to.x, y: to.y - 22),
                                              control1: CGPoint(x: from.x, y: (from.y + to.y) / 2),
                                              control2: CGPoint(x: to.x, y: (from.y + to.y) / 2))
                                context.stroke(link, with: .color(MLTheme.muted.opacity(0.35)), style: StrokeStyle(lineWidth: 1, lineCap: .round))
                            }
                        }
                        .accessibilityHidden(true)
                        ForEach(visible) { node in
                            if let position = positions[node.id], let row = rows.first(where: { $0.contains(where: { $0.id == node.id }) }) {
                                nodeView(node, width: min(116, max(48, geometry.size.width / CGFloat(row.count) - 6)), hasChildren: uniqueNodes.contains(where: { $0.parentID == node.id }))
                                    .position(position)
                            }
                        }
                    }
                }
                .frame(height: CGFloat(rows.count) * 77 - 15)
            }
            if hiddenCount > 0 {
                Text("\(hiddenCount) deeper nodes continue below this view. Predictions use the full tree.")
                    .font(.caption)
                    .foregroundStyle(MLTheme.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .background(MLTheme.canvas, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(0.075), lineWidth: 0.7)
        }
    }

    private func nodeView(_ node: TreeDisplayNode, width: CGFloat, hasChildren: Bool) -> some View {
        let color = node.prediction < 0 ? MLTheme.muted : MLTheme.classColor(node.prediction)
        return VStack(spacing: 3) {
            Text(hasChildren ? "SPLIT" : "LEAF")
                .font(.system(size: 8, weight: .semibold, design: .monospaced))
                .tracking(0.7)
                .foregroundStyle(hasChildren ? MLTheme.muted : color)
            Text(displayLabel(node))
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(MLTheme.ink)
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .minimumScaleFactor(0.85)
                .frame(maxWidth: .infinity)
        }
        .padding(.horizontal, 4)
        .frame(width: width, height: 44)
        .background(LinearGradient(colors: [MLTheme.panel, MLTheme.canvas], startPoint: .top, endPoint: .bottom), in: RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(hasChildren ? .white.opacity(0.18) : color.opacity(0.6), lineWidth: 0.8)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityIdentifier("treeNode-\(node.id)")
        .accessibilityLabel("Depth \(node.depth). \(displayLabel(node)). \(hasChildren ? "Split" : "Leaf"). \(node.label.hasPrefix("Δscore") ? "Score correction." : "Predicts class \(node.prediction == 0 ? "A" : "B").")")
    }

    private func displayLabel(_ node: TreeDisplayNode) -> String {
        node.label.replacingOccurrences(of: "Class 0", with: "Class A")
            .replacingOccurrences(of: "Class 1", with: "Class B")
            .replacingOccurrences(of: "x ≤", with: "x₁ ≤")
            .replacingOccurrences(of: "y ≤", with: "x₂ ≤")
    }

    private func positions(for rows: [[TreeDisplayNode]], size: CGSize) -> [Int: CGPoint] {
        var positions: [Int: CGPoint] = [:]
        for (depthIndex, row) in rows.enumerated() {
            for (index, node) in row.enumerated() {
                positions[node.id] = CGPoint(x: size.width * (CGFloat(index) + 0.5) / CGFloat(row.count),
                                             y: 24 + CGFloat(depthIndex) * 77)
            }
        }
        return positions
    }
}
