import SwiftUI
import MLCore

/// A projection of a recorded simulation frame. This view never trains a model.
struct MLPlotView: View {
    let algorithm: MLAlgorithm
    let frame: SimulationFrame
    let probe: MLPoint?
    let onTap: (Double, Double) -> Void

    private var isRegression: Bool {
        algorithm == .linearRegression || algorithm == .polynomialRegression
    }

    private var spokenSummary: String {
        let heldOut = frame.points.filter(\.isTest).count
        let centers: String
        switch algorithm {
        case .dbscan:
            centers = frame.centers.isEmpty ? "" : " \(frame.centers.count) core samples."
        case .pca:
            centers = frame.centers.isEmpty ? "" : " Cross marks the training mean."
        case .naiveBayes:
            centers = frame.centers.isEmpty ? "" : " \(frame.centers.count) class means."
        case .kMeans:
            centers = frame.centers.isEmpty ? "" : " \(frame.centers.count) cluster centers."
        default:
            centers = ""
        }
        let selected = probe.map { String(format: " Probe at x %.2f, y %.2f.", $0.x, $0.y) } ?? ""
        let sampleSummary = algorithm == .monteCarlo
            ? "\(frame.points.count) sampled points. No train/test split applies."
            : "\(frame.points.count) samples, \(heldOut) held out."
        return "Simulation plot. \(frame.caption). \(sampleSummary)\(centers)\(selected) Axes range from zero to one. Accessibility actions offer probe locations."
    }

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                let plot = Self.plotRect(size)
                drawFace(context: &context, plot: plot, size: size)
                var dataContext = context
                dataContext.clip(to: Path(plot))
                drawRegions(context: &dataContext, plot: plot)
                if algorithm == .monteCarlo {
                    drawMonteCarlo(context: &dataContext, plot: plot)
                }
                drawGrid(context: &dataContext, plot: plot)
                if isRegression {
                    drawResiduals(context: &dataContext, plot: plot)
                }
                drawVectors(context: &dataContext, plot: plot)
                if isRegression { drawCurve(context: &dataContext, plot: plot) }
                drawPoints(context: &dataContext, plot: plot)
                drawCenters(context: &dataContext, plot: plot)
                drawProbe(context: &dataContext, plot: plot)
                drawAxes(context: &context, plot: plot)
            }
            .contentShape(Rectangle())
            .gesture(SpatialTapGesture().onEnded { event in
                let plot = Self.plotRect(geometry.size)
                guard plot.contains(event.location) else { return }
                onTap(
                    Double((event.location.x - plot.minX) / plot.width),
                    Double(1 - (event.location.y - plot.minY) / plot.height)
                )
            })
        }
        .background {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(LinearGradient(colors: [Color(red: 0.055, green: 0.073, blue: 0.090), MLTheme.canvas], startPoint: .topLeading, endPoint: .bottomTrailing))
        }
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(LinearGradient(colors: [.white.opacity(0.20), .white.opacity(0.035), .white.opacity(0.09)], startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 1)
                .allowsHitTesting(false)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityIdentifier("simulationPlot")
        .accessibilityLabel(spokenSummary)
        .accessibilityAction(named: Text("Probe center")) { onTap(0.5, 0.5) }
        .accessibilityAction(named: Text("Probe lower left")) { onTap(0.25, 0.25) }
        .accessibilityAction(named: Text("Probe upper right")) { onTap(0.75, 0.75) }
    }

    private static func plotRect(_ size: CGSize) -> CGRect {
        let availableWidth = max(1, size.width - 51)
        let availableHeight = max(1, size.height - 53)
        let side = min(availableWidth, availableHeight)
        return CGRect(x: 35 + (availableWidth - side) / 2,
                      y: 20 + (availableHeight - side) / 2,
                      width: side, height: side)
    }

    private func mapped(_ x: Double, _ y: Double, plot: CGRect) -> CGPoint {
        CGPoint(x: plot.minX + (x.isFinite ? x : 0) * plot.width,
                y: plot.maxY - (y.isFinite ? y : 0) * plot.height)
    }

    private func drawFace(context: inout GraphicsContext, plot: CGRect, size: CGSize) {
        context.fill(Path(roundedRect: plot, cornerRadius: 2), with: .color(Color(red: 0.022, green: 0.036, blue: 0.047)))
        for x in [CGFloat(10), size.width - 10] {
            for y in [CGFloat(10), size.height - 10] {
                context.fill(Path(ellipseIn: CGRect(x: x - 1.4, y: y - 1.4, width: 2.8, height: 2.8)), with: .color(.white.opacity(0.15)))
            }
        }
    }

    private func drawRegions(context: inout GraphicsContext, plot: CGRect) {
        guard !frame.grid.isEmpty else { return }
        // Cell spacing is display geometry, derived from the already evaluated grid.
        let xs = Array(Set(frame.grid.map(\.x))).sorted()
        let ys = Array(Set(frame.grid.map(\.y))).sorted()
        let dx = xs.count > 1 ? xs[1] - xs[0] : 1
        let dy = ys.count > 1 ? ys[1] - ys[0] : 1
        for cell in frame.grid where cell.x.isFinite && cell.y.isFinite {
            let location = mapped(cell.x, cell.y, plot: plot)
            let rectangle = CGRect(x: location.x - dx * plot.width / 2,
                                   y: location.y - dy * plot.height / 2,
                                   width: dx * plot.width + 0.5, height: dy * plot.height + 0.5)
            let confidence = cell.confidence.isFinite ? max(0, min(1, cell.confidence)) : 0
            context.fill(Path(rectangle), with: .color(sampleColor(cell.label).opacity(0.065 + confidence * 0.115)))
        }
    }

    private func drawGrid(context: inout GraphicsContext, plot: CGRect) {
        var grid = Path()
        for index in 0...10 {
            let fraction = CGFloat(index) / 10
            grid.move(to: CGPoint(x: plot.minX + fraction * plot.width, y: plot.minY))
            grid.addLine(to: CGPoint(x: plot.minX + fraction * plot.width, y: plot.maxY))
            grid.move(to: CGPoint(x: plot.minX, y: plot.minY + fraction * plot.height))
            grid.addLine(to: CGPoint(x: plot.maxX, y: plot.minY + fraction * plot.height))
        }
        context.stroke(grid, with: .color(.white.opacity(0.055)), lineWidth: 0.5)
        var mid = Path()
        mid.move(to: CGPoint(x: plot.midX, y: plot.minY))
        mid.addLine(to: CGPoint(x: plot.midX, y: plot.maxY))
        mid.move(to: CGPoint(x: plot.minX, y: plot.midY))
        mid.addLine(to: CGPoint(x: plot.maxX, y: plot.midY))
        context.stroke(mid, with: .color(.white.opacity(0.09)), style: StrokeStyle(lineWidth: 0.6, dash: [2, 4]))
    }

    private func drawAxes(context: inout GraphicsContext, plot: CGRect) {
        context.stroke(Path(plot), with: .color(.white.opacity(0.18)), lineWidth: 0.7)
        for value in [0.0, 0.5, 1.0] {
            let x = plot.minX + value * plot.width
            let y = plot.maxY - value * plot.height
            let label = Text(value.formatted(.number.precision(.fractionLength(1))))
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundColor(MLTheme.muted)
            context.draw(label, at: CGPoint(x: x, y: plot.maxY + 12))
            context.draw(label, at: CGPoint(x: plot.minX - 9, y: y), anchor: .trailing)
        }
        let xTitle = algorithm == .monteCarlo || isRegression ? "x" : "x₁"
        let yTitle = algorithm == .monteCarlo || isRegression ? "y" : "x₂"
        context.draw(Text(xTitle).font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundColor(MLTheme.ink), at: CGPoint(x: plot.midX, y: plot.maxY + 26))
        context.draw(Text(yTitle).font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundColor(MLTheme.ink), at: CGPoint(x: plot.minX - 22, y: plot.midY - 13))
    }

    private func drawMonteCarlo(context: inout GraphicsContext, plot: CGRect) {
        var area = Path()
        area.move(to: mapped(0, 0, plot: plot))
        area.addLine(to: mapped(0, 1, plot: plot))
        for index in 0...100 {
            let angle = Double(index) / 100 * .pi / 2
            area.addLine(to: mapped(sin(angle), cos(angle), plot: plot))
        }
        area.closeSubpath()
        context.fill(area, with: .color(MLTheme.cyan.opacity(0.055)))
        var arc = Path()
        for index in 0...100 {
            let angle = Double(index) / 100 * .pi / 2
            let location = mapped(sin(angle), cos(angle), plot: plot)
            if index == 0 { arc.move(to: location) } else { arc.addLine(to: location) }
        }
        context.stroke(arc, with: .color(MLTheme.cyan.opacity(0.65)), style: StrokeStyle(lineWidth: 1.4, dash: [5, 4]))
        context.draw(Text("x² + y² = 1").font(.system(size: 10, weight: .medium, design: .monospaced)).foregroundColor(MLTheme.muted), at: mapped(0.74, 0.81, plot: plot))
    }

    private func drawPoints(context: inout GraphicsContext, plot: CGRect) {
        for point in frame.points where point.x.isFinite && point.y.isFinite {
            let position = mapped(point.x, point.y, plot: plot)
            let size: CGFloat = algorithm == .monteCarlo ? 2.7 : 3.8
            let displayLabel = isRegression || algorithm == .pca ? 0 : point.label
            let symbol = samplePath(at: position, label: displayLabel, radius: size)
            let color = sampleColor(displayLabel)
            if displayLabel == -2 {
                context.stroke(symbol, with: .color(MLTheme.muted.opacity(0.55)), lineWidth: 1)
            } else if displayLabel < 0 {
                context.stroke(symbol, with: .color(MLTheme.muted.opacity(0.8)), lineWidth: 1.2)
            } else if point.isTest {
                context.fill(symbol, with: .color(MLTheme.canvas))
                context.stroke(symbol, with: .color(color), lineWidth: 1.6)
            } else {
                context.fill(symbol, with: .color(color.opacity(0.9)))
                context.stroke(symbol, with: .color(.black.opacity(0.3)), lineWidth: 0.5)
            }
        }
    }

    private func samplePath(at p: CGPoint, label: Int, radius r: CGFloat) -> Path {
        if label == -2 {
            return Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2))
        }
        if label < 0 {
            var cross = Path()
            cross.move(to: CGPoint(x: p.x - r * 0.7, y: p.y - r * 0.7))
            cross.addLine(to: CGPoint(x: p.x + r * 0.7, y: p.y + r * 0.7))
            cross.move(to: CGPoint(x: p.x - r * 0.7, y: p.y + r * 0.7))
            cross.addLine(to: CGPoint(x: p.x + r * 0.7, y: p.y - r * 0.7))
            return cross
        }
        switch label % 3 {
        case 1:
            var triangle = Path()
            triangle.move(to: CGPoint(x: p.x, y: p.y - r * 1.2))
            triangle.addLine(to: CGPoint(x: p.x + r, y: p.y + r * 0.8))
            triangle.addLine(to: CGPoint(x: p.x - r, y: p.y + r * 0.8))
            triangle.closeSubpath()
            return triangle
        case 2:
            return Path(roundedRect: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2), cornerRadius: 0.7)
        default:
            return Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2))
        }
    }

    private func sampleColor(_ label: Int) -> Color {
        label < 0 ? MLTheme.muted : MLTheme.classColor(label)
    }

    private func drawCurve(context: inout GraphicsContext, plot: CGRect) {
        guard frame.curve.count > 1 else { return }
        var curve = Path()
        for (index, point) in frame.curve.enumerated() where point.x.isFinite && point.y.isFinite {
            let p = mapped(point.x, point.y, plot: plot)
            if index == 0 { curve.move(to: p) } else { curve.addLine(to: p) }
        }
        let curveColor = isRegression ? MLTheme.amber : MLTheme.cyan
        context.stroke(curve, with: .color(curveColor.opacity(0.12)), style: StrokeStyle(lineWidth: 7, lineCap: .round, lineJoin: .round))
        context.stroke(curve, with: .color(curveColor), style: StrokeStyle(lineWidth: 2.3, lineCap: .round, lineJoin: .round))
    }

    private func drawResiduals(context: inout GraphicsContext, plot: CGRect) {
        guard frame.curve.count > 1 else { return }
        var residuals = Path()
        let sortedCurve = frame.curve.sorted { $0.x < $1.x }
        for point in frame.points {
            guard let upper = sortedCurve.firstIndex(where: { $0.x >= point.x }), upper > 0 else { continue }
            let a = sortedCurve[upper - 1], b = sortedCurve[upper]
            let fraction = (point.x - a.x) / max(0.000_001, b.x - a.x)
            let prediction = a.y + (b.y - a.y) * fraction
            residuals.move(to: mapped(point.x, point.y, plot: plot))
            residuals.addLine(to: mapped(point.x, prediction, plot: plot))
        }
        context.stroke(residuals, with: .color(MLTheme.amber.opacity(0.25)), style: StrokeStyle(lineWidth: 0.8, dash: [2, 3]))
    }

    private func drawCenters(context: inout GraphicsContext, plot: CGRect) {
        for center in frame.centers {
            let p = mapped(center.x, center.y, plot: plot)
            let color = sampleColor(center.label)
            if algorithm == .dbscan {
                context.stroke(Path(ellipseIn: CGRect(x: p.x - 6, y: p.y - 6, width: 12, height: 12)), with: .color(color.opacity(0.35)), lineWidth: 0.7)
                continue
            }
            context.stroke(Path(ellipseIn: CGRect(x: p.x - 12, y: p.y - 12, width: 24, height: 24)), with: .color(color.opacity(0.65)), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            context.fill(Path(ellipseIn: CGRect(x: p.x - 7, y: p.y - 7, width: 14, height: 14)), with: .color(MLTheme.canvas))
            var cross = Path()
            cross.move(to: CGPoint(x: p.x - 5, y: p.y)); cross.addLine(to: CGPoint(x: p.x + 5, y: p.y))
            cross.move(to: CGPoint(x: p.x, y: p.y - 5)); cross.addLine(to: CGPoint(x: p.x, y: p.y + 5))
            context.stroke(cross, with: .color(color), style: StrokeStyle(lineWidth: 2, lineCap: .round))
        }
    }

    private func drawVectors(context: inout GraphicsContext, plot: CGRect) {
        for (index, vector) in frame.vectors.enumerated() {
            let start = mapped(vector.x1, vector.y1, plot: plot)
            let end = mapped(vector.x2, vector.y2, plot: plot)
            let color = algorithm == .pca ? (index == 0 ? MLTheme.amber : MLTheme.violet) : sampleColor(vector.label)
            var line = Path(); line.move(to: start); line.addLine(to: end)
            context.stroke(line, with: .color(color.opacity(algorithm == .pca ? 0.95 : 0.5)), style: StrokeStyle(lineWidth: algorithm == .pca ? 2.2 : 1, lineCap: .round))
            if algorithm == .pca {
                let direction = atan2(end.y - start.y, end.x - start.x)
                var tip = Path()
                tip.move(to: CGPoint(x: end.x - 8 * cos(direction - .pi / 6), y: end.y - 8 * sin(direction - .pi / 6)))
                tip.addLine(to: end)
                tip.addLine(to: CGPoint(x: end.x - 8 * cos(direction + .pi / 6), y: end.y - 8 * sin(direction + .pi / 6)))
                context.stroke(tip, with: .color(color), style: StrokeStyle(lineWidth: 2.2, lineCap: .round, lineJoin: .round))
                context.draw(Text("PC\(index + 1)").font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundColor(color), at: CGPoint(x: end.x, y: end.y - 13))
            }
        }
    }

    private func drawProbe(context: inout GraphicsContext, plot: CGRect) {
        guard let probe else { return }
        let p = mapped(probe.x, probe.y, plot: plot)
        var guides = Path()
        guides.move(to: CGPoint(x: plot.minX, y: p.y)); guides.addLine(to: CGPoint(x: plot.maxX, y: p.y))
        guides.move(to: CGPoint(x: p.x, y: plot.minY)); guides.addLine(to: CGPoint(x: p.x, y: plot.maxY))
        context.stroke(guides, with: .color(MLTheme.ink.opacity(0.5)), style: StrokeStyle(lineWidth: 0.8, dash: [4, 4]))
        context.stroke(Path(ellipseIn: CGRect(x: p.x - 9, y: p.y - 9, width: 18, height: 18)), with: .color(MLTheme.ink), lineWidth: 1.5)
        context.fill(Path(ellipseIn: CGRect(x: p.x - 2, y: p.y - 2, width: 4, height: 4)), with: .color(MLTheme.ink))
    }
}

struct HistorySparkline: View {
    let values: [Double]
    var color: Color = MLTheme.cyan

    var body: some View {
        Canvas { context, size in
            let finite = values.filter(\.isFinite)
            guard let minimum = finite.min(), let maximum = finite.max(), !finite.isEmpty else { return }
            let span = max(0.000_001, maximum - minimum)
            let inset: CGFloat = 3
            let width = max(1, size.width - inset * 2)
            let height = max(1, size.height - inset * 2)
            let coordinates = finite.enumerated().map { index, value in
                CGPoint(x: inset + CGFloat(index) / CGFloat(max(1, finite.count - 1)) * width,
                        y: inset + height * (maximum == minimum ? 0.5 : 1 - CGFloat((value - minimum) / span)))
            }
            var baseline = Path()
            baseline.move(to: CGPoint(x: 0, y: size.height - 2)); baseline.addLine(to: CGPoint(x: size.width, y: size.height - 2))
            context.stroke(baseline, with: .color(MLTheme.muted.opacity(0.18)), lineWidth: 0.5)
            var line = Path()
            for (index, point) in coordinates.enumerated() {
                if index == 0 { line.move(to: point) } else { line.addLine(to: point) }
            }
            var fill = line
            if let first = coordinates.first, let last = coordinates.last {
                fill.addLine(to: CGPoint(x: last.x, y: size.height))
                fill.addLine(to: CGPoint(x: first.x, y: size.height))
                fill.closeSubpath()
            }
            context.fill(fill, with: .linearGradient(Gradient(colors: [color.opacity(0.17), color.opacity(0)]), startPoint: .zero, endPoint: CGPoint(x: 0, y: size.height)))
            context.stroke(line, with: .color(color), style: StrokeStyle(lineWidth: 1.6, lineCap: .round, lineJoin: .round))
            if let last = coordinates.last {
                context.fill(Path(ellipseIn: CGRect(x: last.x - 2.5, y: last.y - 2.5, width: 5, height: 5)), with: .color(color))
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Metric history, \(values.count) recorded values")
    }
}
