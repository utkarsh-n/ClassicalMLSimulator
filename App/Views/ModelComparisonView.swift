import SwiftUI
import MLCore

private struct ComparisonRow: Identifiable, Sendable {
    let algorithm: MLAlgorithm
    let frame: SimulationFrame
    var id: String { algorithm.rawValue }
}

struct ModelComparisonView: View {
    let points: [MLPoint]
    let configuration: MLConfiguration
    let regression: Bool
    let datasetName: String
    var onSelect: (MLAlgorithm) -> Void
    @State private var rows: [ComparisonRow] = []
    @State private var computing = true

    var body: some View {
        ZStack {
            LabBackground()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    LabPanel("One dataset / several approaches") {
                        Text(datasetName).font(.system(.title2, design: .rounded, weight: .semibold)).foregroundStyle(MLTheme.ink)
                        Text("\(points.filter { !$0.isTest }.count) training · \(points.filter(\.isTest).count) test · seed \(configuration.seed)")
                            .font(.system(.caption, design: .monospaced)).foregroundStyle(MLTheme.cyan)
                        Text(regression ? "Compare final mean squared errors. Lower is better. Each model sees exactly the same split." : "Compare final classification accuracies. Higher is better. Each model sees exactly the same split.")
                            .font(.subheadline).foregroundStyle(MLTheme.muted)
                    }
                    if computing {
                        ProgressView("Computing on this iPhone…").frame(maxWidth: .infinity).padding(30)
                    }
                    ForEach(rows) { row in
                        LabPanel {
                            Button { onSelect(row.algorithm) } label: {
                                HStack {
                                    Label(row.algorithm.title, systemImage: row.algorithm.symbol)
                                        .font(.system(.headline, design: .rounded)).foregroundStyle(MLTheme.ink)
                                    Spacer()
                                    Image(systemName: "arrow.up.right").foregroundStyle(MLTheme.cyan)
                                }.frame(minHeight: 44)
                            }.buttonStyle(.plain).accessibilityLabel("Open \(row.algorithm.title) lab")
                            ViewThatFits(in: .horizontal) {
                                HStack(alignment: .top, spacing: 20) { comparisonMetrics(row) }
                                VStack(alignment: .leading, spacing: 16) { comparisonMetrics(row) }
                            }
                        }
                    }
                    Text("A small synthetic split is an illustration, not a model benchmark. Results depend on the data, seed, settings, and training budget. No hyperparameter search or cross-validation is performed.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                }.frame(maxWidth: 620).padding(16).frame(maxWidth: .infinity)
            }.accessibilityIdentifier("comparisonScroll")
        }.navigationTitle("Compare models").navigationBarTitleDisplayMode(.inline)
            .task {
                let models: [MLAlgorithm] = regression ? [.linearRegression, .polynomialRegression] : [.logisticRegression, .knn, .decisionTree, .randomForest, .gradientBoosting, .naiveBayes, .svm]
                let data = points, settings = configuration
                let computation = Task.detached(priority: .userInitiated) {
                    var results: [ComparisonRow] = []
                    for model in models {
                        guard !Task.isCancelled else { break }
                        let result = MLEngine.run(algorithm: model, points: data, configuration: settings)
                        guard let frame = result.frames.last else { continue }
                        results.append(ComparisonRow(algorithm: model, frame: frame))
                    }
                    return results
                }
                let results = await withTaskCancellationHandler {
                    await computation.value
                } onCancel: {
                    computation.cancel()
                }
                guard !Task.isCancelled else { return }
                rows = results; computing = false
            }
    }

    private func comparisonMetrics(_ row: ComparisonRow) -> some View {
        Group {
            ForEach(row.frame.metrics.filter { $0.name.lowercased().contains("train") || $0.name.lowercased().contains("test") }.prefix(2)) { metric in
                MetricReadout(metric: metric).frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}
