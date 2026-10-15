import SwiftUI
import MLCore

struct ModelControlsView: View {
    @Bindable var session: LabSession
    var body: some View {
        LabPanel("02 / Model controls") {
            switch session.algorithm {
            case .linearRegression:
                learningRate
                regularization
                Text("A straight line learns its slope and intercept with full-batch gradient descent.").labCaption()
            case .polynomialRegression:
                integer("Polynomial degree", keyPath: \.degree, range: 1...6)
                learningRate
                regularization
            case .logisticRegression:
                learningRate
                regularization
            case .knn:
                integer("Neighbors · k", keyPath: \.neighborCount, range: 1...21, step: 2)
                Text("Tap the plot to connect a query to its nearest training samples. More neighbors smooth the boundary.").labCaption()
            case .decisionTree:
                integer("Maximum depth", keyPath: \.treeDepth, range: 1...6)
                Text("Deeper trees can carve finer regions, but may fit noise. Watch the held-out score.").labCaption()
            case .randomForest:
                integer("Trees", keyPath: \.treeCount, range: 3...30, step: 3)
                integer("Maximum depth", keyPath: \.treeDepth, range: 1...6)
                Text("Each tree learns from a bootstrap sample and randomized feature choices. The forest averages leaf probabilities.").labCaption()
            case .gradientBoosting:
                integer("Boosting stages", keyPath: \.treeCount, range: 3...30, step: 3)
                learningRate
                Text("Small residual stumps are added in sequence to improve a binary classifier.").labCaption()
            case .naiveBayes:
                Text("No tuning knob is needed here. Step through accumulating training evidence and inspect Gaussian class regions.").labCaption()
            case .svm:
                learningRate
                regularization
                Text("This lab uses a linear separator with hinge loss and L2 regularization. It cannot untangle XOR or rings.").labCaption()
            case .kMeans:
                integer("Clusters · k", keyPath: \.clusters, range: 2...6)
                Text("Alternate nearest-center assignments with center updates. Reseed to explore initialization.").labCaption()
            case .dbscan:
                decimal("Neighborhood radius · ε", keyPath: \.epsilon, range: 0.05...0.3, step: 0.01)
                integer("Minimum neighbors", keyPath: \.minPoints, range: 2...12)
                Text("Dense neighborhoods form clusters; isolated points become noise. The minimum count includes the point itself.").labCaption()
            case .pca:
                Text("Principal axes come from the training covariance matrix. Step through the incoming observations to see those axes stabilize.").labCaption()
            case .monteCarlo:
                Text("Every sample is an independent uniform draw. More samples reduce typical error, but individual estimates can move away from π.").labCaption()
            }
            if [.linearRegression, .polynomialRegression, .logisticRegression, .svm, .kMeans].contains(session.algorithm) {
                integer("Iterations", keyPath: \.iterations, range: 10...40, step: 5)
            }
        }
    }

    private var learningRate: some View {
        decimal("Learning rate · η", keyPath: \.learningRate, range: 0.02...0.8, step: 0.02)
    }
    private var regularization: some View {
        decimal("Regularization · λ", keyPath: \.regularization, range: 0...0.2, step: 0.01)
    }
    private func integer(_ label: String, keyPath: WritableKeyPath<MLConfiguration, Int>, range: ClosedRange<Int>, step: Int = 1) -> some View {
        Stepper(value: Binding(get: { session.configuration[keyPath: keyPath] }, set: { value in
            session.configuration[keyPath: keyPath] = value; session.recompute()
        }), in: range, step: step) {
            HStack {
                Text(label).foregroundStyle(MLTheme.ink)
                Spacer()
                Text("\(session.configuration[keyPath: keyPath])").monospacedDigit().foregroundStyle(MLTheme.cyan)
            }.font(.system(.subheadline, design: .rounded))
        }.accessibilityIdentifier("parameter-\(label)")
    }
    private func decimal(_ label: String, keyPath: WritableKeyPath<MLConfiguration, Double>, range: ClosedRange<Double>, step: Double) -> some View {
        VStack(spacing: 3) {
            HStack {
                Text(label).foregroundStyle(MLTheme.ink)
                Spacer()
                Text(session.configuration[keyPath: keyPath].formatted(.number.precision(.fractionLength(2))))
                    .monospacedDigit().foregroundStyle(MLTheme.cyan)
                    .accessibilityIdentifier("parameterValue-\(label)")
            }.font(.system(.subheadline, design: .rounded))
            Slider(value: Binding(get: { session.configuration[keyPath: keyPath] }, set: { value in
                guard value != session.configuration[keyPath: keyPath] else { return }
                session.configuration[keyPath: keyPath] = value
                session.recompute()
            }), in: range, step: step)
                .accessibilityLabel(label).accessibilityIdentifier("parameter-\(label)")
        }
    }
}

private extension View {
    func labCaption() -> some View { font(.subheadline).foregroundStyle(MLTheme.muted).fixedSize(horizontal: false, vertical: true) }
}
