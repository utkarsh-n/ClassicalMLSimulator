import SwiftUI

struct FieldGuideView: View {
    var body: some View {
        ZStack {
            LabBackground()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    VStack(alignment: .leading, spacing: 10) {
                        Engraving("The field guide")
                        Text("Learn by changing\none thing.")
                            .font(.system(.largeTitle, design: .serif)).foregroundStyle(MLTheme.ink)
                        Text("A small dataset can expose a big idea. Use these experiments to build intuition before reaching for a larger model.")
                            .font(.subheadline).foregroundStyle(MLTheme.muted)
                    }.padding(.vertical, 12)
                    guideCard("01", "Draw a boundary", symbol: "point.topleft.down.to.point.bottomright.curvepath", color: MLTheme.cyan,
                              text: "Open k-nearest neighbors on Moons. Run the model, then tap between the two arcs. Set k to 1, then 15. Observe how local votes trade detail for smoother decisions.")
                    guideCard("02", "Grow a forest", symbol: "tree", color: MLTheme.amber,
                              text: "Open Decision tree. Increase depth and compare training and test accuracy. Switch to Random forest and step through the votes as trees join the ensemble. One displayed tree explains a single member.")
                    guideCard("03", "Find structure", symbol: "circle.hexagongrid", color: MLTheme.violet,
                              text: "Open K-means on Blobs. Change the cluster count. Follow the crosses as centers move. Try Moons, then DBSCAN: spherical distance and density make different assumptions.")
                    guideCard("04", "Fit a curve", symbol: "waveform.path", color: MLTheme.cyan,
                              text: "Open Polynomial regression on Curved trend. Try degree 1 and degree 6. Add noise and inspect the test error. More flexibility and a finite training budget do not guarantee a better fit.")
                    guideCard("05", "Rotate the viewpoint", symbol: "arrow.up.right.and.arrow.down.left", color: MLTheme.violet,
                              text: "Open PCA on Stretched cloud. The longer arrow follows the greatest training variance. PCA summarizes geometry; it does not use class labels or discover a causal explanation.")
                    guideCard("06", "Let chance compute", symbol: "dice", color: MLTheme.amber,
                              text: "Open Monte Carlo. Count points inside the quarter circle. The area ratio estimates π/4. Increase the sample count and reseed: error fluctuates, even though typical uncertainty falls with more draws.")
                    LabPanel("Read the instrument") {
                        explanation("Features", "Axes are synthetic normalized feature values between 0 and 1. Regression uses the vertical coordinate as the target y. No real-world physical units are implied.")
                        explanation("Filled vs hollow", "Filled points train the model; hollow points are held out. Classification uses two groups: the first source group is A and the remaining source groups are B. Cluster colors are arbitrary identities.")
                        explanation("Decision regions", "The background evaluates a 24 × 24 grid of the current model. Probe class labels refer to the nearest grid cell. A grid is a sampled approximation, not an exact analytic boundary.")
                        explanation("Frames", "Frames reveal actual epochs, tree votes, assignments, or batches. Replay scrubs precomputed states; it does not retrain on each animation tick.")
                        explanation("Loss and accuracy", "Training scores describe familiar data. Test scores describe the held-out split. This small fixed split is useful for exploration but cannot establish real-world reliability.")
                        explanation("Trace", "The sparkline shows a recorded learning quantity: training MSE, log or hinge loss, cluster inertia, noise fraction, PC1 variance, or a Monte Carlo estimate. Read its caption for the quantity and direction.")
                    }
                    LabPanel("Model boundaries") {
                        Text("These are real, bounded computations on synthetic two-dimensional data. They are intentionally small enough to inspect.")
                            .font(.subheadline).foregroundStyle(MLTheme.ink)
                        Text("SVM uses a linear separator. Boosting uses binary logistic residual stumps. PCA shows a two-dimensional covariance decomposition. DBSCAN’s held-out assignment is a nearest-core extension. Monte Carlo is numerical sampling, rather than a predictive ML model.")
                            .font(.subheadline).foregroundStyle(MLTheme.muted)
                        Text("No model downloads, server training, tracking, or network access are required. Closing a lab does not save your added samples; use it as a scratch workbench.")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    LabPanel("Primary references") {
                        source("Linear models & regression", url: "https://scikit-learn.org/stable/modules/linear_model.html")
                        source("Trees, forests & boosting", url: "https://scikit-learn.org/stable/modules/ensemble.html")
                        source("Clustering methods", url: "https://scikit-learn.org/stable/modules/clustering.html")
                        source("PCA & decomposition", url: "https://scikit-learn.org/stable/modules/decomposition.html")
                        source("Monte Carlo · Los Alamos", url: "https://mcnp-green.lanl.gov/pdf_files/Article_1987_LAS_Metropolis_125--130.pdf")
                        Text("The algorithms are implemented in Swift; scikit-learn is a conceptual reference, not a runtime dependency.")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                    }
                }.frame(maxWidth: 620).padding(16).frame(maxWidth: .infinity)
            }.scrollIndicators(.hidden)
        }.navigationTitle("Field guide").navigationBarTitleDisplayMode(.inline)
    }
    private func guideCard(_ number: String, _ title: String, symbol: String, color: Color, text: String) -> some View {
        LabPanel {
            HStack {
                Engraving("Experiment \(number)"); Spacer()
                Image(systemName: symbol).foregroundStyle(color)
            }
            Text(title).font(.system(.title3, design: .rounded, weight: .semibold)).foregroundStyle(MLTheme.ink)
            Text(text).font(.subheadline).foregroundStyle(MLTheme.muted)
        }
    }
    private func explanation(_ title: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.system(.subheadline, design: .rounded, weight: .semibold)).foregroundStyle(MLTheme.cyan)
            Text(text).font(.subheadline).foregroundStyle(MLTheme.muted)
        }
    }
    private func source(_ title: String, url: String) -> some View {
        Link(destination: URL(string: url)!) {
            HStack { Text(title); Spacer(); Image(systemName: "arrow.up.right") }.font(.subheadline).frame(minHeight: 44)
        }
    }
}
