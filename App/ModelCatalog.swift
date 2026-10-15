import MLCore

private struct AlgorithmMetadata {
    let title: String
    let shortTitle: String
    let family: String
    let symbol: String
    let tagline: String
    let equation: String
    let explanation: String
    let steps: [String]
    let limitation: String
    let defaultDataset: DatasetKind
    let accentIndex: Int
    let stepUnit: String
}

extension MLAlgorithm {
    var title: String { metadata.title }
    var shortTitle: String { metadata.shortTitle }
    var family: String { metadata.family }
    var symbol: String { metadata.symbol }
    var tagline: String { metadata.tagline }
    var equation: String { metadata.equation }
    var explanation: String { metadata.explanation }
    var steps: [String] { metadata.steps }
    var limitation: String { metadata.limitation }
    var defaultDataset: DatasetKind { metadata.defaultDataset }
    var accentIndex: Int { metadata.accentIndex }
    var stepUnit: String { metadata.stepUnit }

    func supportsDataset(_ dataset: DatasetKind) -> Bool {
        switch self {
        case .linearRegression, .polynomialRegression:
            return [.linear, .curve].contains(dataset)
        case .pca:
            return [.anisotropic, .blobs].contains(dataset)
        case .monteCarlo:
            return true // Sampling supplies its own uniform square.
        case .kMeans, .dbscan:
            return [.blobs, .moons, .circles, .anisotropic].contains(dataset)
        default:
            return [.blobs, .moons, .circles, .xor].contains(dataset)
        }
    }

    private var metadata: AlgorithmMetadata {
        switch self {
        case .linearRegression:
            return .init(
                title: "Linear Regression", shortTitle: "Linear", family: "Regression",
                symbol: "line.diagonal", tagline: "Fit a line. Watch the error shrink.",
                equation: "z = 2x − 1;  ŷ = b + wz\nL = MSE + λw²",
                explanation: "A slope and intercept turn x into a predicted y. Full-batch gradient descent reduces mean squared error plus an L2 penalty on the slope. Centered coordinates z keep the updates stable; the intercept is unpenalized.",
                steps: ["Predict y using centered x coordinates.", "Measure the training residuals and slope penalty.", "Update slope and intercept with the full-batch gradient."],
                limitation: "One straight line cannot follow a curved relationship. Outliers can pull the fit strongly.",
                defaultDataset: .linear, accentIndex: 0, stepUnit: "epoch")
        case .polynomialRegression:
            return .init(
                title: "Polynomial Regression", shortTitle: "Polynomial", family: "Regression",
                symbol: "waveform.path", tagline: "Give a linear model curved features.",
                equation: "z = 2x − 1\nŷ = b + w₁z + ⋯ + wₘzᵐ\nL = MSE + λΣwⱼ²",
                explanation: "Powers of centered x bend the curve while the model remains linear in its learned weights. Full-batch gradient descent reduces squared error plus an L2 weight penalty, excluding the intercept. Degree controls flexibility.",
                steps: ["Expand centered z into powers up to the degree.", "Predict with the weighted features.", "Update weights using error gradients and L2 shrinkage."],
                limitation: "High degree can fit noise and behave poorly beyond the observed x range. More flexibility needs more evidence.",
                defaultDataset: .curve, accentIndex: 0, stepUnit: "epoch")
        case .logisticRegression:
            return .init(
                title: "Logistic Regression", shortTitle: "Logistic", family: "Classification",
                symbol: "arrow.left.and.right", tagline: "Turn a linear score into a class probability.",
                equation: "p(class 1) = 1 / (1 + e⁻ˢ)\ns = b + w₁(2x−1) + w₂(2y−1)",
                explanation: "The sigmoid converts a linear score into a number between zero and one. Full-batch training reduces cross entropy with an L2 weight penalty; a 0.5 threshold selects a class.",
                steps: ["Score every training point.", "Map scores through the sigmoid.", "Update weights using classification error."],
                limitation: "This binary, linear boundary cannot separate concentric circles or XOR. A probability estimate is not a guarantee.",
                defaultDataset: .blobs, accentIndex: 1, stepUnit: "epoch")
        case .knn:
            return .init(
                title: "k-Nearest Neighbors", shortTitle: "k-NN", family: "Classification",
                symbol: "point.3.connected.trianglepath.dotted", tagline: "Ask the nearest examples to vote.",
                equation: "ŷ(q) = majority{yᵢ : xᵢ ∈ Nₖ(q)}",
                explanation: "This model stores examples rather than learning a fitted boundary. Frames grow its memory in batches. A query inherits the majority class of its k closest stored training points.",
                steps: ["Store a growing batch of training points.", "Select the k nearest stored points for a query.", "Count their votes and choose a class."],
                limitation: "Small k follows noise; large k smooths away small patterns. Euclidean distance depends on feature scale.",
                defaultDataset: .moons, accentIndex: 1, stepUnit: "batch")
        case .decisionTree:
            return .init(
                title: "Decision Tree", shortTitle: "Tree", family: "Classification",
                symbol: "point.topleft.down.to.point.bottomright.curvepath", tagline: "Split a hard question into simple questions.",
                equation: "Gini = 1 − Σ p(class)²\nsplit → largest weighted impurity decrease",
                explanation: "Each node asks whether x or y lies below a threshold. Greedy splits make child groups purer; each leaf predicts its majority class. Frames compare fresh fits with increasing depth limits.",
                steps: ["Compare candidate axis-aligned splits.", "Choose the split that reduces impurity most.", "Repeat until the depth limit or a pure leaf."],
                limitation: "Deep trees can memorize noise. Greedy choices and small changes in the data can create a different tree.",
                defaultDataset: .moons, accentIndex: 1, stepUnit: "depth")
        case .randomForest:
            return .init(
                title: "Random Forest", shortTitle: "Forest", family: "Classification",
                symbol: "square.stack.3d.up.fill", tagline: "Grow varied trees. Average their predictions.",
                equation: "p(1 | x) = mean{p₁(1 | x), …, pᴮ(1 | x)}\nclass 1 when p ≥ 0.5",
                explanation: "Trees train on bootstrap samples and consider one randomly chosen coordinate per split. The forest averages their empirical leaf probabilities, reducing dependence on any single tree.",
                steps: ["Resample training points with replacement.", "Grow a tree with randomized feature choices.", "Average its leaf probabilities with the forest."],
                limitation: "More trees cost more computation and do not remove bias. Mean leaf probability is not automatically calibrated confidence.",
                defaultDataset: .moons, accentIndex: 1, stepUnit: "tree")
        case .gradientBoosting:
            return .init(
                title: "Gradient Boosting", shortTitle: "Boosting", family: "Classification",
                symbol: "arrow.up.right", tagline: "Let the next tree correct the current ensemble.",
                equation: "Fₘ(x) = Fₘ₋₁(x) + η hₘ(x)",
                explanation: "A sequence of decision stumps improves a shared binary logistic score. Each stump targets the loss gradient of the current ensemble; the learning rate scales its contribution.",
                steps: ["Calculate the current model's errors.", "Fit a one-split stump to the loss gradient.", "Add a learning-rate-scaled correction."],
                limitation: "This binary logistic variant uses one-split stumps. Too many corrections can fit noise; learning rate and tree count interact.",
                defaultDataset: .moons, accentIndex: 1, stepUnit: "tree")
        case .naiveBayes:
            return .init(
                title: "Gaussian Naive Bayes", shortTitle: "Naive Bayes", family: "Classification",
                symbol: "bell.fill", tagline: "Combine simple likelihoods with Bayes' rule.",
                equation: "P(c | x, y) ∝ P(c) P(x | c) P(y | c)",
                explanation: "For each class, fit a Gaussian to each coordinate and estimate its prior frequency. Multiply those likelihoods with the prior to compare class scores. Frames refit those statistics from growing batches.",
                steps: ["Estimate class frequencies and coordinate means.", "Estimate each coordinate's variance by class.", "Combine the Gaussian likelihoods and priors."],
                limitation: "Coordinates are assumed independent within a class. Curved or correlated groups violate this assumption, and confidence can be excessive.",
                defaultDataset: .blobs, accentIndex: 1, stepUnit: "batch")
        case .svm:
            return .init(
                title: "Linear Support Vector Machine", shortTitle: "Linear SVM", family: "Classification",
                symbol: "rectangle.split.2x1", tagline: "Balance a wide margin against violations.",
                equation: "L = λ‖w‖²/2 + mean max(0, 1 − t(w·x + b))",
                explanation: "A linear score separates two classes. Hinge loss penalizes points on the wrong side or inside the target margin; regularization keeps the weights small.",
                steps: ["Calculate signed distance scores.", "Penalize margin violations with hinge loss.", "Update weights and apply regularization."],
                limitation: "This linear, soft-margin hinge-loss model is trained iteratively. It has no nonlinear kernel and cannot solve XOR directly.",
                defaultDataset: .blobs, accentIndex: 1, stepUnit: "epoch")
        case .kMeans:
            return .init(
                title: "k-Means Clustering", shortTitle: "k-Means", family: "Clustering",
                symbol: "circle.hexagongrid.fill", tagline: "Assign, average, repeat.",
                equation: "J = Σᵢ ‖xᵢ − μ(clusterᵢ)‖²",
                explanation: "Choose k centers, assign each point to its nearest center, then move centers to their assigned means. Repeat to reduce within-cluster squared distance.",
                steps: ["Assign points to the nearest center.", "Move each center to its group's mean.", "Repeat until centers settle or the limit ends."],
                limitation: "You must choose k. Initialization matters, and distance-based partitions struggle with moons, rings, and unequal densities.",
                defaultDataset: .blobs, accentIndex: 2, stepUnit: "iteration")
        case .dbscan:
            return .init(
                title: "DBSCAN", shortTitle: "DBSCAN", family: "Clustering",
                symbol: "circle.dotted.circle.fill", tagline: "Follow dense neighborhoods into clusters.",
                equation: "Nε(p) = {q : distance(p, q) ≤ ε}\ncore point: |Nε(p)| ≥ minPoints",
                explanation: "A core point has enough neighbors inside radius ε, including itself. Connected dense neighborhoods grow clusters; unsupported points remain noise.",
                steps: ["Count neighbors within radius ε.", "Expand a cluster through connected core points.", "Attach border points and leave sparse points as noise."],
                limitation: "One radius may not suit groups with different densities. A large radius can merge groups; a small one can mark most points as noise.",
                defaultDataset: .moons, accentIndex: 2, stepUnit: "point")
        case .pca:
            return .init(
                title: "Principal Component Analysis", shortTitle: "PCA", family: "Projection",
                symbol: "arrow.up.right.and.arrow.down.left", tagline: "Find the direction that carries the most variation.",
                equation: "C v₁ = λ₁ v₁\nz = (x − mean) · v₁",
                explanation: "As training batches accumulate, center the observed cloud and solve its 2D covariance axes. Projecting onto the dominant axis compresses two coordinates into one while retaining maximal linear variance.",
                steps: ["Center the growing batch of observations.", "Find the largest-eigenvalue covariance direction.", "Measure reconstruction error after projection onto PC1."],
                limitation: "Variance does not imply usefulness or class separation. PCA is linear, scale-sensitive, and does not discover curved manifolds.",
                defaultDataset: .anisotropic, accentIndex: 2, stepUnit: "batch")
        case .monteCarlo:
            return .init(
                title: "Monte Carlo Sampling", shortTitle: "Monte Carlo", family: "Sampling",
                symbol: "dice.fill", tagline: "Estimate π by counting random hits.",
                equation: "π estimate = 4 × hits / samples\nhit when x² + y² ≤ 1",
                explanation: "Uniform draws in the unit square turn a quarter-circle's area ratio into an estimate of π. The estimate fluctuates as evidence accumulates. This is numerical sampling, not a predictive ML model.",
                steps: ["Draw independent uniform coordinates.", "Test whether x² + y² is at most one.", "Multiply the fraction of hits by four."],
                limitation: "Random error shrinks slowly, typically like 1/√N. More samples do not guarantee a closer estimate on every step.",
                defaultDataset: .blobs, accentIndex: 0, stepUnit: "sample")
        }
    }
}

extension DatasetKind {
    var title: String {
        switch self {
        case .blobs: "Blobs"
        case .moons: "Two Moons"
        case .circles: "Concentric Circles"
        case .xor: "XOR"
        case .linear: "Linear Trend"
        case .curve: "Curved Trend"
        case .anisotropic: "Stretched Cloud"
        }
    }

    var detail: String {
        switch self {
        case .blobs: "Compact groups with controllable overlap."
        case .moons: "Two curved groups that challenge a straight boundary."
        case .circles: "An inner ring and an outer ring."
        case .xor: "Opposite quadrants share a class."
        case .linear: "A straight relationship with adjustable noise."
        case .curve: "A nonlinear relationship with adjustable noise."
        case .anisotropic: "A stretched, tilted cloud with correlated coordinates."
        }
    }
}
