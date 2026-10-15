import Foundation

public enum MLEngine {
    public static func run(algorithm: MLAlgorithm, points: [MLPoint], configuration: MLConfiguration) -> SimulationResult {
        if algorithm == .monteCarlo { return monteCarlo(configuration) }
        let isClassifier: Bool
        switch algorithm {
        case .logisticRegression, .knn, .decisionTree, .randomForest, .gradientBoosting, .naiveBayes, .svm:
            isClassifier = true
        default: isClassifier = false
        }
        let clean = normalized(points, binary: isClassifier)
        guard clean.contains(where: { !$0.isTest }) else {
            return SimulationResult(frames: [SimulationFrame(step: 0, caption: "Training samples needed",
                detail: "Add at least one training sample before fitting a model. Held-out samples do not train the model.", points: clean)],
                summary: "No model fitted: there are no training samples.")
        }
        var result: SimulationResult
        switch algorithm {
        case .linearRegression: result = regression(clean, configuration, polynomial: false)
        case .polynomialRegression: result = regression(clean, configuration, polynomial: true)
        case .logisticRegression: result = logistic(clean, configuration)
        case .knn: result = nearestNeighbors(clean, configuration)
        case .decisionTree: result = cart(clean, configuration)
        case .randomForest: result = forest(clean, configuration)
        case .gradientBoosting: result = boosting(clean, configuration)
        case .naiveBayes: result = naiveBayes(clean, configuration)
        case .svm: result = supportVectorMachine(clean, configuration)
        case .kMeans: result = kMeans(clean, configuration)
        case .dbscan: result = dbscan(clean, configuration)
        case .pca: result = pca(clean, configuration)
        case .monteCarlo: result = monteCarlo(configuration)
        }
        if isClassifier && points.contains(where: { $0.label > 1 }) {
            let explanation = " Binary task: source group 0 is class 0; source groups 1 and 2 are class 1."
            result.summary += explanation
            for index in result.frames.indices { result.frames[index].detail += explanation }
        }
        if !clean.contains(where: \.isTest) {
            result.summary += " No held-out samples were supplied, so no test metric is available."
        }
        if Set(points.prefix(320).map(\.id)).count < min(320, points.count) {
            result.summary += " Duplicate input IDs were assigned distinct display identities; all samples were retained."
        }
        return result
    }
}

extension MLEngine {
    static func regression(_ points: [MLPoint], _ config: MLConfiguration, polynomial: Bool) -> SimulationResult {
        let train = points.filter { !$0.isTest }, test = points.filter(\.isTest)
        let degree = polynomial ? min(6, max(1, config.degree)) : 1
        let lambda = ridge(config)
        var weights = [Double](repeating: 0, count: degree + 1)
        weights[0] = average(train.map(\.y))
        func basis(_ x: Double) -> [Double] {
            let z = 2 * x - 1
            var result = [1.0]
            for _ in 1...degree { result.append(result.last! * z) }
            return result
        }
        let features = train.map { basis($0.x) }
        var frames: [SimulationFrame] = [], history: [Double] = []
        for step in 1...steps(config) {
            var gradient = [Double](repeating: 0, count: weights.count)
            for (point, feature) in zip(train, features) {
                let prediction = zip(weights, feature).reduce(0) { $0 + $1.0 * $1.1 }
                for j in weights.indices { gradient[j] += 2 * (prediction - point.y) * feature[j] / Double(train.count) }
            }
            for j in weights.indices {
                weights[j] -= rate(config) * (gradient[j] + (j == 0 ? 0 : 2 * lambda * weights[j]))
            }
            let current = weights
            let predict: (Double) -> Double = { x in zip(current, basis(x)).reduce(0) { $0 + $1.0 * $1.1 } }
            func mse(_ samples: [MLPoint]) -> Double { average(samples.map { pow(predict($0.x) - $0.y, 2) }) }
            history.append(mse(train))
            var metrics = [MLMetric("Train MSE", mse(train))]
            if !test.isEmpty { metrics.append(MLMetric("Test MSE", mse(test))) }
            metrics.append(MLMetric("Degree", Double(degree)))
            let curve = (0...60).map { index in
                let x = Double(index) / 60
                return MLPoint(id: 10_000 + index, x: x, y: predict(x))
            }
            frames.append(SimulationFrame(step: step, caption: "Gradient update \(step)",
                detail: "Fit y from x. Full-batch descent minimizes MSE + λΣw², excluding the intercept. z = 2x − 1; w = [\(weights.map { number($0) }.joined(separator: ", "))]. λ = \(number(lambda)).",
                metrics: metrics, points: points, curve: curve, history: history))
        }
        return SimulationResult(frames: frames, summary: "\(polynomial ? "Polynomial" : "Linear") least-squares regression with \(steps(config)) actual gradient updates. Training fits the weights; test MSE uses only the held-out fifth. Finite updates do not imply convergence.")
    }

    static func logistic(_ points: [MLPoint], _ config: MLConfiguration) -> SimulationResult {
        let train = points.filter { !$0.isTest }
        var weights = [logit(average(train.map { Double($0.label) })), 0, 0]
        var frames: [SimulationFrame] = [], history: [Double] = []
        for step in 1...steps(config) {
            var gradient = [Double](repeating: 0, count: 3)
            for point in train {
                let features = [1, 2 * point.x - 1, 2 * point.y - 1]
                let p = sigmoid(zip(weights, features).reduce(0) { $0 + $1.0 * $1.1 })
                for j in 0..<3 { gradient[j] += (p - Double(point.label)) * features[j] / Double(train.count) }
            }
            for j in 0..<3 { weights[j] -= rate(config) * (gradient[j] + (j == 0 ? 0 : ridge(config) * weights[j])) }
            let w = weights
            let probability: (MLPoint) -> Double = { sigmoid(w[0] + w[1] * (2 * $0.x - 1) + w[2] * (2 * $0.y - 1)) }
            history.append(logLoss(train, probability: probability))
            frames.append(classificationFrame(step: step, caption: "Logistic update \(step)",
                detail: "p(class 1) = sigmoid(b + wₓ(2x−1) + wᵧ(2y−1)). Cross-entropy gradients update b = \(number(w[0])), wₓ = \(number(w[1])), wᵧ = \(number(w[2])). The boundary remains linear.",
                points: points, probability: probability, history: history))
        }
        return SimulationResult(frames: frames, summary: "Binary logistic regression, fitted by regularized full-batch cross-entropy descent. The colored grid evaluates the actual fitted probability at 576 locations; it cannot bend around XOR or concentric classes.")
    }

    static func nearestNeighbors(_ points: [MLPoint], _ config: MLConfiguration) -> SimulationResult {
        let train = points.filter { !$0.isTest }
        let frameCount = min(12, train.count), requestedK = min(31, max(1, config.neighborCount))
        var frames: [SimulationFrame] = [], history: [Double] = []
        for step in 1...frameCount {
            let count = (step * train.count + frameCount - 1) / frameCount
            let memory = Array(train.prefix(count)), k = min(requestedK, count)
            let probability: (MLPoint) -> Double = { query in
                let neighbors = memory.sorted { a, b in
                    let da = squaredDistance(a, query), db = squaredDistance(b, query)
                    return da == db ? a.id < b.id : da < db
                }.prefix(k)
                return Double(neighbors.reduce(0) { $0 + $1.label }) / Double(k)
            }
            history.append(logLoss(train, probability: probability))
            var frame = classificationFrame(step: step, caption: "Memorize \(count) training samples",
                detail: "No gradient or fitted equation: store samples, find the \(k) closest by Euclidean distance, and vote. Ties vote class 1. Metrics evaluate all training samples, including self-neighbors; test samples never enter memory.",
                points: points, probability: probability, history: history, extra: [MLMetric("Neighbors k", Double(k)), MLMetric("Memorized", Double(count))])
            let known = Set(memory.map(\.id))
            frame.points = points.filter { $0.isTest || known.contains($0.id) }
            frames.append(frame)
        }
        return SimulationResult(frames: frames, summary: "k-nearest neighbors gradually reveals the stored training set. In-sample accuracy includes each sample itself; held-out accuracy is the useful generalization check. Grid confidence is the local vote fraction.")
    }

    static func naiveBayes(_ points: [MLPoint], _ config: MLConfiguration) -> SimulationResult {
        let train = points.filter { !$0.isTest }, frameCount = min(12, train.count)
        var frames: [SimulationFrame] = [], history: [Double] = []
        for step in 1...frameCount {
            let count = (step * train.count + frameCount - 1) / frameCount
            let known = Array(train.prefix(count))
            var means = [(Double, Double)](), variances = [(Double, Double)](), priors = [Double]()
            for label in 0...1 {
                let group = known.filter { $0.label == label }
                let meanX = average(group.map(\.x)), meanY = average(group.map(\.y))
                means.append((meanX, meanY))
                variances.append((max(1e-4, average(group.map { pow($0.x - meanX, 2) })),
                                  max(1e-4, average(group.map { pow($0.y - meanY, 2) }))))
                priors.append(Double(group.count) / Double(known.count))
            }
            let m = means, v = variances, prior = priors
            let probability: (MLPoint) -> Double = { point in
                func score(_ c: Int) -> Double {
                    guard prior[c] > 0 else { return -1e6 }
                    return log(prior[c]) - 0.5 * (log(2 * .pi * v[c].0) + pow(point.x - m[c].0, 2) / v[c].0
                        + log(2 * .pi * v[c].1) + pow(point.y - m[c].1, 2) / v[c].1)
                }
                return sigmoid(score(1) - score(0))
            }
            history.append(logLoss(train, probability: probability))
            let centers = (0...1).filter { prior[$0] > 0 }.map { MLPoint(id: 20_000 + $0, x: m[$0].0, y: m[$0].1, label: $0) }
            frames.append(classificationFrame(step: step, caption: "Estimate from \(count) training samples",
                detail: "Fit one mean and variance per feature and class, plus class priors. Gaussian naïve Bayes assumes x and y are conditionally independent. Variances have a 0.0001 floor to keep identical samples finite.",
                points: points, probability: probability, history: history, centers: centers, extra: [MLMetric("Samples fitted", Double(count))]))
        }
        return SimulationResult(frames: frames, summary: "Gaussian naïve Bayes updates real sufficient statistics as training samples arrive. Curved boundaries follow class Gaussian densities; correlated or multimodal classes can violate its assumptions.")
    }

    static func supportVectorMachine(_ points: [MLPoint], _ config: MLConfiguration) -> SimulationResult {
        let train = points.filter { !$0.isTest }, lambda = ridge(config)
        var wx = 0.0, wy = 0.0, bias = 0.0
        var frames: [SimulationFrame] = [], history: [Double] = []
        for step in 1...steps(config) {
            var gx = lambda * wx, gy = lambda * wy, gb = 0.0
            for point in train {
                let target = point.label == 1 ? 1.0 : -1.0
                let x = 2 * point.x - 1, y = 2 * point.y - 1
                if target * (wx * x + wy * y + bias) < 1 {
                    gx -= target * x / Double(train.count)
                    gy -= target * y / Double(train.count)
                    gb -= target / Double(train.count)
                }
            }
            wx -= rate(config) * gx; wy -= rate(config) * gy; bias -= rate(config) * gb
            let w1 = wx, w2 = wy, b = bias
            let score: (MLPoint) -> Double = { w1 * (2 * $0.x - 1) + w2 * (2 * $0.y - 1) + b }
            let loss = average(train.map { max(0, 1 - ($0.label == 1 ? 1.0 : -1.0) * score($0)) })
            history.append(loss)
            var frame = classificationFrame(step: step, caption: "Hinge-loss update \(step)",
                detail: "Linear soft-margin SVM minimizes mean hinge loss + λ‖w‖²/2. Samples with y·score < 1 drive the update. Shading uses sigmoid(3·score) for display; it is not a calibrated probability.",
                points: points, probability: { sigmoid(3 * score($0)) }, history: history,
                customLoss: MLMetric("Train hinge loss", loss))
            let supportCount = train.filter { ($0.label == 1 ? 1.0 : -1.0) * score($0) <= 1 + 1e-8 }.count
            frame.metrics.append(MLMetric("Within margin", Double(supportCount)))
            frames.append(frame)
        }
        return SimulationResult(frames: frames, summary: "A genuine primal, full-batch linear SVM with hinge-loss subgradient updates. The model has no kernel; non-linearly separable data remains challenging. Shading visualizes signed margin, not calibrated confidence.")
    }
}
