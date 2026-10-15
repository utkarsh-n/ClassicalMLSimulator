import Foundation

indirect enum ClassificationTree {
    case leaf(probability: Double, count: Int)
    case split(feature: Int, threshold: Double, probability: Double, count: Int, left: ClassificationTree, right: ClassificationTree)

    func probability(_ point: MLPoint) -> Double {
        switch self {
        case let .leaf(probability, _): return probability
        case let .split(feature, threshold, _, _, left, right):
            return ((feature == 0 ? point.x : point.y) <= threshold ? left : right).probability(point)
        }
    }

    func display() -> [TreeDisplayNode] {
        var result: [TreeDisplayNode] = []
        func visit(_ node: ClassificationTree, parent: Int?, depth: Int) {
            let id = result.count
            switch node {
            case let .leaf(probability, count):
                result.append(TreeDisplayNode(id: id, parentID: parent,
                    label: "Class \(probability >= 0.5 ? 1 : 0) · n=\(count)", prediction: probability >= 0.5 ? 1 : 0, depth: depth))
            case let .split(feature, threshold, probability, _, left, right):
                result.append(TreeDisplayNode(id: id, parentID: parent, label: "\(feature == 0 ? "x" : "y") ≤ \(number(threshold, digits: 2))",
                    prediction: probability >= 0.5 ? 1 : 0, depth: depth))
                visit(left, parent: id, depth: depth + 1)
                visit(right, parent: id, depth: depth + 1)
            }
        }
        visit(self, parent: nil, depth: 0)
        return result
    }
}

func fitTree(_ samples: [MLPoint], depth: Int, featureSubsample: Bool, rng: inout SeededRandom) -> ClassificationTree {
    let positive = samples.reduce(0) { $0 + $1.label }
    let probability = Double(positive) / Double(max(1, samples.count))
    guard depth > 0, samples.count >= 4, probability > 0, probability < 1 else {
        return .leaf(probability: probability, count: samples.count)
    }
    let features = featureSubsample ? [rng.index(2)] : [0, 1]
    var bestGain = -Double.infinity, bestFeature = 0, bestThreshold = 0.0
    let parentGini = 2 * probability * (1 - probability)
    for feature in features {
        let ordered = samples.sorted { a, b in
            let av = feature == 0 ? a.x : a.y, bv = feature == 0 ? b.x : b.y
            return av == bv ? a.id < b.id : av < bv
        }
        var leftPositive = 0
        for i in 0..<(ordered.count - 1) {
            leftPositive += ordered[i].label
            let leftCount = i + 1, rightCount = ordered.count - leftCount
            guard leftCount >= 2, rightCount >= 2 else { continue }
            let a = feature == 0 ? ordered[i].x : ordered[i].y
            let b = feature == 0 ? ordered[i + 1].x : ordered[i + 1].y
            guard b - a > 1e-12 else { continue }
            let lp = Double(leftPositive) / Double(leftCount)
            let rp = Double(positive - leftPositive) / Double(rightCount)
            let impurity = (Double(leftCount) * 2 * lp * (1 - lp) + Double(rightCount) * 2 * rp * (1 - rp)) / Double(samples.count)
            let gain = parentGini - impurity
            if gain > bestGain {
                bestGain = gain; bestFeature = feature; bestThreshold = (a + b) / 2
            }
        }
    }
    // A zero-gain split can unlock informative descendants (balanced XOR is the
    // canonical example). Stop only when no valid split exists or gain is negative.
    guard bestGain >= -1e-12 else { return .leaf(probability: probability, count: samples.count) }
    let left = samples.filter { (bestFeature == 0 ? $0.x : $0.y) <= bestThreshold }
    let right = samples.filter { (bestFeature == 0 ? $0.x : $0.y) > bestThreshold }
    return .split(feature: bestFeature, threshold: bestThreshold, probability: probability, count: samples.count,
        left: fitTree(left, depth: depth - 1, featureSubsample: featureSubsample, rng: &rng),
        right: fitTree(right, depth: depth - 1, featureSubsample: featureSubsample, rng: &rng))
}

struct RegressionStump {
    var feature: Int
    var threshold: Double
    var left: Double
    var right: Double
    func value(_ point: MLPoint) -> Double { (feature == 0 ? point.x : point.y) <= threshold ? left : right }
    func display() -> [TreeDisplayNode] {
        [TreeDisplayNode(id: 0, parentID: nil, label: "\(feature == 0 ? "x" : "y") ≤ \(number(threshold, digits: 2))", prediction: 0, depth: 0),
         TreeDisplayNode(id: 1, parentID: 0, label: "Δscore \(number(left))", prediction: left >= 0 ? 1 : 0, depth: 1),
         TreeDisplayNode(id: 2, parentID: 0, label: "Δscore \(number(right))", prediction: right >= 0 ? 1 : 0, depth: 1)]
    }
}

extension MLEngine {
    static func cart(_ points: [MLPoint], _ config: MLConfiguration) -> SimulationResult {
        let train = points.filter { !$0.isTest }, depthLimit = min(6, max(1, config.treeDepth))
        var frames: [SimulationFrame] = [], history: [Double] = []
        for depth in 1...depthLimit {
            var rng = SeededRandom(seed: config.seed)
            let tree = fitTree(train, depth: depth, featureSubsample: false, rng: &rng)
            history.append(logLoss(train, probability: tree.probability))
            frames.append(classificationFrame(step: depth, caption: "Fit CART with depth ≤ \(depth)",
                detail: "At each node choose the x/y threshold with greatest weighted Gini decrease. Leaves require at least two samples. This frame is the actual fit with depth limit \(depth); left branches mean ≤ and right branches mean >.",
                points: points, probability: tree.probability, history: history, tree: tree.display(), extra: [MLMetric("Depth limit", Double(depth))]))
        }
        return SimulationResult(frames: frames, summary: "Greedy binary CART with Gini splits and empirical leaf probabilities. Frames compare increasing depth limits; the displayed tree is the fitted classifier. Deeper trees can improve training fit while hurting held-out accuracy.")
    }

    static func forest(_ points: [MLPoint], _ config: MLConfiguration) -> SimulationResult {
        let train = points.filter { !$0.isTest }, count = min(32, max(1, config.treeCount))
        let depth = min(6, max(1, config.treeDepth))
        var rng = SeededRandom(seed: config.seed), trees: [ClassificationTree] = []
        var frames: [SimulationFrame] = [], history: [Double] = []
        for step in 1...count {
            let bootstrap = (0..<train.count).map { _ in train[rng.index(train.count)] }
            trees.append(fitTree(bootstrap, depth: depth, featureSubsample: true, rng: &rng))
            let ensemble = trees
            let probability: (MLPoint) -> Double = { point in average(ensemble.map { $0.probability(point) }) }
            history.append(logLoss(train, probability: probability))
            frames.append(classificationFrame(step: step, caption: "Average \(step) bootstrap tree\(step == 1 ? "" : "s")",
                detail: "Each tree samples \(train.count) training points with replacement and considers one randomly selected feature per split (√2 rounded down). Grid and metrics average all \(step) trees. Diagram: representative tree 1 only, never the whole forest.",
                points: points, probability: probability, history: history, tree: trees[0].display(), extra: [MLMetric("Trees fitted", Double(step))]))
        }
        return SimulationResult(frames: frames, summary: "Random forest with bootstrap samples, per-node feature subsampling, CART Gini splits, and probability averaging. All trees use training samples only. The diagram shows the first tree as a representative, while the grid uses the complete ensemble.")
    }

    static func boosting(_ points: [MLPoint], _ config: MLConfiguration) -> SimulationResult {
        let train = points.filter { !$0.isTest }, learningRate = rate(config)
        let stageCount = min(30, max(3, config.treeCount))
        let base = logit(average(train.map { Double($0.label) }))
        var logits = [Double](repeating: base, count: train.count), stumps: [RegressionStump] = []
        var frames: [SimulationFrame] = [], history: [Double] = []
        for step in 1...stageCount {
            let probabilities = logits.map(sigmoid)
            let residuals = zip(train, probabilities).map { Double($0.0.label) - $0.1 }
            var best = RegressionStump(feature: 0, threshold: 1, left: 0, right: 0), bestError = Double.infinity
            // Fit a least-squares stump to the negative logistic-loss gradient.
            for feature in 0...1 {
                let ordered = train.indices.sorted {
                    let a = feature == 0 ? train[$0].x : train[$0].y
                    let b = feature == 0 ? train[$1].x : train[$1].y
                    return a == b ? train[$0].id < train[$1].id : a < b
                }
                let total = residuals.reduce(0, +), totalSquared = residuals.reduce(0) { $0 + $1 * $1 }
                var leftSum = 0.0, leftSquared = 0.0
                if train.count > 1 {
                    for i in 0..<(ordered.count - 1) {
                        let r = residuals[ordered[i]]
                        leftSum += r; leftSquared += r * r
                        let leftCount = i + 1, rightCount = ordered.count - leftCount
                        let a = feature == 0 ? train[ordered[i]].x : train[ordered[i]].y
                        let b = feature == 0 ? train[ordered[i + 1]].x : train[ordered[i + 1]].y
                        guard a < b else { continue }
                        let error = leftSquared - leftSum * leftSum / Double(leftCount)
                            + totalSquared - leftSquared - pow(total - leftSum, 2) / Double(rightCount)
                        if error < bestError {
                            bestError = error
                            best = RegressionStump(feature: feature, threshold: (a + b) / 2, left: 0, right: 0)
                        }
                    }
                }
            }
            var leftGradient = 0.0, rightGradient = 0.0, leftHessian = 0.0, rightHessian = 0.0
            for i in train.indices {
                let isLeft = (best.feature == 0 ? train[i].x : train[i].y) <= best.threshold
                let hessian = probabilities[i] * (1 - probabilities[i])
                if isLeft { leftGradient += residuals[i]; leftHessian += hessian }
                else { rightGradient += residuals[i]; rightHessian += hessian }
            }
            // Newton leaf values are bounded only to keep tiny leaves numerically stable.
            best.left = bounded(leftGradient / max(1e-6, leftHessian), -5, 5)
            best.right = bounded(rightGradient / max(1e-6, rightHessian), -5, 5)
            stumps.append(best)
            for i in train.indices { logits[i] += learningRate * best.value(train[i]) }
            let fitted = stumps
            let probability: (MLPoint) -> Double = { point in sigmoid(base + learningRate * fitted.reduce(0) { $0 + $1.value(point) }) }
            history.append(logLoss(train, probability: probability))
            frames.append(classificationFrame(step: step, caption: "Add residual stump \(step)",
                detail: "Compute residual y−p, split a stump by residual squared error, and add η times its Newton leaf scores to the running logit. η = \(number(learningRate)). Diagram: newest stump only; the grid includes all \(step) stumps.",
                points: points, probability: probability, history: history, tree: best.display(), extra: [MLMetric("Stumps fitted", Double(step))]))
        }
        return SimulationResult(frames: frames, summary: "Binary logistic gradient boosting with regression stumps: each new learner fits the current residuals, then contributes a shrunken Newton score. The diagram shows the latest stump; probability and held-out metrics use their cumulative sum.")
    }
}
