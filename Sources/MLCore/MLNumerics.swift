import Foundation

// SplitMix64 makes identical seeds reproducible across platforms and launches.
struct SeededRandom {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func uniform() -> Double { Double(next() >> 11) / 9_007_199_254_740_992 }
    mutating func index(_ count: Int) -> Int { Int(next() % UInt64(max(1, count))) }
    mutating func normal() -> Double {
        sqrt(-2 * log(max(1e-12, uniform()))) * cos(2 * .pi * uniform())
    }
}

func bounded(_ value: Double, _ lower: Double, _ upper: Double, fallback: Double = 0) -> Double {
    min(upper, max(lower, value.isFinite ? value : fallback))
}
func unit(_ value: Double) -> Double { bounded(value, 0, 1) }
func sigmoid(_ value: Double) -> Double {
    let x = bounded(value, -40, 40)
    return 1 / (1 + exp(-x))
}
func logit(_ p: Double) -> Double {
    let p = bounded(p, 1e-6, 1 - 1e-6, fallback: 0.5)
    return log(p / (1 - p))
}
func squaredDistance(_ a: MLPoint, _ b: MLPoint) -> Double {
    let dx = a.x - b.x, dy = a.y - b.y
    return dx * dx + dy * dy
}
func average(_ numbers: [Double]) -> Double {
    numbers.isEmpty ? 0 : numbers.reduce(0, +) / Double(numbers.count)
}
func steps(_ config: MLConfiguration) -> Int { min(40, max(1, config.iterations)) }
func rate(_ config: MLConfiguration) -> Double {
    bounded(config.learningRate, 0.001, 1, fallback: 0.15)
}
func ridge(_ config: MLConfiguration) -> Double {
    bounded(config.regularization, 0, 2, fallback: 0.02)
}
func normalized(_ points: [MLPoint], binary: Bool = false) -> [MLPoint] {
    let source = Array(points.prefix(320))
    let reservedIDs = Set(source.map(\.id))
    var seenIDs = Set<Int>(), nextReplacementID = 0
    return source.map { point in
        var id = point.id
        if !seenIDs.insert(id).inserted {
            // Preserve all original unique IDs, including ones occurring later in the input.
            // At most 320 are reserved, so this small nonnegative search cannot overflow.
            while reservedIDs.contains(nextReplacementID) || seenIDs.contains(nextReplacementID) { nextReplacementID += 1 }
            id = nextReplacementID
            seenIDs.insert(id); nextReplacementID += 1
        }
        return MLPoint(id: id, x: unit(point.x), y: unit(point.y),
                       label: binary ? (point.label > 0 ? 1 : 0) : point.label, isTest: point.isTest)
    }
}

func decisionGrid(_ probability: (MLPoint) -> Double) -> [GridCell] {
    let side = 24
    return (0..<(side * side)).map { i in
        let x = (Double(i % side) + 0.5) / Double(side)
        let y = (Double(i / side) + 0.5) / Double(side)
        let p = unit(probability(MLPoint(id: -1, x: x, y: y)))
        return GridCell(x: x, y: y, label: p >= 0.5 ? 1 : 0, confidence: max(p, 1 - p))
    }
}

func classificationMetrics(_ points: [MLPoint], probability: (MLPoint) -> Double) -> [MLMetric] {
    let train = points.filter { !$0.isTest }, test = points.filter(\.isTest)
    func accuracy(_ samples: [MLPoint]) -> Double {
        guard !samples.isEmpty else { return 0 }
        return 100 * Double(samples.filter { (probability($0) >= 0.5 ? 1 : 0) == $0.label }.count) / Double(samples.count)
    }
    var result = [MLMetric("Train accuracy", accuracy(train), unit: "%")]
    if !test.isEmpty { result.append(MLMetric("Test accuracy", accuracy(test), unit: "%")) }
    return result
}

func logLoss(_ samples: [MLPoint], probability: (MLPoint) -> Double) -> Double {
    average(samples.map { point in
        let p = bounded(probability(point), 1e-9, 1 - 1e-9, fallback: 0.5)
        return point.label == 1 ? -log(p) : -log(1 - p)
    })
}

func classificationFrame(step: Int, caption: String, detail: String, points: [MLPoint],
                         probability: (MLPoint) -> Double, history: [Double],
                         tree: [TreeDisplayNode] = [], centers: [MLPoint] = [],
                         extra: [MLMetric] = [], customLoss: MLMetric? = nil) -> SimulationFrame {
    var metrics = classificationMetrics(points, probability: probability)
    metrics.append(customLoss ?? MLMetric("Train log loss", logLoss(points.filter { !$0.isTest }, probability: probability)))
    metrics.append(contentsOf: extra)
    return SimulationFrame(step: step, caption: caption, detail: detail, metrics: metrics,
                           points: points, centers: centers, grid: decisionGrid(probability), tree: tree, history: history)
}

func number(_ value: Double, digits: Int = 3) -> String {
    String(format: "%.*f", digits, value)
}
