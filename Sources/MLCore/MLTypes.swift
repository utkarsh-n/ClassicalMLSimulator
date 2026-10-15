import Foundation

public enum MLAlgorithm: String, CaseIterable, Identifiable, Sendable, Codable {
    case linearRegression, polynomialRegression, logisticRegression, knn
    case decisionTree, randomForest, gradientBoosting, naiveBayes, svm
    case kMeans, dbscan, pca, monteCarlo
    public var id: String { rawValue }
}

public enum DatasetKind: String, CaseIterable, Identifiable, Sendable, Codable {
    case blobs, moons, circles, xor, linear, curve, anisotropic
    public var id: String { rawValue }
}

public struct MLPoint: Identifiable, Sendable, Equatable, Codable {
    public var id: Int
    public var x: Double
    public var y: Double
    public var label: Int
    public var isTest: Bool
    public init(id: Int, x: Double, y: Double, label: Int = 0, isTest: Bool = false) {
        self.id = id; self.x = x; self.y = y; self.label = label; self.isTest = isTest
    }
}

public struct MLConfiguration: Sendable, Equatable, Codable {
    public var sampleCount = 90
    public var noise = 0.08
    public var seed: UInt64 = 42
    public var iterations = 40
    public var learningRate = 0.15
    public var neighborCount = 5
    public var clusters = 3
    public var treeDepth = 3
    public var treeCount = 12
    public var degree = 3
    public var epsilon = 0.14
    public var minPoints = 4
    public var regularization = 0.02
    public init() {}
}

public struct MLMetric: Identifiable, Sendable, Equatable {
    public var name: String
    public var value: Double
    public var unit: String
    public var id: String { name }
    public init(_ name: String, _ value: Double, unit: String = "") {
        self.name = name; self.value = value; self.unit = unit
    }
}

public struct GridCell: Sendable, Equatable {
    public var x: Double
    public var y: Double
    public var label: Int
    public var confidence: Double
    public init(x: Double, y: Double, label: Int, confidence: Double = 1) {
        self.x = x; self.y = y; self.label = label; self.confidence = confidence
    }
}

public struct TreeDisplayNode: Identifiable, Sendable, Equatable {
    public var id: Int
    public var parentID: Int?
    public var label: String
    public var prediction: Int
    public var depth: Int
    public init(id: Int, parentID: Int?, label: String, prediction: Int, depth: Int) {
        self.id = id; self.parentID = parentID; self.label = label
        self.prediction = prediction; self.depth = depth
    }
}

public struct MLLine: Sendable, Equatable {
    public var x1: Double; public var y1: Double
    public var x2: Double; public var y2: Double
    public var label: Int
    public init(x1: Double, y1: Double, x2: Double, y2: Double, label: Int = 0) {
        self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2; self.label = label
    }
}

public struct SimulationFrame: Sendable, Equatable {
    public var step: Int
    public var caption: String
    public var detail: String
    public var metrics: [MLMetric]
    public var points: [MLPoint]
    public var centers: [MLPoint]
    public var curve: [MLPoint]
    public var grid: [GridCell]
    public var tree: [TreeDisplayNode]
    public var vectors: [MLLine]
    public var history: [Double]
    public init(step: Int, caption: String, detail: String, metrics: [MLMetric] = [],
                points: [MLPoint] = [], centers: [MLPoint] = [], curve: [MLPoint] = [],
                grid: [GridCell] = [], tree: [TreeDisplayNode] = [], vectors: [MLLine] = [], history: [Double] = []) {
        self.step = step; self.caption = caption; self.detail = detail; self.metrics = metrics
        self.points = points; self.centers = centers; self.curve = curve; self.grid = grid
        self.tree = tree; self.vectors = vectors; self.history = history
    }
}

public struct SimulationResult: Sendable {
    public var frames: [SimulationFrame]
    public var summary: String
    public init(frames: [SimulationFrame], summary: String) { self.frames = frames; self.summary = summary }
}
