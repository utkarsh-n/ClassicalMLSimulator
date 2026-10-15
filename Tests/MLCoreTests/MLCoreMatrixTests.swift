import Foundation
import XCTest
@testable import MLCore

final class MLCoreMatrixTests: XCTestCase {
    private let classifiers: Set<MLAlgorithm> = [.logisticRegression, .knn, .decisionTree, .randomForest, .gradientBoosting, .naiveBayes, .svm]

    private func datasets(_ algorithm: MLAlgorithm) -> [DatasetKind] {
        switch algorithm {
        case .linearRegression, .polynomialRegression: [.linear, .curve]
        case .logisticRegression, .knn, .decisionTree, .randomForest, .gradientBoosting, .naiveBayes, .svm: [.blobs, .moons, .circles, .xor]
        case .kMeans, .dbscan: [.blobs, .moons, .circles, .anisotropic]
        case .pca: [.anisotropic, .blobs]
        case .monteCarlo: [.blobs]
        }
    }

    private func profile(_ index: Int) -> MLConfiguration {
        var config = MLConfiguration()
        if index == 0 {
            config.seed = 0; config.sampleCount = 30; config.noise = 0
            config.iterations = 10; config.learningRate = 0.02; config.regularization = 0
            config.neighborCount = 1; config.clusters = 2; config.treeDepth = 1
            config.treeCount = 3; config.degree = 1; config.epsilon = 0.05; config.minPoints = 2
        } else if index == 2 {
            config.seed = .max; config.sampleCount = 180; config.noise = 0.22
            config.iterations = 40; config.learningRate = 0.8; config.regularization = 0.2
            config.neighborCount = 21; config.clusters = 6; config.treeDepth = 6
            config.treeCount = 30; config.degree = 6; config.epsilon = 0.3; config.minPoints = 12
        }
        return config
    }

    private func metric(_ name: String, _ frame: SimulationFrame) -> Double? {
        frame.metrics.first { $0.name == name }?.value
    }

    // Count value payload only: this deliberately excludes Array/String/object overhead.
    private func numericPayload(_ result: SimulationResult) -> Int {
        result.frames.reduce(0) { total, frame in
            total + (frame.points.count + frame.centers.count + frame.curve.count) * MemoryLayout<MLPoint>.stride
                + frame.grid.count * MemoryLayout<GridCell>.stride
                + frame.vectors.count * MemoryLayout<MLLine>.stride
                + frame.history.count * MemoryLayout<Double>.stride
                + frame.tree.count * MemoryLayout<TreeDisplayNode>.stride
        }
    }

    private func validate(_ result: SimulationResult, algorithm: MLAlgorithm, points: [MLPoint], config: MLConfiguration,
                          context: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertFalse(result.frames.isEmpty, context, file: file, line: line)
        XCTAssertLessThanOrEqual(result.frames.count, 40, context, file: file, line: line)
        XCTAssertFalse(result.summary.isEmpty, context, file: file, line: line)
        let source = Array(points.prefix(320))
        let sourceIDs = Set(source.map(\.id)), uniqueSourceIDs = sourceIDs.count == source.count
        let expectedTests = source.filter(\.isTest).count
        let isSampling = algorithm == .monteCarlo
        let pointCap = isSampling ? min(10_000, max(20, config.sampleCount)) : 320
        var previousStep = -1
        for frame in result.frames {
            XCTAssertGreaterThanOrEqual(frame.step, previousStep, context, file: file, line: line)
            previousStep = frame.step
            XCTAssertLessThanOrEqual(frame.points.count, pointCap, context, file: file, line: line)
            XCTAssertEqual(Set(frame.points.map(\.id)).count, frame.points.count, context, file: file, line: line)
            XCTAssertEqual(Set(frame.centers.map(\.id)).count, frame.centers.count, context, file: file, line: line)
            XCTAssertEqual(Set(frame.curve.map(\.id)).count, frame.curve.count, context, file: file, line: line)
            XCTAssertEqual(Set(frame.metrics.map(\.id)).count, frame.metrics.count, context, file: file, line: line)
            XCTAssertEqual(Set(frame.tree.map(\.id)).count, frame.tree.count, context, file: file, line: line)
            XCTAssertTrue(frame.points.allSatisfy { $0.x.isFinite && $0.y.isFinite && (0...1).contains($0.x) && (0...1).contains($0.y) }, context, file: file, line: line)
            XCTAssertTrue((frame.centers + frame.curve).allSatisfy { $0.x.isFinite && $0.y.isFinite }, context, file: file, line: line)
            XCTAssertTrue(frame.metrics.allSatisfy { $0.value.isFinite }, context, file: file, line: line)
            XCTAssertTrue(frame.vectors.allSatisfy { $0.x1.isFinite && $0.y1.isFinite && $0.x2.isFinite && $0.y2.isFinite }, context, file: file, line: line)
            XCTAssertTrue(frame.history.allSatisfy(\.isFinite), context, file: file, line: line)
            XCTAssertLessThanOrEqual(frame.history.count, 40, context, file: file, line: line)
            XCTAssertLessThanOrEqual(frame.tree.count, 127, context, file: file, line: line)
            for node in frame.tree {
                XCTAssertTrue(node.depth >= 0 && node.depth <= 6, context, file: file, line: line)
                if let parentID = node.parentID {
                    let parent = frame.tree.first { $0.id == parentID }
                    XCTAssertEqual(parent?.depth, node.depth - 1, context, file: file, line: line)
                } else { XCTAssertEqual(node.depth, 0, context, file: file, line: line) }
            }
            if !frame.grid.isEmpty {
                XCTAssertEqual(frame.grid.count, 576, context, file: file, line: line)
                for (index, cell) in frame.grid.enumerated() {
                    XCTAssertEqual(cell.x, (Double(index % 24) + 0.5) / 24, accuracy: 1e-12, context, file: file, line: line)
                    XCTAssertEqual(cell.y, (Double(index / 24) + 0.5) / 24, accuracy: 1e-12, context, file: file, line: line)
                    XCTAssertTrue(cell.confidence.isFinite && (0...1).contains(cell.confidence), context, file: file, line: line)
                }
            }
            if !isSampling {
                XCTAssertEqual(frame.points.filter(\.isTest).count, expectedTests, context, file: file, line: line)
                if uniqueSourceIDs { XCTAssertTrue(Set(frame.points.map(\.id)).isSubset(of: sourceIDs), context, file: file, line: line) }
                if !source.contains(where: { !$0.isTest }) { XCTAssertTrue(frame.metrics.isEmpty, context, file: file, line: line); continue }
                if classifiers.contains(algorithm) {
                    XCTAssertTrue(frame.points.allSatisfy { $0.label == 0 || $0.label == 1 }, context, file: file, line: line)
                    for name in ["Train accuracy", "Test accuracy"] {
                        if let value = metric(name, frame) { XCTAssertTrue((0...100).contains(value), context, file: file, line: line) }
                    }
                    XCTAssertNotNil(metric("Train accuracy", frame), context, file: file, line: line)
                    if expectedTests > 0 { XCTAssertNotNil(metric("Test accuracy", frame), context, file: file, line: line) }
                    if algorithm == .knn { XCTAssertEqual(Double(frame.points.filter { !$0.isTest }.count), metric("Memorized", frame), context, file: file, line: line) }
                }
            }
            let historyMetric: String
            switch algorithm {
            case .linearRegression, .polynomialRegression: historyMetric = "Train MSE"
            case .svm: historyMetric = "Train hinge loss"
            case .kMeans: historyMetric = "Train inertia"
            case .dbscan: historyMetric = "Train noise"
            case .pca: historyMetric = "PC1 variance"
            case .monteCarlo: historyMetric = "π estimate"
            default: historyMetric = "Train log loss"
            }
            if let last = frame.history.last { XCTAssertEqual(last, metric(historyMetric, frame), context, file: file, line: line) }
        }
        if let last = result.frames.last {
            XCTAssertEqual(last.points.count, isSampling ? pointCap : source.count, context, file: file, line: line)
        }
        XCTAssertLessThan(numericPayload(result), 16 * 1_024 * 1_024, context, file: file, line: line)
    }

    func testSupportedDatasetAndUIExtremeMatrix() {
        let started = Date()
        var configurations = 0, engineRuns = 0, maximumSeconds = 0.0, largestPayload = 0
        var slowest = ""
        for algorithm in MLAlgorithm.allCases {
            for dataset in datasets(algorithm) {
                for index in 0..<3 {
                    var config = profile(index)
                    if algorithm == .monteCarlo { config.sampleCount = [100, 1_000, 2_000][index] }
                    let points = DatasetGenerator.generate(dataset, configuration: config)
                    let context = "\(algorithm.rawValue)/\(dataset.rawValue)/profile\(index)"
                    let runStart = Date()
                    let first = MLEngine.run(algorithm: algorithm, points: points, configuration: config)
                    let elapsed = Date().timeIntervalSince(runStart)
                    if elapsed > maximumSeconds { maximumSeconds = elapsed; slowest = context }
                    XCTAssertLessThan(elapsed, 15, context)
                    let second = MLEngine.run(algorithm: algorithm, points: points, configuration: config)
                    XCTAssertEqual(first.frames, second.frames, context)
                    XCTAssertEqual(first.summary, second.summary, context)
                    validate(first, algorithm: algorithm, points: points, config: config, context: context)
                    largestPayload = max(largestPayload, numericPayload(first))
                    engineRuns += 2; configurations += 1
                }
            }
        }
        XCTAssertEqual(configurations, 129); XCTAssertEqual(engineRuns, 258)
        print("ML_STRESS_MATRIX configurations=\(configurations) runs=\(engineRuns) seconds=\(Date().timeIntervalSince(started)) max_run_seconds=\(maximumSeconds) slowest=\(slowest) max_numeric_payload_bytes=\(largestPayload)")
    }

    func testMalformedDegenerateTiedAndOverCapMatrix() {
        let empty: [MLPoint] = []
        let allTest = (0..<5).map { MLPoint(id: $0, x: 0.2, y: 0.8, label: $0 % 2, isTest: true) }
        let single = [MLPoint(id: .max, x: .nan, y: .infinity, label: .min)]
        let tied = (0..<24).map { MLPoint(id: $0, x: 0.5, y: 0.5, label: $0 % 2, isTest: $0 % 5 == 4) }
        let malformed = (0..<24).map { i -> MLPoint in
            let coordinates: [Double] = [.nan, .infinity, -.infinity, -3, 4, 0.5]
            let identifiers = [Int.max, Int.min, 0, 0]
            return MLPoint(id: identifiers[i % 4], x: coordinates[i % 6], y: coordinates[(i + 2) % 6], label: i % 2 == 0 ? .max : .min, isTest: i % 5 == 4)
        }
        let overCap = (0..<400).map { i in MLPoint(id: i, x: Double(i % 20) / 19, y: Double(i / 20) / 19, label: i % 3, isTest: i % 5 == 4) }
        let cases = [empty, allTest, single, tied, malformed, overCap]
        let names = ["empty", "allTest", "singleInvalid", "tied", "malformedIDs", "over320"]
        var config = MLConfiguration()
        config.seed = .max; config.sampleCount = .max; config.noise = .nan
        config.iterations = .max; config.learningRate = .infinity; config.regularization = .nan
        config.neighborCount = .max; config.treeDepth = .max; config.treeCount = .max
        config.clusters = .max; config.degree = .max; config.epsilon = .nan; config.minPoints = .max
        let started = Date()
        var runs = 0, largestPayload = 0
        for algorithm in MLAlgorithm.allCases {
            for (index, points) in cases.enumerated() {
                let context = "\(algorithm.rawValue)/\(names[index])"
                let first = MLEngine.run(algorithm: algorithm, points: points, configuration: config)
                let second = MLEngine.run(algorithm: algorithm, points: points, configuration: config)
                XCTAssertEqual(first.frames, second.frames, context)
                validate(first, algorithm: algorithm, points: points, config: config, context: context)
                largestPayload = max(largestPayload, numericPayload(first)); runs += 2
            }
        }
        XCTAssertEqual(runs, 156)
        print("ML_STRESS_MALFORMED cases=78 runs=\(runs) seconds=\(Date().timeIntervalSince(started)) max_numeric_payload_bytes=\(largestPayload)")
    }
}
