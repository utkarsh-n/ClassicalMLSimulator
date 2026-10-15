import Foundation
import XCTest
@testable import MLCore

final class MLCoreNumericalAuditTests: XCTestCase {
    private func metric(_ name: String, _ frame: SimulationFrame) -> Double { frame.metrics.first { $0.name == name }?.value ?? .nan }
    private func probability(_ cell: GridCell) -> Double { cell.label == 1 ? cell.confidence : 1 - cell.confidence }
    private func gridCell(_ point: MLPoint, _ frame: SimulationFrame) -> GridCell {
        frame.grid[Int(floor(point.y * 24)) * 24 + Int(floor(point.x * 24))]
    }
    private var alignedSamples: [MLPoint] {
        let xs = [2, 7, 16, 21], ys = [3, 8, 14, 20]
        return (0..<16).map { i in
            let x = xs[i % 4], y = ys[i / 4]
            return MLPoint(id: i, x: (Double(x) + 0.5) / 24, y: (Double(y) + 0.5) / 24,
                           label: x < 12 ? 0 : 1, isTest: i % 5 == 4)
        }
    }

    func testClassifierTelemetryMatchesIndependentGridPredictions() {
        let points = alignedSamples
        for algorithm in [MLAlgorithm.logisticRegression, .knn, .decisionTree, .randomForest, .gradientBoosting, .naiveBayes, .svm] {
            let frame = MLEngine.run(algorithm: algorithm, points: points, configuration: MLConfiguration()).frames.last!
            for isTest in [false, true] {
                let group = points.filter { $0.isTest == isTest }
                let correct = group.filter { gridCell($0, frame).label == $0.label }.count
                XCTAssertEqual(metric(isTest ? "Test accuracy" : "Train accuracy", frame), 100 * Double(correct) / Double(group.count), accuracy: 1e-12, algorithm.rawValue)
            }
            let train = points.filter { !$0.isTest }
            if algorithm == .svm {
                let loss = train.reduce(0.0) { result, point in
                    let cell = gridCell(point, frame), p = probability(cell)
                    let score = log(p / (1 - p)) / 3
                    return result + max(0, 1 - (point.label == 1 ? 1.0 : -1.0) * score)
                } / Double(train.count)
                XCTAssertEqual(metric("Train hinge loss", frame), loss, accuracy: 1e-10)
            } else {
                let loss = train.reduce(0.0) { result, point in
                    let p = min(1 - 1e-9, max(1e-9, probability(gridCell(point, frame))))
                    return result - (point.label == 1 ? log(p) : log(1 - p))
                } / Double(train.count)
                XCTAssertEqual(metric("Train log loss", frame), loss, accuracy: 1e-10, algorithm.rawValue)
            }
        }
    }

    func testLogisticFirstGradientAgainstAnalyticalWeights() {
        let points = [MLPoint(id: 0, x: 0.1, y: 0.2, label: 0), MLPoint(id: 1, x: 0.2, y: 0.8, label: 0),
                      MLPoint(id: 2, x: 0.7, y: 0.3, label: 1), MLPoint(id: 3, x: 0.9, y: 0.9, label: 1)]
        var config = MLConfiguration(); config.iterations = 1; config.learningRate = 0.2
        let frame = MLEngine.run(algorithm: .logisticRegression, points: points, configuration: config).frames[0]
        // Balanced prior gives b=0. The first averaged gradients are gx=-.325, gy=-.05.
        for cell in frame.grid {
            let score = 0.065 * (2 * cell.x - 1) + 0.01 * (2 * cell.y - 1)
            XCTAssertEqual(probability(cell), 1 / (1 + exp(-score)), accuracy: 1e-12)
        }
    }

    func testSVMPositiveRegularizationAgainstAnalyticalWeights() {
        let points = [MLPoint(id: 0, x: 0.25, y: 0.5, label: 0), MLPoint(id: 1, x: 0.75, y: 0.5, label: 1)]
        var config = MLConfiguration(); config.iterations = 2; config.learningRate = 0.2; config.regularization = 0.2
        let frame = MLEngine.run(algorithm: .svm, points: points, configuration: config).frames.last!
        // w1=.1; w2=.1 + .2(.5 − .2·.1) = .196. No intercept/vertical gradient.
        for cell in frame.grid {
            let score = 3 * 0.196 * (2 * cell.x - 1)
            XCTAssertEqual(probability(cell), 1 / (1 + exp(-score)), accuracy: 1e-12)
        }
    }

    func testKNNMatchesBruteForceVotesIncludingDistanceAndVoteTies() {
        let qx = 12.5 / 24.0, qy = 11.5 / 24.0
        let points = [MLPoint(id: 7, x: qx - 0.25, y: qy, label: 0), MLPoint(id: 3, x: qx + 0.25, y: qy, label: 1),
                      MLPoint(id: 9, x: qx, y: 0.95, label: 1), MLPoint(id: 10, x: 0.02, y: 0.05, label: 0)]
        for k in [1, 2, 3, 99] {
            var config = MLConfiguration(); config.neighborCount = k
            let frame = MLEngine.run(algorithm: .knn, points: points, configuration: config).frames.last!
            for cell in frame.grid {
                let ordered = points.sorted { a, b in
                    let a2 = pow(a.x - cell.x, 2) + pow(a.y - cell.y, 2)
                    let b2 = pow(b.x - cell.x, 2) + pow(b.y - cell.y, 2)
                    return a2 == b2 ? a.id < b.id : a2 < b2
                }.prefix(min(points.count, k))
                let expected = Double(ordered.filter { $0.label == 1 }.count) / Double(ordered.count)
                XCTAssertEqual(probability(cell), expected, accuracy: 1e-12)
                XCTAssertEqual(cell.label, expected >= 0.5 ? 1 : 0)
            }
            if k == 1 { XCTAssertEqual(gridCell(MLPoint(id: -1, x: qx, y: qy), frame).label, 1, "Exact distance ties choose lower sample ID") }
        }
    }

    func testNaiveBayesMatchesDirectGaussianDensityReference() {
        let points = alignedSamples, train = points.filter { !$0.isTest }
        let frame = MLEngine.run(algorithm: .naiveBayes, points: points, configuration: MLConfiguration()).frames.last!
        func density(_ x: Double, _ values: [Double]) -> Double {
            let mean = values.reduce(0, +) / Double(values.count)
            let variance = max(1e-4, values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count))
            return exp(-pow(x - mean, 2) / (2 * variance)) / sqrt(2 * .pi * variance)
        }
        for cell in frame.grid {
            let likelihoods = (0...1).map { label -> Double in
                let group = train.filter { $0.label == label }
                return Double(group.count) / Double(train.count) * density(cell.x, group.map(\.x)) * density(cell.y, group.map(\.y))
            }
            let expected = likelihoods[1] / (likelihoods[0] + likelihoods[1])
            XCTAssertEqual(probability(cell), expected, accuracy: 1e-12)
        }
    }

    func testRegressionMSEMatchesCurveAgainstTargetsAtEveryRecordedState() {
        let points = (0..<30).map { i -> MLPoint in
            let x = Double(i * 2) / 60
            return MLPoint(id: i, x: x, y: 0.18 + 0.6 * pow(2 * x - 1, 2) + (i % 2 == 0 ? 0.01 : -0.01), isTest: i % 5 == 4)
        }
        for degree in [1, 2, 6] {
            for eta in [0.02, 0.8] {
                var config = MLConfiguration(); config.degree = degree; config.learningRate = eta; config.regularization = 0.2
                let result = MLEngine.run(algorithm: .polynomialRegression, points: points, configuration: config)
                for frame in result.frames {
                    for isTest in [false, true] {
                        let indices = points.indices.filter { points[$0].isTest == isTest }
                        let expected = indices.reduce(0.0) { sum, i in sum + pow(frame.curve[i * 2].y - points[i].y, 2) } / Double(indices.count)
                        XCTAssertEqual(metric(isTest ? "Test MSE" : "Train MSE", frame), expected, accuracy: max(1e-12, abs(expected) * 1e-12))
                    }
                }
            }
        }
    }

    func testHeldOutCoordinatesAndLabelsDoNotChangeAnyFitAcrossSeeds() {
        let started = Date()
        var runs = 0
        for seed: UInt64 in [0, .max] {
            var config = MLConfiguration(); config.seed = seed; config.sampleCount = 60
            let source = DatasetGenerator.generate(.blobs, configuration: config)
            var changed = source
            for i in changed.indices where changed[i].isTest {
                changed[i].x = 1 - changed[i].x; changed[i].y = 1 - changed[i].y; changed[i].label = changed[i].label == 0 ? 1 : 0
            }
            for algorithm in MLAlgorithm.allCases where algorithm != .monteCarlo {
                let a = MLEngine.run(algorithm: algorithm, points: source, configuration: config)
                let b = MLEngine.run(algorithm: algorithm, points: changed, configuration: config)
                XCTAssertEqual(a.frames.count, b.frames.count, algorithm.rawValue)
                for (first, second) in zip(a.frames, b.frames) {
                    XCTAssertEqual(first.grid, second.grid, algorithm.rawValue)
                    XCTAssertEqual(first.centers, second.centers, algorithm.rawValue)
                    XCTAssertEqual(first.curve, second.curve, algorithm.rawValue)
                    XCTAssertEqual(first.tree, second.tree, algorithm.rawValue)
                    XCTAssertEqual(first.vectors, second.vectors, algorithm.rawValue)
                    XCTAssertEqual(first.history, second.history, algorithm.rawValue)
                    XCTAssertEqual(first.points.filter { !$0.isTest }, second.points.filter { !$0.isTest }, algorithm.rawValue)
                    XCTAssertEqual(first.metrics.filter { !$0.name.lowercased().contains("test") }, second.metrics.filter { !$0.name.lowercased().contains("test") }, algorithm.rawValue)
                }
                runs += 2
            }
        }
        XCTAssertEqual(runs, 48)
        print("ML_STRESS_ISOLATION pairs=24 runs=\(runs) seconds=\(Date().timeIntervalSince(started))")
    }

    func testDBSCANReclassifiesBorderNoiseAndPredictsFromCoreOnly() {
        let points = [MLPoint(id: 0, x: 0.187, y: 0.11),
                      MLPoint(id: 1, x: 0.10, y: 0.10), MLPoint(id: 2, x: 0.11, y: 0.10),
                      MLPoint(id: 3, x: 0.10, y: 0.11), MLPoint(id: 4, x: 0.11, y: 0.11),
                      MLPoint(id: 5, x: 0.8, y: 0.8), MLPoint(id: 6, x: 0.255, y: 0.11, isTest: true)]
        var config = MLConfiguration(); config.epsilon = 0.08; config.minPoints = 4
        let result = MLEngine.run(algorithm: .dbscan, points: points, configuration: config)
        XCTAssertEqual(result.frames.first?.points.first?.label, -1)
        let final = result.frames.last!
        XCTAssertEqual(final.points.first?.label, 0)
        XCTAssertEqual(final.centers.count, 4, "The border sample must not become a core")
        XCTAssertEqual(final.points.first { $0.id == 6 }?.label, -1, "A held-out query near only a border is outside all fitted core neighborhoods")
        XCTAssertEqual(metric("Train noise", final), 100.0 / 6, accuracy: 1e-12)
    }

    func testConstantPCAHasZeroVarianceWithoutInventedDirectionLength() {
        let points = (0..<10).map { MLPoint(id: $0, x: 0.2, y: 0.8, isTest: $0 % 5 == 4) }
        let frame = MLEngine.run(algorithm: .pca, points: points, configuration: MLConfiguration()).frames.last!
        XCTAssertEqual(metric("PC1 variance", frame), 0)
        XCTAssertEqual(metric("Train reconstruction MSE", frame), 0, accuracy: 1e-25)
        XCTAssertEqual(metric("Test reconstruction MSE", frame), 0, accuracy: 1e-25)
        for vector in frame.vectors {
            XCTAssertEqual(vector.x1, vector.x2, accuracy: 1e-12)
            XCTAssertEqual(vector.y1, vector.y2, accuracy: 1e-12)
        }
    }
}
