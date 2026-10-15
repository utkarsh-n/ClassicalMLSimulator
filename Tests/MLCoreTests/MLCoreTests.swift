import XCTest
@testable import MLCore

final class MLCoreTests: XCTestCase {
    private func value(_ name: String, in frame: SimulationFrame) -> Double {
        frame.metrics.first { $0.name == name }?.value ?? .nan
    }

    func testDatasetsAreDeterministicNormalizedAndHeldOut() {
        var config = MLConfiguration(); config.sampleCount = 100
        for kind in DatasetKind.allCases {
            let first = DatasetGenerator.generate(kind, configuration: config)
            XCTAssertEqual(first, DatasetGenerator.generate(kind, configuration: config), kind.rawValue)
            XCTAssertEqual(first.count, 100)
            XCTAssertEqual(first.filter(\.isTest).count, 20)
            XCTAssertEqual(Set(first.map(\.id)).count, first.count)
            XCTAssertTrue(first.allSatisfy { (0...1).contains($0.x) && (0...1).contains($0.y) })
            var different = config; different.seed = 43
            XCTAssertNotEqual(first, DatasetGenerator.generate(kind, configuration: different))
        }
    }

    func testLinearRegressionLearnsKnownLineAndImprovesLoss() {
        var config = MLConfiguration(); config.noise = 0; config.regularization = 0
        let points = (0..<100).map { i -> MLPoint in
            let x = Double(i) / 99
            return MLPoint(id: i, x: x, y: 0.15 + 0.7 * x, isTest: i % 5 == 4)
        }
        let result = MLEngine.run(algorithm: .linearRegression, points: points, configuration: config)
        let first = result.frames.first!, last = result.frames.last!
        XCTAssertLessThan(value("Train MSE", in: last), value("Train MSE", in: first) * 0.01)
        XCTAssertLessThan(value("Test MSE", in: last), 0.0001)
        XCTAssertEqual(last.curve.first!.y, 0.15, accuracy: 0.015)
        XCTAssertEqual(last.curve.last!.y, 0.85, accuracy: 0.015)
    }

    func testPolynomialFitCapturesCurvatureThatALineCannot() {
        var config = MLConfiguration(); config.regularization = 0; config.degree = 2; config.learningRate = 0.6
        let points = (0..<100).map { i -> MLPoint in
            let x = Double(i) / 99
            return MLPoint(id: i, x: x, y: 0.15 + 0.7 * pow(2 * x - 1, 2), isTest: i % 5 == 4)
        }
        let line = MLEngine.run(algorithm: .linearRegression, points: points, configuration: config).frames.last!
        let polynomial = MLEngine.run(algorithm: .polynomialRegression, points: points, configuration: config).frames.last!
        XCTAssertLessThan(value("Test MSE", in: polynomial), value("Test MSE", in: line) * 0.03)
        config.degree = 1
        let degreeOne = MLEngine.run(algorithm: .polynomialRegression, points: points, configuration: config).frames.last!
        XCTAssertEqual(value("Test MSE", in: degreeOne), value("Test MSE", in: line), accuracy: 1e-12)
    }

    private var separable: [MLPoint] {
        (0..<100).map { i in
            let label = i % 2
            return MLPoint(id: i, x: label == 0 ? 0.15 + Double(i % 7) * 0.01 : 0.75 + Double(i % 7) * 0.01,
                           y: 0.15 + Double(i % 11) * 0.06, label: label, isTest: i % 5 == 4)
        }
    }

    func testAllClassifiersFitSeparableDataAndEvaluateHeldOut() {
        let algorithms: [MLAlgorithm] = [.logisticRegression, .knn, .decisionTree, .randomForest, .gradientBoosting, .naiveBayes, .svm]
        for algorithm in algorithms {
            let final = MLEngine.run(algorithm: algorithm, points: separable, configuration: MLConfiguration()).frames.last!
            XCTAssertGreaterThanOrEqual(value("Train accuracy", in: final), 95, algorithm.rawValue)
            XCTAssertGreaterThanOrEqual(value("Test accuracy", in: final), 95, algorithm.rawValue)
            XCTAssertEqual(final.grid.count, 576)
        }
    }

    func testHeldOutLabelsCannotAffectFittedClassifiers() {
        var changed = separable
        for i in changed.indices where changed[i].isTest { changed[i].label = 1 - changed[i].label }
        let algorithms: [MLAlgorithm] = [.logisticRegression, .knn, .decisionTree, .randomForest, .gradientBoosting, .naiveBayes, .svm]
        for algorithm in algorithms {
            let a = MLEngine.run(algorithm: algorithm, points: separable, configuration: MLConfiguration()).frames.last!
            let b = MLEngine.run(algorithm: algorithm, points: changed, configuration: MLConfiguration()).frames.last!
            XCTAssertEqual(a.grid, b.grid, algorithm.rawValue)
            XCTAssertEqual(a.tree, b.tree, algorithm.rawValue)
            XCTAssertEqual(value("Train accuracy", in: a), value("Train accuracy", in: b), accuracy: 1e-12)
            XCTAssertEqual(value("Test accuracy", in: a) + value("Test accuracy", in: b), 100, accuracy: 1e-12)
        }
    }

    func testXORRequiresNonlinearModelCapacity() {
        let points = (0..<100).map { i -> MLPoint in
            let corner = i % 4, x = corner % 2 == 0 ? 0.15 : 0.85, y = corner < 2 ? 0.15 : 0.85
            return MLPoint(id: i, x: x + Double(i % 3) * 0.005, y: y + Double(i % 7) * 0.003,
                           label: (corner == 1 || corner == 2) ? 1 : 0, isTest: i % 5 == 4)
        }
        var config = MLConfiguration(); config.neighborCount = 3
        let linear = MLEngine.run(algorithm: .logisticRegression, points: points, configuration: config).frames.last!
        let knn = MLEngine.run(algorithm: .knn, points: points, configuration: config).frames.last!
        XCTAssertLessThanOrEqual(value("Test accuracy", in: linear), 75)
        XCTAssertEqual(value("Test accuracy", in: knn), 100)
    }

    func testSVMHonorsZeroRegularizationExactly() {
        let points = [MLPoint(id: 0, x: 0.25, y: 0.5, label: 0),
                      MLPoint(id: 1, x: 0.75, y: 0.5, label: 1)]
        var config = MLConfiguration()
        config.iterations = 2; config.learningRate = 0.2; config.regularization = 0
        let final = MLEngine.run(algorithm: .svm, points: points, configuration: config).frames.last!
        // Both points stay inside the margin: each update adds η·0.5 = 0.1 to wₓ.
        // With λ = 0 the second weight is exactly 0.2; an implicit λ floor changes it.
        for cell in final.grid {
            let score = 3 * 0.2 * (2 * cell.x - 1)
            let expectedConfidence = 1 / (1 + exp(-abs(score)))
            XCTAssertEqual(cell.confidence, expectedConfidence, accuracy: 1e-12)
        }
    }

    func testTreesHaveRealLearnedThresholdsAndEnsembleProgress() {
        var config = MLConfiguration(); config.treeCount = 7
        let tree = MLEngine.run(algorithm: .decisionTree, points: separable, configuration: config).frames.last!
        XCTAssertEqual(tree.tree.count, 3)
        XCTAssertTrue(tree.tree[0].label.contains("x ≤"))
        XCTAssertTrue(tree.tree.dropFirst().allSatisfy { $0.parentID == 0 })
        let forest = MLEngine.run(algorithm: .randomForest, points: separable, configuration: config)
        XCTAssertEqual(forest.frames.count, 7)
        XCTAssertEqual(value("Trees fitted", in: forest.frames.last!), 7)
        XCTAssertTrue(forest.frames.last!.detail.contains("representative tree 1"))
        let boosting = MLEngine.run(algorithm: .gradientBoosting, points: separable, configuration: config)
        XCTAssertLessThan(value("Train log loss", in: boosting.frames.last!), value("Train log loss", in: boosting.frames.first!))
        XCTAssertEqual(boosting.frames.last!.tree.count, 3)
    }

    func testKMeansRecoversThreeBlobsAndDecreasesInertia() {
        var config = MLConfiguration(); config.noise = 0.025; config.sampleCount = 150
        let points = DatasetGenerator.generate(.blobs, configuration: config)
        let result = MLEngine.run(algorithm: .kMeans, points: points, configuration: config)
        let last = result.frames.last!
        XCTAssertEqual(last.centers.count, 3)
        XCTAssertEqual(Set(last.points.map(\.label)).count, 3)
        XCTAssertLessThan(value("Train inertia", in: last), 1)
        XCTAssertEqual(value("Reassigned", in: last), 0)
        for pair in zip(last.history, last.history.dropFirst()) { XCTAssertLessThanOrEqual(pair.1, pair.0 + 1e-10) }
        XCTAssertTrue(last.metrics.contains { $0.name == "Test inertia" })
    }

    func testDBSCANFindsTwoGroupsAndRejectsIsolatedNoise() {
        var points: [MLPoint] = []
        for i in 0..<20 {
            points.append(MLPoint(id: i, x: 0.2 + Double(i % 5) * 0.01, y: 0.2 + Double(i / 5) * 0.01))
            points.append(MLPoint(id: 20 + i, x: 0.7 + Double(i % 5) * 0.01, y: 0.7 + Double(i / 5) * 0.01))
        }
        points.append(MLPoint(id: 99, x: 0.05, y: 0.95))
        points.append(MLPoint(id: 100, x: 0.21, y: 0.22, isTest: true))
        points.append(MLPoint(id: 101, x: 0.95, y: 0.05, isTest: true))
        var config = MLConfiguration(); config.epsilon = 0.08; config.minPoints = 4
        let last = MLEngine.run(algorithm: .dbscan, points: points, configuration: config).frames.last!
        XCTAssertEqual(value("Clusters", in: last), 2)
        XCTAssertEqual(last.points.first { $0.id == 99 }?.label, -1)
        XCTAssertGreaterThanOrEqual(last.points.first { $0.id == 100 }!.label, 0)
        XCTAssertEqual(last.points.first { $0.id == 101 }?.label, -1)
        XCTAssertEqual(value("Test outside ε", in: last), 50)
        XCTAssertFalse(last.points.contains { $0.label == -2 })
    }

    func testPCARecoversKnownPrincipalAxisAndHeldOutReconstruction() {
        let points = (0..<100).map { i -> MLPoint in
            let t = Double(i) / 99 - 0.5
            return MLPoint(id: i, x: 0.5 + 0.7 * t, y: 0.5 + 0.7 * t, isTest: i % 5 == 4)
        }
        let final = MLEngine.run(algorithm: .pca, points: points, configuration: MLConfiguration()).frames.last!
        XCTAssertEqual(value("PC1 angle", in: final), 45, accuracy: 1e-8)
        XCTAssertEqual(value("PC1 variance", in: final), 100, accuracy: 1e-8)
        XCTAssertLessThan(value("Test reconstruction MSE", in: final), 1e-20)
        XCTAssertEqual(final.vectors.count, 2)
    }

    func testUnsupervisedFitsExcludeHeldOutCoordinates() {
        var config = MLConfiguration(); config.noise = 0.03
        let points = DatasetGenerator.generate(.blobs, configuration: config)
        var changed = points
        for i in changed.indices where changed[i].isTest { changed[i].x = 0.99; changed[i].y = 0.01 }
        for algorithm in [MLAlgorithm.kMeans, .dbscan, .pca] {
            let a = MLEngine.run(algorithm: algorithm, points: points, configuration: config).frames.last!
            let b = MLEngine.run(algorithm: algorithm, points: changed, configuration: config).frames.last!
            XCTAssertEqual(a.centers, b.centers, algorithm.rawValue)
            XCTAssertEqual(a.vectors, b.vectors, algorithm.rawValue)
            XCTAssertEqual(a.grid, b.grid, algorithm.rawValue)
        }
    }

    func testMonteCarloUsesActualCountsAndReproducibleUniformSamples() {
        var config = MLConfiguration(); config.sampleCount = 2_000
        let a = MLEngine.run(algorithm: .monteCarlo, points: [], configuration: config)
        let b = MLEngine.run(algorithm: .monteCarlo, points: [], configuration: config)
        XCTAssertEqual(a.frames, b.frames)
        XCTAssertEqual(a.frames.count, 40)
        let last = a.frames.last!, inside = last.points.filter { $0.x * $0.x + $0.y * $0.y <= 1 }.count
        XCTAssertEqual(last.points.count, 2_000)
        XCTAssertEqual(last.points.filter { $0.label == 0 }.count, inside)
        XCTAssertEqual(value("π estimate", in: last), 4 * Double(inside) / 2_000, accuracy: 1e-12)
        XCTAssertLessThan(value("Absolute error", in: last), 0.15)
        XCTAssertTrue(zip(a.frames, a.frames.dropFirst()).allSatisfy { $0.points.count < $1.points.count })
    }

    func testAllModelsRemainFiniteBoundedAndDeterministicOnDegenerateData() {
        var config = MLConfiguration(); config.iterations = 200; config.sampleCount = 20
        let identical = (0..<10).map { MLPoint(id: $0, x: 0.5, y: 0.5, label: $0 % 2, isTest: $0 % 5 == 4) }
        for algorithm in MLAlgorithm.allCases {
            let a = MLEngine.run(algorithm: algorithm, points: identical, configuration: config)
            let b = MLEngine.run(algorithm: algorithm, points: identical, configuration: config)
            XCTAssertFalse(a.frames.isEmpty, algorithm.rawValue)
            XCTAssertLessThanOrEqual(a.frames.count, 40, algorithm.rawValue)
            XCTAssertEqual(a.frames, b.frames, algorithm.rawValue)
            for frame in a.frames {
                XCTAssertTrue(frame.metrics.allSatisfy { $0.value.isFinite }, algorithm.rawValue)
                XCTAssertTrue(frame.history.allSatisfy(\.isFinite), algorithm.rawValue)
                XCTAssertTrue((frame.points + frame.centers + frame.curve).allSatisfy { $0.x.isFinite && $0.y.isFinite })
                XCTAssertTrue(frame.grid.allSatisfy { $0.confidence.isFinite && (0...1).contains($0.confidence) })
            }
        }
    }

    func testEmptyAndAllTestDataDoNotInventAModel() {
        for algorithm in MLAlgorithm.allCases where algorithm != .monteCarlo {
            let empty = MLEngine.run(algorithm: algorithm, points: [], configuration: MLConfiguration())
            XCTAssertEqual(empty.frames.count, 1)
            XCTAssertTrue(empty.frames[0].metrics.isEmpty)
            let testOnly = MLEngine.run(algorithm: algorithm, points: [MLPoint(id: 0, x: 0.4, y: 0.7, isTest: true)], configuration: MLConfiguration())
            XCTAssertTrue(testOnly.summary.contains("no training samples"))
        }
    }

    func testSingleSampleAndInvalidInputStayFinite() {
        var config = MLConfiguration(); config.learningRate = .nan; config.noise = .infinity
        config.clusters = -5; config.minPoints = -1; config.epsilon = .nan
        let points = [MLPoint(id: 1, x: .nan, y: .infinity, label: 99)]
        for algorithm in MLAlgorithm.allCases {
            let result = MLEngine.run(algorithm: algorithm, points: points, configuration: config)
            XCTAssertLessThanOrEqual(result.frames.count, 40)
            XCTAssertTrue(result.frames.allSatisfy { $0.metrics.allSatisfy { $0.value.isFinite } }, algorithm.rawValue)
        }
    }
}
