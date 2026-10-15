import XCTest
@testable import MLCore

final class MLCoreStressTests: XCTestCase {
    private func metric(_ name: String, _ frame: SimulationFrame) -> Double {
        frame.metrics.first { $0.name == name }?.value ?? .nan
    }

    func testBalancedXORCanUseZeroGainRootSplit() {
        let points = (0..<80).map { i -> MLPoint in
            let corner = i % 4
            return MLPoint(id: i, x: corner % 2 == 0 ? 0.2 : 0.8, y: corner < 2 ? 0.2 : 0.8,
                           label: corner == 1 || corner == 2 ? 1 : 0, isTest: i % 5 == 4)
        }
        var config = MLConfiguration(); config.treeDepth = 2
        let tree = MLEngine.run(algorithm: .decisionTree, points: points, configuration: config).frames.last!
        XCTAssertEqual(metric("Train accuracy", tree), 100)
        XCTAssertEqual(metric("Test accuracy", tree), 100)
        XCTAssertEqual(tree.tree.count, 7)
        let forest = MLEngine.run(algorithm: .randomForest, points: points, configuration: config).frames.last!
        XCTAssertEqual(metric("Test accuracy", forest), 100)
    }

    func testDuplicateInputIDsCannotExposeUnmemorizedKNNPoints() {
        let points = (0..<24).map { i in MLPoint(id: 7, x: Double(i) / 24, y: 0.5, label: i % 2, isTest: i % 5 == 4) }
        let result = MLEngine.run(algorithm: .knn, points: points, configuration: MLConfiguration())
        let first = result.frames.first!, last = result.frames.last!
        XCTAssertEqual(first.points.filter { !$0.isTest }.count, Int(metric("Memorized", first)))
        XCTAssertEqual(Set(last.points.map(\.id)).count, last.points.count)
        XCTAssertEqual(last.points.count, points.count)
        XCTAssertEqual(last.points.first?.id, 7, "Preserve the first occurrence of an existing identity")
    }
}
