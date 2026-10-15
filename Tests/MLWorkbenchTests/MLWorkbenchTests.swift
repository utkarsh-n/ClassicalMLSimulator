import XCTest
import MLCore
@testable import MLWorkbench

private func markedResult(_ request: LabComputationRequest, frameCount: Int = 3) -> SimulationResult {
    SimulationResult(frames: (0..<frameCount).map { index in
        SimulationFrame(step: index + 1, caption: "\(request.algorithm.rawValue) seed \(request.configuration.seed)",
                        detail: "request \(request.revision)", points: request.points)
    }, summary: "\(request.algorithm.rawValue) seed \(request.configuration.seed)")
}

private actor RequestRecorder {
    private var requests: [LabComputationRequest] = []
    func run(_ request: LabComputationRequest) -> SimulationResult {
        requests.append(request)
        return markedResult(request)
    }
    func captured() -> [LabComputationRequest] { requests }
}

/// Deliberately ignores task cancellation so tests can finish old fits after new
/// fits. Publication must remain correct even when a solver cannot be stopped.
private actor ControlledRunner {
    private var requests: [LabComputationRequest] = []
    private var continuations: [UInt64: CheckedContinuation<SimulationResult, Never>] = [:]
    func run(_ request: LabComputationRequest) async -> SimulationResult {
        requests.append(request)
        return await withCheckedContinuation { continuations[request.revision] = $0 }
    }
    func captured() -> [LabComputationRequest] { requests }
    @discardableResult func complete(_ revision: UInt64) -> Bool {
        guard let continuation = continuations.removeValue(forKey: revision),
              let request = requests.first(where: { $0.revision == revision }) else { return false }
        continuation.resume(returning: markedResult(request))
        return true
    }
    func completeAll() {
        for revision in Array(continuations.keys) { complete(revision) }
    }
}

final class MLWorkbenchTests: XCTestCase {
    func testCatalogDefaultsAndFamilyDatasetContracts() {
        XCTAssertEqual(MLAlgorithm.allCases.count, 13)
        for model in MLAlgorithm.allCases {
            XCTAssertTrue(model.supportsDataset(model.defaultDataset), model.rawValue)
            XCTAssertEqual(model.steps.count, 3, model.rawValue)
            XCTAssertFalse(model.equation.isEmpty, model.rawValue)
            XCTAssertFalse(model.limitation.isEmpty, model.rawValue)
            let supported = DatasetKind.allCases.filter(model.supportsDataset)
            switch model {
            case .linearRegression, .polynomialRegression:
                XCTAssertEqual(Set(supported), [.linear, .curve])
            case .kMeans, .dbscan:
                XCTAssertEqual(Set(supported), [.blobs, .moons, .circles, .anisotropic])
            case .pca:
                XCTAssertEqual(Set(supported), [.anisotropic, .blobs])
            case .monteCarlo:
                XCTAssertEqual(supported.count, DatasetKind.allCases.count)
            default:
                XCTAssertEqual(Set(supported), [.blobs, .moons, .circles, .xor])
            }
        }
    }

    @MainActor
    func testAllThirteenModelsSwitchAndPublishTheCurrentExperiment() async throws {
        let session = LabSession(debounce: .zero)
        var previousRevision: UInt64 = 0
        for model in MLAlgorithm.allCases {
            session.select(model)
            session.configuration.sampleCount = model == .monteCarlo ? 100 : 30
            session.configuration.iterations = 10
            session.configuration.treeCount = 3
            session.recompute()
            XCTAssertTrue(session.isComputing)
            XCTAssertTrue(session.result.frames.isEmpty, "Old frames under \(model.rawValue)")
            XCTAssertNil(session.completedComputation)
            try await ready(session)
            let request = try XCTUnwrap(session.completedComputation)
            XCTAssertEqual(request.algorithm, model)
            XCTAssertEqual(request.dataset, session.dataset)
            XCTAssertEqual(request.points, session.points)
            XCTAssertEqual(request.configuration, session.configuration)
            XCTAssertGreaterThan(request.revision, previousRevision)
            XCTAssertEqual(request.revision, session.computationRevision)
            XCTAssertEqual(request, session.requestedComputation)
            XCTAssertTrue(model.supportsDataset(session.dataset))
            XCTAssertFalse(session.result.frames.isEmpty, model.rawValue)
            let direct = MLEngine.run(algorithm: model, points: request.points, configuration: request.configuration)
            XCTAssertEqual(session.result.frames, direct.frames, model.rawValue)
            previousRevision = request.revision
        }
    }

    @MainActor
    func testEveryModelControlAtBothUIEndpointsMatchesTheFittedResult() async throws {
        let session = LabSession(debounce: .zero)
        for model in MLAlgorithm.allCases {
            for upper in [false, true] {
                session.select(model)
                var settings = MLConfiguration()
                settings.sampleCount = model == .monteCarlo ? (upper ? 2000 : 100) : (upper ? 180 : 30)
                settings.noise = upper ? 0.22 : 0
                settings.iterations = upper ? 40 : 10
                settings.learningRate = upper ? 0.8 : 0.02
                settings.neighborCount = upper ? 21 : 1
                settings.clusters = upper ? 6 : 2
                settings.treeDepth = upper ? 6 : 1
                settings.treeCount = upper ? 30 : 3
                settings.degree = upper ? 6 : 1
                settings.epsilon = upper ? 0.3 : 0.05
                settings.minPoints = upper ? 12 : 2
                settings.regularization = upper ? 0.2 : 0
                session.configuration = settings
                session.recompute()
                try await ready(session)
                let request = try XCTUnwrap(session.completedComputation)
                XCTAssertEqual(request.configuration, settings, model.rawValue)
                let expected = MLEngine.run(algorithm: model, points: session.points, configuration: settings)
                XCTAssertEqual(session.result.frames, expected.frames, "\(model.rawValue) upper=\(upper)")
                let last = try XCTUnwrap(session.result.frames.last)
                XCTAssertTrue(last.metrics.allSatisfy { $0.value.isFinite }, model.rawValue)
                switch model {
                case .polynomialRegression:
                    XCTAssertEqual(metric("Degree", in: last), Double(settings.degree))
                case .knn:
                    XCTAssertEqual(metric("Neighbors k", in: last), Double(settings.neighborCount))
                case .decisionTree:
                    XCTAssertEqual(metric("Depth limit", in: last), Double(settings.treeDepth))
                case .randomForest:
                    XCTAssertEqual(metric("Trees fitted", in: last), Double(settings.treeCount))
                case .gradientBoosting:
                    XCTAssertEqual(metric("Stumps fitted", in: last), Double(settings.treeCount))
                case .kMeans:
                    XCTAssertEqual(last.centers.count, settings.clusters)
                case .monteCarlo:
                    XCTAssertEqual(last.points.count, settings.sampleCount)
                default: break
                }
            }
        }
    }

    @MainActor
    func testRapidRecomputationDebouncesToOneCurrentRequest() async throws {
        let recorder = RequestRecorder()
        let session = LabSession(algorithm: .linearRegression, debounce: .milliseconds(40),
                                 runner: { await recorder.run($0) })
        for index in 0..<80 {
            session.configuration.learningRate = 0.02 + Double(index % 40) * 0.02
            session.configuration.seed = UInt64(index)
            session.recompute()
        }
        let latestRevision = session.computationRevision
        try await ready(session)
        let captured = await recorder.captured()
        XCTAssertEqual(captured.count, 1)
        XCTAssertEqual(captured[0].configuration.seed, 79)
        XCTAssertEqual(captured[0].revision, latestRevision)
        XCTAssertEqual(session.completedComputation?.revision, latestRevision)
        XCTAssertEqual(session.result.summary, "linearRegression seed 79")
    }

    @MainActor
    func testOlderFitCannotPublishAfterANewerModelOrConfiguration() async throws {
        let controlled = ControlledRunner()
        defer { Task { await controlled.completeAll() } }
        let session = LabSession(algorithm: .linearRegression, debounce: .zero,
                                 runner: { await controlled.run($0) })
        session.recompute()
        try await requests(controlled, count: 1)
        let first = session.computationRevision
        await controlled.complete(first)
        try await ready(session)
        XCTAssertFalse(session.result.frames.isEmpty)

        session.configuration.seed = 88
        session.recompute()
        let obsolete = session.computationRevision
        XCTAssertTrue(session.result.frames.isEmpty)
        XCTAssertNil(session.completedComputation)
        try await requests(controlled, count: 2)
        session.select(.pca)
        let newest = session.computationRevision
        try await requests(controlled, count: 3)
        await controlled.complete(newest)
        try await ready(session)
        let currentFrames = session.result.frames
        XCTAssertEqual(session.completedComputation?.algorithm, .pca)
        XCTAssertEqual(session.completedComputation?.configuration.seed, 88)
        await controlled.complete(obsolete)
        try await Task.sleep(for: .milliseconds(20))
        XCTAssertEqual(session.result.frames, currentFrames)
        XCTAssertEqual(session.completedComputation?.revision, newest)
        XCTAssertFalse(session.isComputing)
    }

    @MainActor
    func testExplicitCancellationDiscardsALateSolverResult() async throws {
        let controlled = ControlledRunner()
        defer { Task { await controlled.completeAll() } }
        let session = LabSession(debounce: .zero, runner: { await controlled.run($0) })
        session.recompute()
        try await requests(controlled, count: 1)
        let canceledRevision = session.computationRevision
        session.cancelComputation()
        XCTAssertFalse(session.isComputing)
        XCTAssertNil(session.requestedComputation)
        await controlled.complete(canceledRevision)
        try await Task.sleep(for: .milliseconds(20))
        XCTAssertTrue(session.result.frames.isEmpty)
        XCTAssertNil(session.completedComputation)
        XCTAssertFalse(session.isPlaying)
    }

    @MainActor
    func testAnInFlightSolverDoesNotRetainADiscardedSession() async throws {
        let controlled = ControlledRunner()
        var session: LabSession? = LabSession(debounce: .zero, runner: { await controlled.run($0) })
        weak var weakSession = session
        session?.recompute()
        try await requests(controlled, count: 1)
        session = nil
        XCTAssertNil(weakSession)
        await controlled.completeAll()
    }

    @MainActor
    func testEmptyAndSingleFramePlaybackIsBoundedAndCannotStart() async throws {
        for count in [0, 1] {
            let session = LabSession(debounce: .zero, runner: { markedResult($0, frameCount: count) })
            session.recompute()
            session.togglePlayback()
            XCTAssertFalse(session.isPlaying)
            try await ready(session)
            for destination in [Int.min, -1, 0, 1, Int.max] {
                session.seek(to: destination)
                session.step(destination)
                session.tick()
                XCTAssertEqual(session.frameIndex, 0)
                XCTAssertEqual(session.progress, 0)
                _ = session.frame
            }
            session.togglePlayback()
            session.isPlaying = true
            XCTAssertFalse(session.canPlay)
            XCTAssertFalse(session.isPlaying)
            session.reset()
        }
    }

    @MainActor
    func testSeekStepAndPlaybackSafelyHandleBothBoundariesAndIntegerOverflow() async throws {
        let session = fastSession()
        session.recompute()
        try await ready(session)
        session.seek(to: Int.max)
        XCTAssertEqual(session.frameIndex, 2)
        XCTAssertEqual(session.progress, 1)
        session.step(Int.max)
        XCTAssertEqual(session.frameIndex, 2)
        session.step(Int.min)
        XCTAssertEqual(session.frameIndex, 0)
        session.seek(to: -500)
        XCTAssertEqual(session.frameIndex, 0)
        session.seek(to: 2)
        session.togglePlayback()
        XCTAssertEqual(session.frameIndex, 0)
        XCTAssertTrue(session.isPlaying)
        session.tick()
        XCTAssertEqual(session.frameIndex, 1)
        XCTAssertTrue(session.isPlaying)
        session.tick()
        XCTAssertEqual(session.frameIndex, 2)
        XCTAssertFalse(session.isPlaying)
        session.tick()
        XCTAssertEqual(session.frameIndex, 2)
        session.togglePlayback()
        session.frameIndex = 1
        XCTAssertFalse(session.isPlaying)
        session.touchPlot(x: 0.4, y: 0.6)
        session.reset()
        XCTAssertEqual(session.frameIndex, 0)
        XCTAssertNil(session.probe)
    }

    @MainActor
    func testDataSettingChangesRegenerateSourcesWhileModelSettingsPreserveEdits() async throws {
        let session = fastSession(algorithm: .knn)
        session.configuration.sampleCount = 30
        session.recompute()
        try await ready(session)
        session.isEditing = true
        session.editLabel = 1
        session.touchPlot(x: 0.4, y: 0.7)
        let edited = session.points
        session.configuration.neighborCount = 9
        session.recompute()
        try await ready(session)
        XCTAssertEqual(session.points, edited)
        XCTAssertEqual(session.customPointCount, 1)
        session.configuration.sampleCount = 40
        session.recompute() // Source signatures make the regeneration flag optional.
        try await ready(session)
        XCTAssertEqual(session.points.count, 40)
        XCTAssertEqual(session.customPointCount, 0)
        XCTAssertEqual(session.points, DatasetGenerator.generate(session.dataset, configuration: session.configuration))
        let beforeNoise = session.points
        session.configuration.noise = 0.2
        session.recompute()
        try await ready(session)
        XCTAssertNotEqual(session.points, beforeNoise)
        session.configuration.seed = .max
        session.newSeed()
        try await ready(session)
        XCTAssertEqual(session.configuration.seed, 0)
        XCTAssertEqual(session.points, DatasetGenerator.generate(session.dataset, configuration: session.configuration))
    }

    @MainActor
    func testCompatibleModelsKeepEditsAndSamplingRestoresSeparateSampleCounts() async throws {
        let session = fastSession()
        session.configuration.sampleCount = 140
        session.recompute()
        try await ready(session)
        session.isEditing = true
        session.touchPlot(x: 0.2, y: 0.3)
        let edited = session.points
        session.select(.dbscan)
        try await ready(session)
        XCTAssertEqual(session.points, edited)
        XCTAssertEqual(session.customPointCount, 1)
        session.select(.linearRegression)
        try await ready(session)
        XCTAssertEqual(session.dataset, .linear)
        XCTAssertEqual(session.points.count, 140)
        XCTAssertEqual(session.customPointCount, 0)
        session.select(.monteCarlo)
        try await ready(session)
        XCTAssertEqual(session.configuration.sampleCount, 1000)
        XCTAssertTrue(session.points.isEmpty)
        session.isEditing = true
        XCTAssertFalse(session.isEditing)
        session.configuration.sampleCount = 2000
        session.recompute()
        try await ready(session)
        session.select(.kMeans)
        try await ready(session)
        XCTAssertEqual(session.configuration.sampleCount, 140)
        XCTAssertEqual(session.points.count, 140)
        session.select(.monteCarlo)
        try await ready(session)
        XCTAssertEqual(session.configuration.sampleCount, 2000)
    }

    @MainActor
    func testUnsupportedAndUnchangedDatasetCallbacksDoNotRebuild() async throws {
        let session = fastSession(algorithm: .linearRegression)
        session.recompute()
        try await ready(session)
        let revision = session.computationRevision, points = session.points
        session.changeDataset(.moons)
        session.changeDataset(.linear)
        session.select(.linearRegression)
        XCTAssertEqual(session.computationRevision, revision)
        XCTAssertEqual(session.points, points)
        session.changeDataset(.curve)
        try await ready(session)
        XCTAssertEqual(session.dataset, .curve)
        XCTAssertGreaterThan(session.computationRevision, revision)
        session.select(.pca)
        try await ready(session)
        XCTAssertEqual(session.dataset, .anisotropic)
        XCTAssertTrue(session.algorithm.supportsDataset(session.dataset))
    }

    @MainActor
    func testEditCapacityUndoAndInvalidCoordinatesPreserveSourceIntegrity() async throws {
        let session = fastSession(algorithm: .knn)
        session.configuration.sampleCount = 180
        session.recompute()
        try await ready(session)
        session.isEditing = true
        session.editLabel = 99
        XCTAssertEqual(session.editLabel, 1)
        for index in 0..<50 { session.touchPlot(x: Double(index) / 50, y: 0.7) }
        try await ready(session)
        XCTAssertEqual(session.points.count, 220)
        XCTAssertEqual(session.customPointCount, 40)
        XCTAssertEqual(session.remainingPointCapacity, 0)
        XCTAssertEqual(Set(session.points.map(\.id)).count, 220)
        XCTAssertTrue(session.points.suffix(40).allSatisfy { !$0.isTest && $0.label == 1 })
        let full = session.points, revision = session.computationRevision
        session.touchPlot(x: 0.9, y: 0.9)
        XCTAssertEqual(session.points, full)
        XCTAssertEqual(session.computationRevision, revision)
        XCTAssertNil(session.probe, "A rejected draw should not silently become an inspect action.")
        session.removeLastPoint()
        try await ready(session)
        XCTAssertEqual(session.points.count, 219)
        XCTAssertEqual(session.customPointCount, 39)
        session.touchPlot(x: .nan, y: 0.2)
        session.touchPlot(x: 0.2, y: .infinity)
        XCTAssertEqual(session.points.count, 219)
        session.touchPlot(x: -1, y: 3)
        try await ready(session)
        XCTAssertEqual(session.points.last?.x, 0)
        XCTAssertEqual(session.points.last?.y, 1)
        session.isEditing = false
        session.touchPlot(x: -2, y: 4)
        XCTAssertEqual(session.probe?.x, 0)
        XCTAssertEqual(session.probe?.y, 1)
        session.isEditing = true
        XCTAssertNil(session.probe)
    }

    @MainActor
    func testEveryModelSupportsItsIntendedEditUndoProbeAndReseedActions() async throws {
        let session = fastSession()
        session.configuration.sampleCount = 30
        for model in MLAlgorithm.allCases {
            session.select(model)
            session.recompute()
            try await ready(session)
            XCTAssertEqual(session.editLabel, 0)
            session.editLabel = 2
            XCTAssertEqual(session.editLabel, session.isRegression || session.isUnsupervised ? 0 : 1)
            let baseline = session.points
            session.isEditing = true
            session.touchPlot(x: 0.5, y: 0.5)
            try await ready(session)
            if model == .monteCarlo {
                XCTAssertFalse(session.isEditing)
                XCTAssertTrue(session.points.isEmpty)
                XCTAssertNotNil(session.probe)
            } else {
                XCTAssertEqual(session.points.count, baseline.count + 1, model.rawValue)
                XCTAssertEqual(session.customPointCount, 1)
                XCTAssertFalse(try XCTUnwrap(session.points.last).isTest)
                session.removeLastPoint()
                try await ready(session)
                XCTAssertEqual(session.points, baseline, model.rawValue)
                session.removeLastPoint()
                XCTAssertEqual(session.points, baseline)
            }
            session.isEditing = false
            session.touchPlot(x: 0.25, y: 0.25)
            XCTAssertNotNil(session.probeDescription)
            let seed = session.configuration.seed
            session.newSeed()
            try await ready(session)
            XCTAssertEqual(session.configuration.seed, seed &+ 1)
            XCTAssertNil(session.probe)
            XCTAssertEqual(session.customPointCount, 0)
            if model != .monteCarlo { XCTAssertNotEqual(session.points, baseline, model.rawValue) }
        }
    }

    @MainActor
    func testKNNProbeUsesCurrentMemoryAndExcludesHeldOutSamples() async throws {
        let session = LabSession(algorithm: .knn, debounce: .zero)
        session.configuration.sampleCount = 30
        session.recompute()
        try await ready(session)
        session.touchPlot(x: 0.5, y: 0.5)
        let frame = session.frame
        let storedTraining = frame.points.filter { !$0.isTest }
        XCTAssertEqual(frame.vectors.count, min(5, storedTraining.count))
        for vector in frame.vectors {
            XCTAssertTrue(storedTraining.contains { $0.x == vector.x2 && $0.y == vector.y2 })
        }
        session.configuration.neighborCount = -100
        XCTAssertEqual(session.frame.vectors.count, 1, "Invalid pending k must not trap in prefix(_:).")
        session.recompute()
        XCTAssertNil(session.probe)
        try await ready(session)
    }

    @MainActor
    func testProbeNamesDoNotCollapseGroupsBeyondZ() async throws {
        let session = LabSession(algorithm: .dbscan, debounce: .zero, runner: { _ in
            SimulationResult(frames: [SimulationFrame(step: 1, caption: "Many clusters", detail: "",
                                                     grid: [GridCell(x: 0.5, y: 0.5, label: 26)])], summary: "")
        })
        session.recompute()
        try await ready(session)
        session.touchPlot(x: 0.5, y: 0.5)
        XCTAssertTrue(try XCTUnwrap(session.probeDescription).contains("G27"))
    }

    @MainActor
    private func fastSession(algorithm: MLAlgorithm = .kMeans) -> LabSession {
        LabSession(algorithm: algorithm, debounce: .zero, runner: { markedResult($0) })
    }

    @MainActor
    private func ready(_ session: LabSession, file: StaticString = #filePath, line: UInt = #line) async throws {
        for _ in 0..<1500 {
            if !session.isComputing { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Timed out waiting for revision \(session.computationRevision)", file: file, line: line)
        throw WorkbenchTimeout.waiting
    }

    @MainActor
    private func requests(_ runner: ControlledRunner, count: Int,
                          file: StaticString = #filePath, line: UInt = #line) async throws {
        for _ in 0..<500 {
            if await runner.captured().count >= count { return }
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTFail("The controlled solver did not receive \(count) requests.", file: file, line: line)
        throw WorkbenchTimeout.waiting
    }

    private func metric(_ name: String, in frame: SimulationFrame) -> Double {
        frame.metrics.first { $0.name == name }?.value ?? .nan
    }
}

private enum WorkbenchTimeout: Error { case waiting }
