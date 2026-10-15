import Foundation
import Observation
import MLCore

struct LabComputationRequest: Sendable, Equatable {
    let revision: UInt64
    let algorithm: MLAlgorithm
    let dataset: DatasetKind
    let configuration: MLConfiguration
    let points: [MLPoint]
}

typealias LabComputationRunner = @Sendable (LabComputationRequest) async -> SimulationResult

/// Queued canceled requests skip fitting. A running synchronous fit may finish,
/// but its revision cannot replace a newer experiment. This actor keeps CPU work
/// serial and away from the main actor, including during rapid slider changes.
private actor LabEngineWorker {
    func run(_ request: LabComputationRequest) -> SimulationResult {
        guard !Task.isCancelled else { return SimulationResult(frames: [], summary: "") }
        return MLEngine.run(algorithm: request.algorithm, points: request.points,
                            configuration: request.configuration)
    }
}

private struct DatasetSignature: Equatable {
    let kind: DatasetKind
    let sampleCount: Int
    let noise: Double
    let seed: UInt64
}

@MainActor @Observable
final class LabSession {
    private(set) var algorithm: MLAlgorithm = .kMeans
    private(set) var dataset: DatasetKind = .blobs
    var configuration = MLConfiguration()
    private(set) var points: [MLPoint] = []
    private(set) var result = SimulationResult(frames: [], summary: "")
    private(set) var isComputing = false
    private(set) var customPointCount = 0
    private(set) var computationRevision: UInt64 = 0
    private(set) var requestedComputation: LabComputationRequest?
    private(set) var completedComputation: LabComputationRequest?
    var probe: MLPoint?
    var speed = 1.0
    private var storedFrameIndex = 0
    private var playbackRequested = false
    private var editingRequested = false
    private var selectedEditLabel = 0
    @ObservationIgnored private var computation: Task<Void, Never>?
    @ObservationIgnored private var datasetSignature: DatasetSignature?
    @ObservationIgnored private var ordinarySampleCount = 90
    @ObservationIgnored private var samplingSampleCount = 1000
    @ObservationIgnored private let debounce: Duration
    @ObservationIgnored private let runner: LabComputationRunner

    init(algorithm initialAlgorithm: MLAlgorithm? = nil,
         debounce: Duration = .milliseconds(130), runner: LabComputationRunner? = nil) {
        self.debounce = max(.zero, debounce)
        if let runner { self.runner = runner }
        else {
            let worker = LabEngineWorker()
            self.runner = { request in await worker.run(request) }
        }
        var selected = initialAlgorithm
        #if DEBUG
        if selected == nil,
           let flag = ProcessInfo.processInfo.arguments.firstIndex(of: "--model"),
           ProcessInfo.processInfo.arguments.count > flag + 1 {
            selected = MLAlgorithm(rawValue: ProcessInfo.processInfo.arguments[flag + 1])
        }
        #endif
        algorithm = selected ?? .kMeans
        dataset = algorithm.defaultDataset
        if algorithm == .monteCarlo { configuration.sampleCount = samplingSampleCount }
    }

    deinit { computation?.cancel() }

    var frameIndex: Int {
        get { boundedFrameIndex(storedFrameIndex) }
        set { seek(to: newValue) }
    }
    var canPlay: Bool { !isComputing && result.frames.count > 1 }
    var isPlaying: Bool {
        get { playbackRequested && canPlay }
        set { playbackRequested = newValue && canPlay }
    }
    var canEdit: Bool { algorithm != .monteCarlo }
    var remainingPointCapacity: Int { max(0, 220 - points.count) }
    var isEditing: Bool {
        get { editingRequested && canEdit }
        set {
            editingRequested = newValue && canEdit
            if editingRequested { isPlaying = false; probe = nil }
        }
    }
    var allowedEditLabels: [Int] {
        if isRegression || isUnsupervised { return [0] }
        return [0, 1]
    }
    var editLabel: Int {
        get { selectedEditLabel }
        set { selectedEditLabel = min(allowedEditLabels.last ?? 0, max(0, newValue)) }
    }

    var frame: SimulationFrame {
        guard !result.frames.isEmpty else {
            return SimulationFrame(step: 0, caption: isComputing ? "Preparing the experiment" : "No recorded frames",
                                   detail: isComputing ? "Computing the model on your device." : "Run a calculation to prepare this experiment.",
                                   points: points)
        }
        var current = result.frames[frameIndex]
        if algorithm == .knn, let probe {
            let nearest = current.points.filter { !$0.isTest }.sorted {
                let a = hypot($0.x - probe.x, $0.y - probe.y), b = hypot($1.x - probe.x, $1.y - probe.y)
                return a == b ? $0.id < $1.id : a < b
            }.prefix(min(31, max(1, configuration.neighborCount)))
            current.vectors += nearest.map { MLLine(x1: probe.x, y1: probe.y, x2: $0.x, y2: $0.y, label: $0.label > 0 ? 1 : 0) }
        }
        return current
    }

    var progress: Double { Double(frameIndex) / Double(max(1, result.frames.count - 1)) }
    var isRegression: Bool { algorithm == .linearRegression || algorithm == .polynomialRegression }
    var isUnsupervised: Bool { [.kMeans, .dbscan, .pca, .monteCarlo].contains(algorithm) }

    func select(_ model: MLAlgorithm) {
        guard model != algorithm else { return }
        isPlaying = false; isEditing = false; probe = nil; selectedEditLabel = 0
        let wasSampling = algorithm == .monteCarlo
        if wasSampling { samplingSampleCount = configuration.sampleCount }
        else { ordinarySampleCount = configuration.sampleCount }
        algorithm = model
        if model == .monteCarlo {
            configuration.sampleCount = samplingSampleCount
            dataset = model.defaultDataset
        } else if wasSampling { configuration.sampleCount = ordinarySampleCount }
        let incompatible = !model.supportsDataset(dataset)
        if incompatible || wasSampling { dataset = model.defaultDataset }
        recompute(regenerate: incompatible || wasSampling)
    }

    func changeDataset(_ value: DatasetKind) {
        // A callback from a disappearing picker can arrive after a model switch.
        guard algorithm != .monteCarlo, algorithm.supportsDataset(value), value != dataset else { return }
        dataset = value; probe = nil; recompute(regenerate: true)
    }

    func recompute(regenerate: Bool = false) {
        computation?.cancel()
        computationRevision &+= 1
        isPlaying = false; isComputing = true; storedFrameIndex = 0; probe = nil
        // Do not show the previous fit under a newly changed model or setting.
        result = SimulationResult(frames: [], summary: "")
        completedComputation = nil
        if !algorithm.supportsDataset(dataset) { dataset = algorithm.defaultDataset }
        if algorithm == .monteCarlo {
            points = []; customPointCount = 0; datasetSignature = nil
            editingRequested = false; selectedEditLabel = 0
        } else {
            let signature = DatasetSignature(kind: dataset, sampleCount: configuration.sampleCount,
                                             noise: configuration.noise, seed: configuration.seed)
            if regenerate || points.isEmpty || signature != datasetSignature {
                points = DatasetGenerator.generate(dataset, configuration: configuration)
                customPointCount = 0; datasetSignature = signature
            }
        }
        let request = LabComputationRequest(revision: computationRevision, algorithm: algorithm,
                                            dataset: dataset, configuration: configuration, points: points)
        requestedComputation = request
        let delay = debounce, run = runner
        computation = Task(priority: .userInitiated) { [weak self] in
            if delay > .zero {
                do { try await Task.sleep(for: delay) } catch { return }
            }
            guard !Task.isCancelled else { return }
            let computed = await run(request)
            guard !Task.isCancelled, let self,
                  self.requestedComputation?.revision == request.revision,
                  self.computationRevision == request.revision else { return }
            self.result = computed; self.storedFrameIndex = 0
            self.completedComputation = request; self.isComputing = false
            self.computation = nil
        }
    }

    func cancelComputation() {
        computation?.cancel(); computation = nil
        computationRevision &+= 1; requestedComputation = nil
        isComputing = false; isPlaying = false
    }
    func newSeed() { configuration.seed &+= 1; recompute(regenerate: true) }
    func reset() { seek(to: 0); probe = nil }
    func seek(to index: Int) { isPlaying = false; storedFrameIndex = boundedFrameIndex(index) }
    func step(_ delta: Int = 1) {
        let (index, overflow) = frameIndex.addingReportingOverflow(delta)
        seek(to: overflow ? (delta > 0 ? .max : .min) : index)
    }
    func togglePlayback() {
        guard canPlay else { playbackRequested = false; return }
        if !isPlaying && frameIndex >= result.frames.count - 1 { storedFrameIndex = 0 }
        isPlaying.toggle()
    }
    func tick() {
        guard isPlaying else { playbackRequested = false; return }
        if frameIndex < result.frames.count - 1 { storedFrameIndex = frameIndex + 1 }
        if frameIndex >= result.frames.count - 1 { isPlaying = false }
    }

    func touchPlot(x: Double, y: Double) {
        guard x.isFinite, y.isFinite else { return }
        let point = MLPoint(id: (points.map(\.id).max() ?? -1) + 1,
                            x: min(1, max(0, x)), y: min(1, max(0, y)), label: editLabel)
        if isEditing {
            guard remainingPointCapacity > 0 else { return }
            points.append(point); customPointCount += 1; recompute()
        } else { probe = point }
    }
    func removeLastPoint() {
        guard customPointCount > 0, !points.isEmpty else { return }
        points.removeLast(); customPointCount -= 1; recompute()
    }

    var probeDescription: String? {
        guard let probe else { return nil }
        let axes = isRegression || algorithm == .monteCarlo ? "x %.2f · y %.2f" : "x₁ %.2f · x₂ %.2f"
        let position = String(format: axes, probe.x, probe.y)
        if algorithm == .monteCarlo {
            return position + (probe.x * probe.x + probe.y * probe.y <= 1 ? " · inside the circle" : " · outside the circle")
        }
        if isRegression, !frame.curve.isEmpty {
            let nearest = frame.curve.min { abs($0.x - probe.x) < abs($1.x - probe.x) }
            return position + String(format: " · ŷ ≈ %.3f", nearest?.y ?? 0)
        }
        if let cell = frame.grid.min(by: { hypot($0.x - probe.x, $0.y - probe.y) < hypot($1.x - probe.x, $1.y - probe.y) }) {
            let name: String
            if cell.label == -2 { name = "unvisited" }
            else if cell.label < 0 { name = "noise" }
            else if cell.label < 26 { name = String(UnicodeScalar(65 + cell.label)!) }
            else { name = "G\(cell.label + 1)" }
            return position + " · grid \(isUnsupervised ? "group" : "class") \(name)"
        }
        return position
    }
    private func boundedFrameIndex(_ value: Int) -> Int {
        min(max(0, value), max(0, result.frames.count - 1))
    }
}
