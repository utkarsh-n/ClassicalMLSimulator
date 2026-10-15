import SwiftUI
import MLCore

struct MLRootView: View {
    @State private var session = LabSession()
    @State private var tab = 0
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        TabView(selection: $tab) {
            Tab("Workbench", systemImage: "slider.horizontal.3", value: 0) {
                NavigationStack { WorkbenchView(session: session) }
            }
            Tab("Models", systemImage: "square.grid.2x2", value: 1) {
                NavigationStack {
                    ModelLibraryView(selected: session.algorithm) { model in
                        session.select(model); tab = 0
                    }
                }
            }
            Tab("Field Guide", systemImage: "book.closed", value: 2) {
                NavigationStack { FieldGuideView() }
            }
        }
        .task { session.recompute(regenerate: true) }
        .task(id: session.isPlaying) {
            guard session.isPlaying else { return }
            while !Task.isCancelled && session.isPlaying {
                do { try await Task.sleep(for: .milliseconds(Int(500 / session.speed))) }
                catch { return }
                session.tick()
            }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { session.isPlaying = false }
        }
        .onChange(of: tab) { _, _ in session.isPlaying = false }
    }
}

struct WorkbenchView: View {
    @Bindable var session: LabSession
    @State private var showModels = false
    @State private var showComparison = false
    @State private var telemetryHeight: CGFloat = 0
    @Environment(\.dynamicTypeSize) private var textSize

    var body: some View {
        ZStack {
            LabBackground()
            ScrollView {
                VStack(spacing: 16) {
                    brandHeader
                    modelHeader
                    plotInstrument
                    metricsPanel
                    datasetPanel
                    ModelControlsView(session: session)
                    lessonPanel
                    if !session.frame.tree.isEmpty {
                        LabPanel("Inside the tree") {
                            TreeDiagramView(nodes: session.frame.tree)
                            Text(session.algorithm == .randomForest ? "One representative tree. The regions above average the forest’s leaf probabilities." : session.algorithm == .gradientBoosting ? "Newest stump: each leaf adds a score correction. The regions use all accumulated stumps." : "Each branch compares a feature with a threshold; each leaf shows its majority class and sample count.")
                                .font(.caption).foregroundStyle(MLTheme.muted)
                        }
                    }
                    if !session.frame.history.isEmpty {
                        LabPanel("Learning trace") {
                            HistorySparkline(values: session.frame.history, color: MLTheme.violet).frame(height: 64)
                            Text(historyDescription)
                                .font(.caption).foregroundStyle(MLTheme.muted)
                        }
                    }
                    Text("LOCAL COMPUTATION · SYNTHETIC DATA · NO NETWORK")
                        .font(.system(.caption2, design: .monospaced)).foregroundStyle(MLTheme.muted)
                        .padding(.vertical, 8)
                }
                .frame(maxWidth: 620).padding(.horizontal, 16).padding(.bottom, 12)
                .frame(maxWidth: .infinity)
            }
            .scrollIndicators(.hidden)
            .id(session.algorithm.rawValue)
            .accessibilityIdentifier("workbenchScroll")
        }
        .toolbar(.hidden, for: .navigationBar)
        .safeAreaInset(edge: .bottom, spacing: 0) { transport }
        .sheet(isPresented: $showModels) {
            NavigationStack {
                ModelLibraryView(selected: session.algorithm) { model in
                    session.select(model); showModels = false
                }
                .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Done") { showModels = false } } }
            }
        }
        .sheet(isPresented: $showComparison) {
            NavigationStack {
                ModelComparisonView(points: session.points, configuration: session.configuration,
                                    regression: session.isRegression, datasetName: session.dataset.title) { model in
                    session.select(model); showComparison = false
                }
                .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Done") { showComparison = false } } }
            }
        }
    }

    private var brandHeader: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Classical ML").font(.system(textSize.isAccessibilitySize ? .headline : .title2, design: .rounded, weight: .semibold)).foregroundStyle(MLTheme.ink)
                if !textSize.isAccessibilitySize { Engraving("Simulator / 13 living models") }
            }
            Spacer()
                Image(systemName: "point.3.connected.trianglepath.dotted")
                .font(.system(size: 24, weight: .light)).foregroundStyle(MLTheme.cyan)
                .frame(width: 46, height: 46)
                .background(MLTheme.panel, in: .rect(cornerRadius: 14))
                .overlay { RoundedRectangle(cornerRadius: 14).stroke(MLTheme.cyan.opacity(0.2), lineWidth: 0.7) }
                .accessibilityHidden(true)
        }.padding(.top, 12).padding(.bottom, 4)
    }

    private var modelHeader: some View {
        Button { session.isPlaying = false; showModels = true } label: {
            HStack(spacing: 12) {
                Image(systemName: session.algorithm.symbol)
                    .font(.system(size: 23)).foregroundStyle(MLTheme.classColor(session.algorithm.accentIndex))
                    .frame(width: 38)
                VStack(alignment: .leading, spacing: 3) {
                    Engraving(session.algorithm.family)
                    Text(session.algorithm.title).font(.system(textSize.isAccessibilitySize ? .headline : .title2, design: .rounded, weight: .semibold))
                        .foregroundStyle(MLTheme.ink).multilineTextAlignment(.leading).fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.up.chevron.down").foregroundStyle(MLTheme.muted).font(.footnote)
            }.padding(15).frame(maxWidth: .infinity, minHeight: 74)
                .glassEffect(.regular.interactive(), in: .rect(cornerRadius: 22))
        }.buttonStyle(.plain).accessibilityLabel("Choose model, \(session.algorithm.title)")
            .accessibilityIdentifier("chooseModel")
    }

    private var plotInstrument: some View {
        VStack(spacing: 10) {
            HStack {
                Engraving(session.algorithm == .monteCarlo ? "Sampling chamber" : "Feature space")
                Spacer()
                StatusBadge(text: session.isComputing ? "COMPUTING" : session.isPlaying ? "RUNNING" : session.frameIndex == 0 ? "READY" : "PAUSED",
                            color: session.isPlaying ? MLTheme.amber : MLTheme.cyan)
                    .accessibilityElement(children: .combine).accessibilityIdentifier("simulationStatus")
            }.padding(.horizontal, 4)
            MLPlotView(algorithm: session.algorithm, frame: session.frame, probe: session.probe) { x, y in
                session.touchPlot(x: x, y: y)
            }
            .frame(height: textSize.isAccessibilitySize ? 290 : 310)
            .opacity(session.isComputing ? 0.45 : 1)
            .overlay { if session.isComputing { ProgressView().tint(MLTheme.cyan).accessibilityLabel("Computing model") } }
            .allowsHitTesting(!session.isComputing)
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: session.isEditing ? "pencil.tip" : "hand.tap").foregroundStyle(MLTheme.cyan)
                Text(session.probeDescription ?? (session.isEditing ? "Tap the plot to add a training sample." : "Tap to inspect. Step to see the model change."))
                    .font(.caption).foregroundStyle(MLTheme.muted)
                if session.probe != nil {
                    Spacer(minLength: 0)
                    Button { session.probe = nil } label: { Image(systemName: "xmark.circle.fill").frame(minWidth: 32, minHeight: 32) }
                        .accessibilityLabel("Clear probe")
                }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 4)
            legend
        }
        .padding(12)
        .background {
            RoundedRectangle(cornerRadius: 26).fill(LinearGradient(colors: [Color(red: 0.055, green: 0.073, blue: 0.09), Color(red: 0.021, green: 0.032, blue: 0.042)], startPoint: .topLeading, endPoint: .bottomTrailing))
                .shadow(color: .black.opacity(0.5), radius: 14, y: 9)
        }
        .overlay { RoundedRectangle(cornerRadius: 26).stroke(LinearGradient(colors: [.white.opacity(0.25), .white.opacity(0.035), .white.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 1).allowsHitTesting(false) }
    }

    private var legend: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: textSize.isAccessibilitySize ? 220 : 100), alignment: .leading)], alignment: .leading, spacing: 6) {
            legendItems
        }.font(.system(.caption2, design: .monospaced)).foregroundStyle(MLTheme.muted)
            .frame(minHeight: !textSize.isAccessibilitySize && (session.algorithm == .kMeans || session.algorithm == .dbscan) ? 62 : nil, alignment: .top)
            .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 4)
    }

    @ViewBuilder private var legendItems: some View {
        if session.algorithm == .monteCarlo {
            Label("INSIDE", systemImage: "circle.fill").foregroundStyle(MLTheme.cyan)
            Label("OUTSIDE", systemImage: "triangle.fill").foregroundStyle(MLTheme.amber)
        } else if session.algorithm == .pca {
            Label("SAMPLES", systemImage: "circle.fill").foregroundStyle(MLTheme.cyan)
            Label("PC1", systemImage: "arrow.up.right").foregroundStyle(MLTheme.amber)
            Label("PC2", systemImage: "arrow.up.right").foregroundStyle(MLTheme.violet)
        } else if session.isRegression {
            Label("SAMPLES", systemImage: "circle.fill").foregroundStyle(MLTheme.cyan)
            Label("FITTED CURVE", systemImage: "line.diagonal").foregroundStyle(MLTheme.amber)
            Text("HOLLOW = TEST")
        } else {
            Label(session.isUnsupervised ? "GROUP A" : "CLASS A", systemImage: "circle.fill").foregroundStyle(MLTheme.cyan)
            Label(session.isUnsupervised ? "GROUP B" : "CLASS B", systemImage: "triangle.fill").foregroundStyle(MLTheme.amber)
            ForEach(Array(Set(session.frame.points.map(\.label)).filter { $0 >= 2 && $0 < 6 }.sorted()), id: \.self) { group in
                Label(String(UnicodeScalar(65 + min(25, group))!), systemImage: group % 3 == 0 ? "circle.fill" : group % 3 == 1 ? "triangle.fill" : "square.fill")
                    .foregroundStyle(MLTheme.classColor(group))
            }
            let extraGroups = Set(session.frame.points.map(\.label)).filter { $0 >= 6 }.count
            if extraGroups > 0 { Text("+\(extraGroups) GROUPS") }
            if session.algorithm == .dbscan {
                Label("NOISE", systemImage: "xmark").foregroundStyle(MLTheme.muted)
                Label("UNVISITED", systemImage: "circle").foregroundStyle(MLTheme.muted)
            }
            Text("HOLLOW = TEST")
        }
    }

    private var historyDescription: String {
        switch session.algorithm {
        case .linearRegression, .polynomialRegression: "Training mean squared error per epoch. Lower means smaller residuals on training data."
        case .svm: "Training hinge loss per epoch. Lower means fewer or smaller margin violations."
        case .kMeans: "Training inertia: sum of squared distances to the assigned centers. Lower means tighter groups."
        case .dbscan: "Percentage of visited training samples currently marked as noise. This can change as neighborhoods expand."
        case .pca: "Training variance captured by the first principal component, as observations arrive."
        case .monteCarlo: "Running π estimate. Sampling error fluctuates; a single path need not improve at every step."
        default: "Training log loss through the recorded model states. Lower means more confident, correct class probabilities."
        }
    }

    private var metricsPanel: some View {
        LabPanel(minimumHeight: session.isComputing ? telemetryHeight : nil) {
            HStack(alignment: .firstTextBaseline) {
                Engraving("Live telemetry")
                Spacer()
                Text("\(session.frame.step) \(session.algorithm.stepUnit)")
                    .font(.system(.caption, design: .monospaced)).monospacedDigit().foregroundStyle(MLTheme.cyan)
            }
            let columns = [GridItem(.adaptive(minimum: textSize.isAccessibilitySize ? 160 : 100), alignment: .leading)]
            LazyVGrid(columns: columns, alignment: .leading, spacing: 16) {
                ForEach(session.frame.metrics) { metric in MetricReadout(metric: metric) }
            }
            if session.frame.metrics.isEmpty { Text("Waiting for the first calculation…").font(.caption).foregroundStyle(MLTheme.muted) }
            Divider().overlay(.white.opacity(0.08))
            VStack(alignment: .leading, spacing: 5) {
                Text(session.frame.caption).font(.system(.subheadline, design: .rounded, weight: .semibold)).foregroundStyle(MLTheme.ink)
                Text(session.frame.detail).font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal: false, vertical: true)
            }
        }
        .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { height in
            if !session.isComputing { telemetryHeight = height }
        }
        .opacity(session.isComputing ? 0.5 : 1)
    }

    private var datasetPanel: some View {
        LabPanel("01 / Dataset") {
            if session.algorithm == .monteCarlo {
                Text("Uniform random points in the unit square. Count the fraction inside a quarter circle to estimate π.")
                    .font(.subheadline).foregroundStyle(MLTheme.ink)
            } else {
                Picker("Dataset", selection: Binding(get: { session.dataset }, set: { session.changeDataset($0) })) {
                    ForEach(DatasetKind.allCases.filter { session.algorithm.supportsDataset($0) }) { kind in
                        Text(kind.title).tag(kind)
                    }
                }.pickerStyle(.menu).tint(MLTheme.cyan).accessibilityIdentifier("datasetPicker")
                Text(session.dataset.detail).font(.caption).foregroundStyle(MLTheme.muted)
            }
            Stepper(value: Binding(get: { session.configuration.sampleCount }, set: { value in
                session.configuration.sampleCount = value; session.recompute(regenerate: true)
            }), in: session.algorithm == .monteCarlo ? 100...2000 : 30...180, step: session.algorithm == .monteCarlo ? 100 : 10) {
                parameterLabel("Samples", value: "\(session.configuration.sampleCount)")
            }.accessibilityIdentifier("sampleStepper")
            if session.algorithm != .monteCarlo {
                VStack(spacing: 4) {
                    parameterLabel("Noise", value: String(format: "%.2f", session.configuration.noise))
                    Slider(value: Binding(get: { session.configuration.noise }, set: { value in
                        guard value != session.configuration.noise else { return }
                        session.configuration.noise = value; session.recompute(regenerate: true)
                    }), in: 0...0.22, step: 0.01)
                        .accessibilityLabel("Dataset noise").accessibilityIdentifier("datasetNoise")
                }
            }
            HStack {
                Text("SEED \(session.configuration.seed)").font(.system(.caption, design: .monospaced)).foregroundStyle(MLTheme.muted).accessibilityIdentifier("seedReadout")
                Spacer()
                Button { session.newSeed() } label: { Label("Reseed", systemImage: "shuffle").frame(minHeight: 44) }.font(.subheadline).accessibilityIdentifier("reseedDataset")
            }
            if session.algorithm != .monteCarlo {
                Divider()
                Toggle(isOn: $session.isEditing) { Label("Draw training samples", systemImage: "pencil.and.outline") }
                    .font(.subheadline).accessibilityIdentifier("drawToggle")
                if session.isEditing {
                    if !session.isRegression && !session.isUnsupervised {
                        Picker("New sample class", selection: $session.editLabel) {
                            Text("A ●").tag(0); Text("B ▲").tag(1)
                        }.pickerStyle(.segmented).accessibilityIdentifier("editClassPicker")
                    }
                    HStack {
                        Text("\(session.customPointCount) added · \(max(0, 220 - session.points.count)) spaces left")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                        Spacer()
                        Button("Undo point") { session.removeLastPoint() }.font(.caption).frame(minHeight: 44).accessibilityIdentifier("undoPoint")
                            .disabled(session.customPointCount == 0)
                    }
                    Text("You can also add the plot center using the accessible button below.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                    Button("Add sample at center") { session.touchPlot(x: 0.5, y: 0.5) }.frame(minHeight: 44).accessibilityIdentifier("addCenterPoint")
                        .disabled(session.points.count >= 220)
                }
                Text(session.isUnsupervised ? "The centers, clusters, and projection are fit on training points. Hollow points are held out." : "Every fifth generated sample is held out. Test points never fit the model. Added samples train the model.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                if !session.isUnsupervised {
                    Button { session.isPlaying = false; showComparison = true } label: {
                        Label("Compare on this dataset", systemImage: "square.split.2x2").frame(maxWidth: .infinity, minHeight: 44)
                    }.buttonStyle(.glass).accessibilityIdentifier("compareModels")
                }
            }
        }
    }

    private var lessonPanel: some View {
        LabPanel("03 / The idea") {
            Text(session.algorithm.tagline).font(.system(.title3, design: .serif)).italic().foregroundStyle(MLTheme.cyan)
            Text(session.algorithm.equation).font(.system(.callout, design: .monospaced)).foregroundStyle(MLTheme.ink)
                .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            Text(session.algorithm.explanation).font(.subheadline).foregroundStyle(MLTheme.muted)
            ForEach(Array(session.algorithm.steps.enumerated()), id: \.offset) { index, step in
                HStack(alignment: .top, spacing: 12) {
                    Text(String(format: "%02d", index + 1)).font(.system(.caption, design: .monospaced)).foregroundStyle(MLTheme.cyan).padding(.top, 2)
                    Text(step).font(.subheadline).foregroundStyle(MLTheme.ink)
                }
            }
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "info.circle").foregroundStyle(MLTheme.amber)
                Text(session.algorithm.limitation).font(.caption).foregroundStyle(MLTheme.muted)
            }.padding(12).background(MLTheme.amber.opacity(0.045), in: .rect(cornerRadius: 12))
        }
    }

    private var transport: some View {
        let timelineLayout = textSize.isAccessibilitySize
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: 4))
            : AnyLayout(HStackLayout(spacing: 8))
        return VStack(spacing: 8) {
            timelineLayout {
                Text("FRAME \(session.frameIndex + 1) / \(max(1, session.result.frames.count))")
                    .font(.system(.caption2, design: .monospaced)).monospacedDigit().foregroundStyle(MLTheme.muted)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("frameReadout")
                Slider(value: Binding(get: { Double(session.frameIndex) }, set: { session.seek(to: Int($0)) }),
                       in: 0...Double(max(1, session.result.frames.count - 1)), step: 1)
                    .accessibilityLabel("Simulation frame").accessibilityIdentifier("frameScrubber").disabled(session.result.frames.count < 2 || session.isComputing)
            }
            GlassEffectContainer(spacing: 12) {
                HStack(spacing: 12) {
                    transportIconButton(symbol: "backward.end", label: "Reset simulation") { session.reset() }
                    transportIconButton(symbol: "backward.frame", label: "Previous step") { session.step(-1) }
                        .disabled(session.frameIndex == 0)
                    Button { session.togglePlayback() } label: {
                        Group {
                            if textSize.isAccessibilitySize {
                                Image(systemName: session.isPlaying ? "pause.fill" : "play.fill")
                            } else {
                                Label(session.isPlaying ? "Pause" : "Run", systemImage: session.isPlaying ? "pause.fill" : "play.fill")
                            }
                        }
                            .font(.system(.body, design: .rounded, weight: .semibold))
                            .frame(minWidth: 44, maxWidth: .infinity, minHeight: 48)
                            .foregroundStyle(session.isPlaying ? MLTheme.amber : MLTheme.cyan)
                            .contentShape(Rectangle())
                    }.buttonStyle(.plain)
                        .frame(minWidth: 44, maxWidth: .infinity, minHeight: 48)
                        .contentShape(Rectangle())
                        .glassEffect(.regular.tint((session.isPlaying ? MLTheme.amber : MLTheme.cyan).opacity(0.12)).interactive(), in: .capsule)
                        .accessibilityLabel(session.isPlaying ? "Pause simulation" : "Run simulation").accessibilityIdentifier("runSimulation")
                        .disabled(!session.canPlay)
                    transportIconButton(symbol: "forward.frame", label: "Next step") { session.step() }
                        .disabled(session.frameIndex >= session.result.frames.count - 1)
                    Menu {
                        Picker("Playback speed", selection: $session.speed) {
                            Text("0.5×").tag(0.5); Text("1×").tag(1.0); Text("2×").tag(2.0)
                        }
                    } label: {
                        Text(session.speed.formatted(.number.precision(.fractionLength(0...1))) + "×")
                            .font(.system(size: 13, weight: .medium, design: .monospaced))
                            .frame(width: 44, height: 48).contentShape(Rectangle())
                    }.frame(width: 44, height: 48)
                        .contentShape(Rectangle())
                        .glassEffect(.regular.interactive(), in: .capsule)
                        .accessibilityLabel("Playback speed, \(session.speed.formatted(.number.precision(.fractionLength(0...1)))) times").accessibilityIdentifier("playbackSpeed")
                }
            }.disabled(session.isComputing)
        }
        .padding(.horizontal, 18).padding(.top, 10).padding(.bottom, 8)
        .background(MLTheme.canvas.opacity(0.96))
        .overlay(alignment: .top) { Rectangle().fill(.white.opacity(0.08)).frame(height: 0.5) }
        .sensoryFeedback(.selection, trigger: session.isPlaying)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("simulationTransport")
    }

    private func transportIconButton(symbol: String, label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol).font(.system(size: 17, weight: .semibold))
                .foregroundStyle(MLTheme.cyan).frame(width: 48, height: 48)
                .contentShape(Rectangle())
        }.buttonStyle(.plain)
            .frame(width: 48, height: 48)
            .contentShape(Rectangle())
            .glassEffect(.regular.tint(MLTheme.cyan.opacity(0.08)).interactive(), in: .circle)
            .accessibilityLabel(label)
    }

    private func parameterLabel(_ title: String, value: String) -> some View {
        HStack { Text(title).foregroundStyle(MLTheme.ink); Spacer(); Text(value).monospacedDigit().foregroundStyle(MLTheme.cyan) }
            .font(.system(.subheadline, design: .rounded))
    }
}

struct MetricReadout: View {
    let metric: MLMetric
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(metric.name).font(.system(.caption2, design: .monospaced)).foregroundStyle(MLTheme.muted)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(formatted).font(.system(.title2, design: .rounded, weight: .medium)).monospacedDigit()
                    .foregroundStyle(metric.name.lowercased().contains("test") ? MLTheme.amber : MLTheme.cyan)
                Text(metric.unit).font(.caption).foregroundStyle(MLTheme.muted)
            }
        }.accessibilityElement(children: .combine)
    }
    private var formatted: String {
        if !metric.value.isFinite { return "—" }
        if metric.unit == "%" { return String(format: "%.1f", metric.value) }
        if metric.name.lowercased().contains("mse") || abs(metric.value) < 0.01 && metric.value != 0 { return String(format: "%.4f", metric.value) }
        if metric.value.rounded() == metric.value { return String(format: "%.0f", metric.value) }
        return String(format: "%.3f", metric.value)
    }
}
