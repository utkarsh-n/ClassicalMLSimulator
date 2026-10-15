import XCTest

/// Native workflows on fresh seeded data. Hittability alone is insufficient:
/// controls must also lie inside the content viewport above the transport.
@MainActor
final class ClassicalMLSimulatorUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    func testLinearRegressionControlsAndAllTransportActions() {
        let app = launch("linearRegression", "Linear Regression")
        slider("Learning rate · η", 0, "0.02", app)
        slider("Learning rate · η", 1, "0.80", app)
        slider("Regularization · λ", 1, "0.20", app)
        integer("Iterations", 40, 10, step: 5, app)
        transport(app)
        healthy(app)
    }

    func testPolynomialRegressionDegreeBothBounds() {
        let app = launch("polynomialRegression", "Polynomial Regression")
        integer("Polynomial degree", 3, 1, app)
        integer("Polynomial degree", 1, 6, app)
        slider("Learning rate · η", 1, "0.80", app)
        slider("Regularization · λ", 0, "0.00", app)
        integer("Iterations", 40, 10, step: 5, app)
        dataset("Linear Trend", app)
        end(app); healthy(app)
    }

    func testLogisticRegressionRateAndPenaltyBounds() {
        let app = launch("logisticRegression", "Logistic Regression")
        slider("Learning rate · η", 0, "0.02", app)
        slider("Regularization · λ", 0, "0.00", app)
        slider("Regularization · λ", 1, "0.20", app)
        integer("Iterations", 40, 10, step: 5, app)
        dataset("Concentric Circles", app)
        end(app); healthy(app)
    }

    func testNearestNeighborsBoundsAndProbe() {
        let app = launch("knn", "k-Nearest Neighbors")
        integer("Neighbors · k", 5, 1, step: 2, app)
        integer("Neighbors · k", 1, 21, step: 2, app)
        reveal(plot(app), app)
        plot(app).coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let clear = app.buttons["Clear probe"]
        XCTAssertTrue(clear.waitForExistence(timeout: 3))
        reveal(clear, app)
        XCTAssertTrue(plot(app).label.contains("Probe at"))
        clear.tap()
        XCTAssertFalse(plot(app).label.contains("Probe at"))
    }

    func testDecisionTreeDepthBoundsAndReadableNodes() {
        let app = launch("decisionTree", "Decision Tree")
        integer("Maximum depth", 3, 1, app)
        integer("Maximum depth", 1, 6, app)
        end(app)
        let node = element("treeNode-0", app)
        reveal(node, app)
        XCTAssertTrue(node.label.contains("Depth 0"))
        XCTAssertTrue(node.label.contains("class A") || node.label.contains("class B"))
        screenshot(app, "Maximum depth decision tree")
    }

    func testRandomForestTreeCountAndDepth() {
        let app = launch("randomForest", "Random Forest")
        integer("Trees", 12, 3, step: 3, app)
        integer("Trees", 3, 30, step: 3, app)
        integer("Maximum depth", 3, 1, app)
        integer("Maximum depth", 1, 6, app)
        end(app)
        let text = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH %@", "One representative tree.")).firstMatch
        reveal(text, app)
        XCTAssertTrue(text.label.contains("leaf probabilities"))
        healthy(app)
    }

    func testGradientBoostingStageBoundsAndRate() {
        let app = launch("gradientBoosting", "Gradient Boosting")
        integer("Boosting stages", 12, 3, step: 3, app)
        integer("Boosting stages", 3, 30, step: 3, app)
        slider("Learning rate · η", 1, "0.80", app)
        end(app)
        let leaf = app.descendants(matching: .any).matching(NSPredicate(format: "label CONTAINS %@", "Score correction.")).firstMatch
        reveal(leaf, app)
        XCTAssertTrue(leaf.label.contains("Δscore"))
    }

    func testGaussianNaiveBayesEvidenceAndMeans() {
        let app = launch("naiveBayes", "Gaussian Naive Bayes")
        XCTAssertFalse(app.sliders["parameter-Learning rate · η"].exists)
        step(app.steppers["sampleStepper"], "Increment", 1, app)
        end(app)
        label(plot(app), "100 samples")
        XCTAssertTrue(plot(app).label.contains("20 held out"))
        XCTAssertTrue(plot(app).label.contains("2 class means"))
        healthy(app)
    }

    func testLinearSVMPenaltyAndXOR() {
        let app = launch("svm", "Linear Support Vector Machine")
        slider("Learning rate · η", 1, "0.80", app)
        slider("Regularization · λ", 0, "0.00", app)
        integer("Iterations", 40, 10, step: 5, app)
        dataset("XOR", app)
        end(app)
        let text = app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "cannot solve XOR directly")).firstMatch
        reveal(text, app)
        XCTAssertTrue(text.exists)
        healthy(app)
    }

    func testKMeansClusterBoundsAndIterationLimit() {
        let app = launch("kMeans", "k-Means Clustering")
        XCTAssertTrue(plot(app).label.contains("3 cluster centers"))
        integer("Clusters · k", 3, 2, app)
        integer("Clusters · k", 2, 6, app)
        integer("Iterations", 40, 10, step: 5, app)
        end(app)
        label(plot(app), "6 cluster centers")
        healthy(app)
    }

    func testDBSCANDensityExtremes() {
        let app = launch("dbscan", "DBSCAN")
        slider("Neighborhood radius · ε", 0, "0.05", app)
        slider("Neighborhood radius · ε", 1, "0.30", app)
        integer("Minimum neighbors", 4, 2, app)
        integer("Minimum neighbors", 2, 12, app)
        end(app)
        XCTAssertFalse(plot(app).label.contains("cluster centers"))
        healthy(app)
    }

    func testPCAMinimumSampleCountAndAlternateData() {
        let app = launch("pca", "Principal Component Analysis")
        boundary(app.steppers["sampleStepper"], "Decrement", 30, maxTaps: 6, app)
        dataset("Blobs", app)
        end(app)
        label(plot(app), "30 samples")
        XCTAssertTrue(plot(app).label.contains("6 held out"))
        XCTAssertTrue(plot(app).label.contains("training mean"))
        healthy(app)
    }

    func testMonteCarloFullSampleRangeAndReseed() {
        let app = launch("monteCarlo", "Monte Carlo Sampling")
        XCTAssertFalse(app.buttons["datasetPicker"].exists)
        XCTAssertFalse(app.switches["drawToggle"].exists)
        boundary(app.steppers["sampleStepper"], "Decrement", 100, maxTaps: 10, app)
        end(app); label(plot(app), "100 sampled points")
        boundary(app.steppers["sampleStepper"], "Increment", 2000, maxTaps: 20, app)
        end(app); label(plot(app), "2000 sampled points")
        XCTAssertTrue(plot(app).label.contains("No train/test split applies"))
        reseed(app)
        end(app); healthy(app)
        screenshot(app, "Monte Carlo 2000 samples")
    }

    func testDatasetNoiseReseedDrawingAndUndoAtMaximumSamples() {
        let app = launch("knn", "k-Nearest Neighbors")
        boundary(app.steppers["sampleStepper"], "Increment", 180, maxTaps: 9, app)
        let noise = app.sliders["datasetNoise"]
        reveal(noise, app)
        noise.adjust(toNormalizedSliderPosition: 0); ready(app)
        noiseReadout("0.00", app)
        noise.adjust(toNormalizedSliderPosition: 1); ready(app)
        noiseReadout("0.22", app)
        dataset("Concentric Circles", app)
        reseed(app)
        let draw = editingSwitch(app)
        reveal(draw, app); draw.tap()
        let classB = app.segmentedControls["editClassPicker"].buttons["B ▲"]
        reveal(classB, app); classB.tap()
        let add = app.buttons["addCenterPoint"]
        reveal(add, app); add.tap(); ready(app); end(app)
        label(plot(app), "181 samples")
        let undo = app.buttons["undoPoint"]
        reveal(undo, app)
        XCTAssertTrue(undo.isEnabled)
        undo.tap(); ready(app); end(app)
        label(plot(app), "180 samples")
        XCTAssertFalse(undo.isEnabled)
        reveal(plot(app), app)
        plot(app).coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        ready(app); end(app); label(plot(app), "181 samples")
        reveal(draw, app); draw.tap()
        XCTAssertFalse(app.buttons["addCenterPoint"].exists)
    }

    func testComparisonSearchAndAllTabs() {
        let app = launch("linearRegression", "Linear Regression")
        let compare = app.buttons["compareModels"]
        reveal(compare, app); compare.tap()
        let polynomial = app.buttons["Open Polynomial Regression lab"]
        XCTAssertTrue(polynomial.waitForExistence(timeout: 10))
        reveal(polynomial, app); polynomial.tap(); ready(app)
        label(app.buttons["chooseModel"], "Polynomial Regression")

        // Cover the header sheet as well as the separate Models tab. Dismissing
        // preserves the model; selecting a row changes the actual workbench.
        let chooser = app.buttons["chooseModel"]
        reveal(chooser, app); chooser.tap()
        let done = app.buttons["Done"]
        XCTAssertTrue(done.waitForExistence(timeout: 5)); done.tap()
        label(chooser, "Polynomial Regression")
        chooser.tap()
        let forestRow = app.buttons["model-randomForest"]
        reveal(forestRow, app); forestRow.tap(); ready(app)
        label(chooser, "Random Forest")

        app.tabBars.buttons["Models"].tap()
        let search = searchField(app)
        search.tap(); search.typeText("forest")
        XCTAssertTrue(app.buttons["model-randomForest"].waitForExistence(timeout: 3))
        XCTAssertFalse(app.buttons["model-linearRegression"].exists)
        search.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: 6) + "zzzzzz")
        XCTAssertEqual(app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "model-")).count, 0)
        search.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: 6) + "PCA")
        let pca = app.buttons["model-pca"]
        XCTAssertTrue(pca.waitForExistence(timeout: 3))
        reveal(pca, app); pca.tap(); ready(app)
        label(app.buttons["chooseModel"], "Principal Component Analysis")
        app.tabBars.buttons["Field Guide"].tap()
        let features = app.staticTexts["Features"]
        reveal(features, app)
        XCTAssertTrue(features.exists)
        let boundaries = app.staticTexts["MODEL BOUNDARIES"]
        reveal(boundaries, app)
        XCTAssertTrue(boundaries.exists)
        app.tabBars.buttons["Workbench"].tap()
        XCTAssertTrue(app.buttons["runSimulation"].isHittable)
        healthy(app)
    }

    func testClassificationComparisonUsesAllSevenModels() {
        let app = launch("logisticRegression", "Logistic Regression")
        let compare = app.buttons["compareModels"]
        reveal(compare, app); compare.tap()
        let titles = ["Logistic Regression", "k-Nearest Neighbors", "Decision Tree", "Random Forest",
                      "Gradient Boosting", "Gaussian Naive Bayes", "Linear Support Vector Machine"]
        for title in titles {
            XCTAssertTrue(app.buttons["Open \(title) lab"].waitForExistence(timeout: 15),
                          "Each classifier must produce a comparison row on the same dataset.")
        }
        let svm = app.buttons["Open Linear Support Vector Machine lab"]
        reveal(svm, app); svm.tap(); ready(app)
        label(app.buttons["chooseModel"], "Linear Support Vector Machine")
        healthy(app)
    }

    func testAccessibilityXXXLControlFramesAndReachability() {
        let app = launch("linearRegression", "Linear Regression", accessibilityText: true)
        let controls = [app.buttons["Reset simulation"], app.buttons["Previous step"], app.buttons["runSimulation"], app.buttons["Next step"], app.buttons["playbackSpeed"]]
        for control in controls {
            XCTAssertTrue(control.exists && control.isHittable)
            XCTAssertGreaterThanOrEqual(control.frame.width, 44)
            XCTAssertGreaterThanOrEqual(control.frame.height, 44)
            XCTAssertTrue(app.frame.contains(control.frame))
        }
        for pair in zip(controls, controls.dropFirst()) {
            XCTAssertLessThanOrEqual(pair.0.frame.intersection(pair.1.frame).width, 1)
        }
        let increment = stepperButton(app.steppers["sampleStepper"], "Increment")
        reveal(increment, app)
        XCTAssertTrue(viewport(app).contains(increment.frame))
        increment.tap(); ready(app)
        label(plot(app), "100 samples")
        let rate = app.sliders["parameter-Learning rate · η"]
        reveal(rate, app)
        XCTAssertTrue(rate.isHittable)
        screenshot(app, "Accessibility XXXL controls")
    }

    private func launch(_ model: String, _ title: String, accessibilityText: Bool = false) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = ["--model", model]
        if accessibilityText { app.launchArguments += ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryAccessibilityXXXL"] }
        app.launch()
        XCTAssertTrue(app.buttons["chooseModel"].waitForExistence(timeout: 10))
        launchDiagnostics(app, model: model)
        ready(app); label(app.buttons["chooseModel"], title)
        XCTAssertTrue(plot(app).exists)
        return app
    }

    private func element(_ id: String, _ app: XCUIApplication) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: id).firstMatch
    }
    private func plot(_ app: XCUIApplication) -> XCUIElement { element("simulationPlot", app) }
    private func parameter(_ name: String, _ app: XCUIApplication) -> XCUIElement { app.steppers["parameter-\(name)"] }
    private func stepperButton(_ stepper: XCUIElement, _ direction: String) -> XCUIElement {
        // SwiftUI gives each native child a stable identifier and prefixes its
        // label with the control's current value. Keep every lookup scoped to
        // this stepper so a neighboring control cannot satisfy the query.
        let identifier = stepper.identifier
        if !identifier.isEmpty {
            let identified = stepper.buttons["\(identifier)-\(direction)"]
            if identified.exists { return identified }
        }
        return stepper.buttons.matching(NSPredicate(format: "label == %@ OR label ENDSWITH %@", direction, ", \(direction)")).firstMatch
    }
    private func editingSwitch(_ app: XCUIApplication) -> XCUIElement {
        let identified = element("drawToggle", app)
        if identified.exists && identified.elementType == .switch { return identified }
        if identified.exists && identified.switches.firstMatch.exists { return identified.switches.firstMatch }
        return app.switches["Draw training samples"]
    }
    private func searchField(_ app: XCUIApplication) -> XCUIElement {
        let search = app.searchFields.firstMatch
        // Search is navigation chrome, which is outside the content viewport.
        // Some system presentations collapse it to a Search button initially.
        for _ in 0..<4 {
            if search.exists && search.isHittable { return search }
            let button = app.navigationBars.buttons["Search"].firstMatch
            if button.exists && button.isHittable { button.tap() }
            else { scroll(app).swipeDown() }
        }
        XCTAssertTrue(search.exists && search.isHittable, "The native model search field must be reachable.")
        return search
    }

    private func acceptedInteger(_ stepper: XCUIElement, file: StaticString = #filePath, line: UInt = #line) -> Int? {
        // Native UIStepper exposes the accepted number separately from its
        // children. Fall back to the last comma-separated label component.
        let rawValue = stepper.value
        if let text = rawValue as? String, let value = Double(text.trimmingCharacters(in: .whitespacesAndNewlines)),
           value.isFinite, value.rounded() == value, let integer = Int(exactly: value) { return integer }
        if let number = rawValue as? NSNumber, let integer = Int(exactly: number.doubleValue) { return integer }
        let label = stepper.label
        if let tail = label.split(separator: ",").last,
           let integer = Int(tail.trimmingCharacters(in: .whitespacesAndNewlines)) { return integer }
        XCTFail("Native stepper must expose an accepted integer: value=\(String(describing: rawValue)), label=\(label)", file: file, line: line)
        return nil
    }
    private func assertBoundary(_ stepper: XCUIElement, _ direction: String, _ expected: Int, _ app: XCUIApplication,
                                file: StaticString = #filePath, line: UInt = #line) {
        let child = stepperButton(stepper, direction)
        reveal(child, app, file: file, line: line)
        XCTAssertEqual(acceptedInteger(stepper, file: file, line: line), expected, "The native control must reach its exact bound.", file: file, line: line)
        // On iOS 26.5 the dimmed UIStepper child reports isEnabled=true.
        // Verify the actual limit by attempting one more outward touch.
        child.tap(); ready(app)
        XCTAssertEqual(acceptedInteger(stepper, file: file, line: line), expected, "An outward touch must preserve the accepted boundary value.", file: file, line: line)
    }
    private func integer(_ name: String, _ initial: Int, _ target: Int, step stride: Int = 1, _ app: XCUIApplication) {
        let control = parameter(name, app)
        XCTAssertEqual(acceptedInteger(control), initial)
        // These workflows target the declared control endpoints. Native repeat
        // exercises held input, with finite fallback touches to reach the bound.
        boundary(control, target < initial ? "Decrement" : "Increment", target,
                 maxTaps: abs(target - initial) / stride, app)
        XCTAssertEqual(acceptedInteger(control), target, "\(name) must display its exact accepted value.")
    }
    private func step(_ stepper: XCUIElement, _ direction: String, _ count: Int, _ app: XCUIApplication) {
        let child = stepperButton(stepper, direction)
        reveal(child, app)
        for _ in 0..<count {
            reveal(child, app)
            let previous = acceptedInteger(stepper)
            child.tap(); ready(app)
            if let previous, let current = acceptedInteger(stepper) {
                if direction == "Increment" { XCTAssertGreaterThan(current, previous) }
                else { XCTAssertLessThan(current, previous) }
            }
        }
    }
    private func boundary(_ stepper: XCUIElement, _ direction: String, _ expected: Int, maxTaps: Int, _ app: XCUIApplication) {
        let child = stepperButton(stepper, direction)
        reveal(child, app)
        // Test native held repeat, then finish with bounded taps independent of
        // OS repeat cadence. Stop on the accepted value, not a child AX flag.
        child.press(forDuration: 1.2); ready(app)
        for _ in 0..<maxTaps {
            if acceptedInteger(stepper) == expected { break }
            reveal(child, app); child.tap(); ready(app)
        }
        assertBoundary(stepper, direction, expected, app)
    }
    private func slider(_ name: String, _ position: CGFloat, _ expected: String, _ app: XCUIApplication) {
        let control = app.sliders["parameter-\(name)"]
        reveal(control, app); control.adjust(toNormalizedSliderPosition: position); ready(app)
        label(app.staticTexts["parameterValue-\(name)"], expected)
    }
    private func dataset(_ name: String, _ app: XCUIApplication) {
        let picker = app.buttons["datasetPicker"]
        reveal(picker, app); picker.tap()
        let option = app.buttons[name]
        XCTAssertTrue(option.waitForExistence(timeout: 3)); option.tap(); ready(app)
        label(picker, name)
    }
    private func reseed(_ app: XCUIApplication) {
        let seed = app.staticTexts["seedReadout"], button = app.buttons["reseedDataset"]
        reveal(button, app)
        let previous = seed.label
        button.tap(); ready(app)
        changed(seed, previous)
    }
    private func noiseReadout(_ expected: String, _ app: XCUIApplication) {
        // The noise readout is the trailing Text in the Noise HStack. Match its
        // actual row so an equal telemetry or model-control value cannot pass.
        expect(app.staticTexts["Noise"], NSPredicate { object, _ in
            guard let title = object as? XCUIElement, title.exists else { return false }
            let row = title.frame
            return app.staticTexts.matching(NSPredicate(format: "label == %@", expected))
                .allElementsBoundByIndex.contains { candidate in
                    let frame = candidate.frame
                    return abs(frame.midY - row.midY) <= 2 && frame.minX > row.maxX
                }
        })
    }
    private func end(_ app: XCUIApplication) {
        let slider = app.sliders["frameScrubber"]
        // A model may legitimately converge in one recorded frame.
        if slider.isEnabled { slider.adjust(toNormalizedSliderPosition: 1) }
        XCTAssertFalse(app.buttons["Next step"].isEnabled)
        let numbers = app.staticTexts["frameReadout"].label.split { !$0.isNumber }.compactMap { Int($0) }
        XCTAssertEqual(numbers.count, 2)
        if numbers.count == 2 { XCTAssertEqual(numbers[0], numbers[1]) }
    }
    private func transport(_ app: XCUIApplication) {
        let frame = app.staticTexts["frameReadout"], run = app.buttons["runSimulation"]
        let first = frame.label
        XCTAssertTrue(first.hasPrefix("FRAME 1 /")); XCTAssertFalse(app.buttons["Previous step"].isEnabled)
        app.buttons["Next step"].tap(); changed(frame, first)
        app.buttons["Previous step"].tap(); label(frame, first)
        for (option, accepted) in [("0.5×", "0.5"), ("1×", "1"), ("2×", "2"), ("0.5×", "0.5")] {
            app.buttons["playbackSpeed"].tap()
            let choice = app.buttons[option]
            XCTAssertTrue(choice.waitForExistence(timeout: 3)); choice.tap()
            label(app.buttons["playbackSpeed"], "Playback speed, \(accepted) times")
        }
        run.tap(); label(run, "Pause"); changed(frame, first)
        run.tap(); label(run, "Run")
        end(app)
        app.sliders["frameScrubber"].adjust(toNormalizedSliderPosition: 0.5)
        XCTAssertTrue(app.buttons["Previous step"].isEnabled); XCTAssertTrue(app.buttons["Next step"].isEnabled)
        app.buttons["Reset simulation"].tap(); label(frame, first)
        XCTAssertFalse(app.buttons["Previous step"].isEnabled)
    }
    private func healthy(_ app: XCUIApplication) {
        XCTAssertTrue(plot(app).exists)
        let counts = app.staticTexts["frameReadout"].label.split { !$0.isNumber }.compactMap { Int($0) }
        XCTAssertEqual(counts.count, 2)
        if counts.count == 2 { XCTAssertGreaterThanOrEqual(counts[1], 1) }
        XCTAssertTrue(plot(app).label.contains("Axes range from zero to one"))
        XCTAssertFalse(plot(app).label.lowercased().contains("nan"))
        XCTAssertFalse(app.staticTexts["—"].exists, "Supported control limits must keep rendered metrics finite.")
    }

    private func scroll(_ app: XCUIApplication) -> XCUIElement {
        // Presented sheets can leave underlying workbench elements in the tree.
        // Prefer the foreground sheet before considering the retained tabs.
        let candidates = [app.scrollViews["comparisonScroll"], app.scrollViews["modelLibraryScroll"],
                          app.scrollViews["workbenchScroll"]]
        if let visible = candidates.first(where: { $0.exists && $0.isHittable }) { return visible }
        if let visible = app.scrollViews.allElementsBoundByIndex.first(where: { $0.isHittable }) { return visible }
        return candidates.first(where: { $0.exists }) ?? app.scrollViews.firstMatch
    }
    private func viewport(_ app: XCUIApplication, foreground suppliedForeground: XCUIElement? = nil) -> CGRect {
        let foreground = suppliedForeground ?? scroll(app)
        var area = foreground.frame.intersection(app.frame)
        let statusBar = app.statusBars.firstMatch
        if statusBar.exists && statusBar.isHittable && statusBar.frame.maxY > area.minY && statusBar.frame.maxY < area.maxY {
            let bottom = area.maxY
            area.origin.y = statusBar.frame.maxY
            area.size.height = bottom - area.minY
        }
        let navigation = app.navigationBars.firstMatch
        if navigation.exists && navigation.isHittable && navigation.frame.maxY > area.minY && navigation.frame.maxY < area.maxY {
            let bottom = area.maxY
            area.origin.y = navigation.frame.maxY
            area.size.height = bottom - area.minY
        }
        let transport = element("simulationTransport", app)
        if foreground.identifier == "workbenchScroll" {
            // A SwiftUI accessibility container can have bounds larger than its
            // visible material. Measure the actual first transport row instead.
            // MLRootView adds exactly 10 points above that HStack.
            let rowFrames = [app.staticTexts["frameReadout"], app.sliders["frameScrubber"]]
                .filter { $0.exists }.map { $0.frame }.filter { !$0.isEmpty && !$0.isNull }
            let top = rowFrames.map(\.minY).min().map { $0 - 10 }
                ?? (transport.exists ? transport.frame.minY : area.maxY)
            if top > area.minY { area.size.height = max(1, min(area.maxY, top) - area.minY) }
        }
        let tabs = app.tabBars.firstMatch
        if tabs.exists && tabs.isHittable && tabs.frame.minY > area.minY {
            let buttonFrames = tabs.buttons.allElementsBoundByIndex.map { $0.frame }
                .filter { !$0.isEmpty && !$0.isNull }
            let top = buttonFrames.map(\.minY).min().map { $0 - 8 } ?? tabs.frame.minY
            area.size.height = max(1, min(area.maxY, top) - area.minY)
        }
        if foreground.identifier == "modelLibraryScroll" {
            // Native bottom search chrome overlaps the ScrollView's AX bounds.
            // Keep content drags above it without excluding a top search field.
            let search = app.searchFields.firstMatch
            if search.exists && search.isHittable {
                let frame = search.frame
                if frame.minY > area.midY && frame.minY < area.maxY {
                    area.size.height = max(1, frame.minY - 8 - area.minY)
                }
            }
        }
        let keyboard = app.keyboards.firstMatch
        if keyboard.exists, keyboard.frame.minY > area.minY { area.size.height = max(1, min(area.maxY, keyboard.frame.minY) - area.minY) }
        return area.insetBy(dx: 3, dy: 6)
    }
    private func reveal(_ target: XCUIElement, _ app: XCUIApplication, file: StaticString = #filePath, line: UInt = #line) {
        // Chrome stays anchored while content scrolls. Reuse its geometry and
        // avoid querying the same AX containers before every drag.
        let foreground = scroll(app)
        let foregroundID = foreground.identifier
        let isSheetContent = foregroundID == "comparisonScroll" || foregroundID == "modelLibraryScroll"
        let area = viewport(app, foreground: foreground), appFrame = app.frame
        // A custom workbench header has no native navigation bar. Start both
        // scroll directions inside its content, below the system status bar.
        let endpointInset: CGFloat = min(72, area.height / 4)
        let origin = app.coordinate(withNormalizedOffset: .zero)
        for attempt in 0..<14 {
            let exists = target.exists
            let frame = exists ? target.frame : .zero
            if exists && frame.height > 0 && frame.minY >= area.minY && frame.maxY <= area.maxY && target.isHittable { return }
            if attempt >= 12 {
                print("ML reveal attempt \(attempt + 1): foreground=\(foregroundID), target=\(target), exists=\(exists), frame=\(frame), viewport=\(area)")
            }
            let up = !exists || frame.midY > area.midY
            // Sheets add another leading inset and can expose the home-indicator
            // region in AX bounds. Keep sheet gestures inside their scrollable
            // content; the workbench lane must stay outside its plot and sliders.
            let x = area.minX + (isSheetContent ? 48 : 24)
            let from = CGPoint(x: x, y: up ? area.maxY - endpointInset : area.minY + endpointInset)
            let to = CGPoint(x: x, y: up ? area.minY + area.height * 0.32 : area.maxY - endpointInset)
            origin.withOffset(CGVector(dx: from.x - appFrame.minX, dy: from.y - appFrame.minY))
                .press(forDuration: 0.01, thenDragTo: origin.withOffset(CGVector(dx: to.x - appFrame.minX, dy: to.y - appFrame.minY)),
                       withVelocity: .slow, thenHoldForDuration: 0.15)
        }
        XCTFail("Element must be visible above transport/keyboard: \(target)", file: file, line: line)
    }
    private func ready(_ app: XCUIApplication) {
        // Run is correctly disabled for a valid single-frame result. Readiness
        // belongs to the computation status, rather than playback availability.
        expect(element("simulationStatus", app), readingLabelOnce {
            $0.contains("READY") || $0.contains("PAUSED") || $0.contains("RUNNING")
        }, timeout: 15)
        expect(plot(app), readingLabelOnce {
            $0.hasPrefix("Simulation plot.") && !$0.contains("Preparing the experiment") && !$0.contains("No recorded frames")
        }, timeout: 30)
    }
    private func readingLabelOnce(_ condition: @escaping @Sendable (String) -> Bool) -> NSPredicate {
        NSPredicate { object, _ in
            guard let target = object as? XCUIElement, target.exists else { return false }
            let value = target.label
            return condition(value)
        }
    }
    private func label(_ element: XCUIElement, _ text: String) { expect(element, readingLabelOnce { $0.contains(text) }) }
    private func changed(_ element: XCUIElement, _ previous: String) { expect(element, readingLabelOnce { $0 != previous }) }
    private func expect(_ element: XCUIElement, _ predicate: NSPredicate, timeout: TimeInterval = 15,
                        file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(XCTWaiter.wait(for: [XCTNSPredicateExpectation(predicate: predicate, object: element)], timeout: timeout), .completed, file: file, line: line)
    }
    private func screenshot(_ app: XCUIApplication, _ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name; attachment.lifetime = .keepAlways; add(attachment)
    }

    private func launchDiagnostics(_ app: XCUIApplication, model: String) {
        // Preserve evidence before the first synthesized gesture: an AX service
        // failure can prevent the normal failure screenshot from being captured.
        screenshot(app, "Launch \(model)")
        let measured: [(String, XCUIElement)] = [
            ("window", app.windows.firstMatch), ("foregroundScroll", scroll(app)),
            ("transportContainer", element("simulationTransport", app)),
            ("frameReadout", app.staticTexts["frameReadout"]),
            ("frameScrubber", app.sliders["frameScrubber"]),
            ("plot", plot(app)),
            ("run", app.buttons["runSimulation"]),
            ("navigationBar", app.navigationBars.firstMatch),
            ("tabBar", app.tabBars.firstMatch),
            ("workbenchTab", app.tabBars.buttons["Workbench"]),
            ("sampleIncrement", stepperButton(app.steppers["sampleStepper"], "Increment"))
        ]
        let rows = measured.map { name, item in
            item.exists ? "\(name): \(item.frame)" : "\(name): absent"
        }
        let chart = plot(app)
        let chartLabel = chart.exists ? chart.label : "absent"
        let geometry = (["model: \(model)", "app: \(app.frame)", "contentViewport: \(viewport(app))", "plotLabel: \(chartLabel)"] + rows).joined(separator: "\n")
        let frames = XCTAttachment(string: geometry)
        frames.name = "Launch geometry \(model)"; frames.lifetime = .keepAlways; add(frames)
        print("ML launch geometry\n\(geometry)")
        let hierarchy = XCTAttachment(string: app.debugDescription)
        hierarchy.name = "Launch accessibility hierarchy \(model)"
        hierarchy.lifetime = .keepAlways; add(hierarchy)
    }
}
