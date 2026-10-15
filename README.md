# Classical ML Simulator

A native SwiftUI iPhone workbench that makes classical machine learning visible. Designed with Metro Station's dark instrument surfaces, opaque metal plot housing, cyan/amber/violet signals, and native Liquid Glass controls. Requires iOS 26 or later; built with Xcode 27 and Swift 6.

## The labs

| Family | Methods |
| --- | --- |
| Regression | Linear regression, polynomial regression (degree 1–6) |
| Classification | Logistic regression, k-nearest neighbors, decision tree, random forest, gradient boosting, Gaussian naive Bayes, linear SVM |
| Clustering | K-means, DBSCAN |
| Projection | PCA |
| Sampling | Monte Carlo π estimation |

All 13 methods execute numerical algorithms locally in Swift. Monte Carlo is included as a probability/numerical-method lab, rather than a predictive ML model.

## Use the workbench

- Choose a model in the glass model selector or the searchable Models tab.
- Run, pause, step backward/forward, reset, change speed, or scrub recorded computation frames.
- Change synthetic dataset, noise, sample count, seed, and model parameters. Accepted slider changes automatically schedule a debounced fit, including accessibility adjustments.
- Tap the square feature plot to inspect a location. K-NN connects the query to its actual current training neighbors.
- Enable **Draw training samples** to add points to the plot; choose a class where relevant. Undo removes the last added point. Accessible plot actions and a center-sample button offer alternatives to precise tapping.
- Compare final results for supervised models on the exact same training/test split.
- Read the equations, model steps, assumptions, and six suggested experiments in the Field Guide.

The workbench is a scratch space. Model switches preserve compatible datasets; generating a new dataset replaces added points. Experiments are not persisted across app launches.

## Numerical contracts

Coordinates are dimensionless synthetic features in [0,1]. Regression predicts vertical target y; other ML labs use both coordinates. Every fifth generated sample is held out. Added samples train the model. Model fitting, centroid estimation, covariance, tree splitting, and neighbor memory exclude held-out points. Separate training/test metrics make this visible.

Classification is binary: source group 0 is A; any other source group is B. Clustering ignores source labels and displays arbitrary discovered group identifiers. Hollow points are held out; shapes and labels reinforce colors.

Decision regions evaluate the current model on a 24 × 24 grid. Probe classes are nearest-grid estimates. The chart and replay use recorded model state, never decorative random boundaries. The canvas is a square to preserve geometric angles and distances.

Concrete variants are explicit:

- Linear and polynomial regression use centered powers z = 2x − 1 and full-batch descent on MSE plus configurable L2 weight penalty (intercept exempt).
- Random forests bootstrap training samples, choose a random feature at each split, and average leaf probabilities. The diagram shows one representative tree.
- Boosting adds logistic residual stumps with shrunken Newton leaf scores. Its diagram shows the newest stump.
- SVM uses linear hinge loss, with no kernel trick.
- K-means uses seeded k-means++ initialization and Lloyd updates, stopping on stable assignments.
- DBSCAN reports noise and discovered clusters. Its held-out/grid rule is an explicit nearest-core-within-ε extension, since DBSCAN has no native prediction rule.
- PCA streams batches into a centered covariance eigendecomposition. Its vectors show training principal directions, not predictions.
- Monte Carlo uses independent seeded pseudorandom uniform points in a unit square and π ≈ 4 × inside / samples. Error can increase between frames.

These small synthetic experiments are not a benchmark or an exhaustive collection of ML models. References and implementation boundaries are in [Documentation/Research.md](Documentation/Research.md).

## Build

Open `ClassicalMLSimulator.xcodeproj` and run the `ClassicalMLSimulator` scheme on an iPhone simulator. The checked-in project is generated from `project.yml`:

```sh
xcodegen generate
```

The independent Foundation-only core has native tests:

```sh
swift test
```

If the host sandbox blocks the default compiler cache locations:

```sh
CLANG_MODULE_CACHE_PATH=/private/tmp/classicalml-clang-cache \
SWIFTPM_MODULECACHE_OVERRIDE=/private/tmp/classicalml-swift-cache \
swift test --disable-sandbox --scratch-path /private/tmp/classicalml-swift-build
```

The app scheme also includes 16 XCUITest flows for all model controls, data editing, playback, model switching, and accessibility text size. The numerical and controller suites contain 45 passing tests and explicit stress matrices. See [Documentation/StressTest.md](Documentation/StressTest.md) for the latest evidence and limits; [Documentation/QA.md](Documentation/QA.md) retains the initial delivery history.

## Structure

- `Sources/MLCore`: deterministic datasets, numerics, trees/ensembles, unsupervised models, recorded frames.
- `Tests/MLCoreTests`: numerical recovery, held-out isolation, determinism, and degenerate input checks.
- `App/LabSession.swift`: main-actor workbench state, cancellable computation handoff, playback, edits, and probe interpretation.
- `App/Views`: plots, tree diagrams, comparison, model library, and learning guide.
- `App/DesignSystem`: Metro Station domain adaptation; values and controls stay outside drawing updates.

No third-party runtime libraries, model downloads, accounts, analytics, or server computation are required. External reference links open only when selected.


## Public snapshot

See [PUBLICATION.md](PUBLICATION.md) for publication exclusions and setup requirements.
