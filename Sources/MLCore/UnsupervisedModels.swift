import Foundation

func closestCenter(_ point: MLPoint, centers: [MLPoint]) -> Int {
    var best = 0, distance = Double.infinity
    for index in centers.indices {
        let candidate = squaredDistance(point, centers[index])
        if candidate < distance { distance = candidate; best = index }
    }
    return best
}

extension MLEngine {
    static func kMeans(_ points: [MLPoint], _ config: MLConfiguration) -> SimulationResult {
        let train = points.filter { !$0.isTest }, test = points.filter(\.isTest)
        let k = min(8, max(1, min(config.clusters, train.count)))
        var rng = SeededRandom(seed: config.seed), selected = [rng.index(train.count)]
        var centers = [train[selected[0]]]
        // k-means++ seeds favor points far from already-selected centers.
        while centers.count < k {
            let distances = train.map { point in squaredDistance(point, centers[closestCenter(point, centers: centers)]) }
            let total = distances.reduce(0, +)
            var index = 0
            if total > 1e-12 {
                let target = rng.uniform() * total
                var cumulative = 0.0
                for i in distances.indices {
                    cumulative += distances[i]
                    if cumulative > target { index = i; break }
                }
            } else { index = train.indices.first { !selected.contains($0) } ?? 0 }
            selected.append(index); centers.append(train[index])
        }
        centers = centers.enumerated().map { MLPoint(id: 20_000 + $0.offset, x: $0.element.x, y: $0.element.y, label: $0.offset) }
        var assignments = train.map { closestCenter($0, centers: centers) }
        var frames: [SimulationFrame] = [], history: [Double] = [], converged = false
        for step in 1...steps(config) {
            let previous = centers
            for c in centers.indices {
                let group = train.indices.filter { assignments[$0] == c }
                if !group.isEmpty {
                    centers[c].x = average(group.map { train[$0].x })
                    centers[c].y = average(group.map { train[$0].y })
                }
            }
            let newAssignments = train.map { closestCenter($0, centers: centers) }
            let changes = zip(assignments, newAssignments).filter { $0 != $1 }.count
            let movement = zip(previous, centers).reduce(0) { $0 + squaredDistance($1.0, $1.1) }
            assignments = newAssignments
            converged = changes == 0 || movement <= 1e-12
            func inertia(_ samples: [MLPoint]) -> Double {
                samples.reduce(0) { $0 + squaredDistance($1, centers[closestCenter($1, centers: centers)]) }
            }
            let trainInertia = inertia(train)
            history.append(trainInertia)
            let colored = points.map { point -> MLPoint in
                var point = point; point.label = closestCenter(point, centers: centers); return point
            }
            let cells = (0..<(24 * 24)).map { i -> GridCell in
                let point = MLPoint(id: -1, x: (Double(i % 24) + 0.5) / 24, y: (Double(i / 24) + 0.5) / 24)
                return GridCell(x: point.x, y: point.y, label: closestCenter(point, centers: centers))
            }
            var metrics = [MLMetric("Train inertia", trainInertia)]
            if !test.isEmpty { metrics.append(MLMetric("Test inertia", inertia(test))) }
            metrics.append(MLMetric("Reassigned", Double(changes)))
            let stopping = changes == 0 ? "stable assignments" : "centers within tolerance"
            frames.append(SimulationFrame(step: step, caption: converged ? "Lloyd update \(step) · \(stopping)" : "Lloyd update \(step)",
                detail: "Initialize by seeded k-means++; alternate means and nearest-center assignment. Inertia = Σ squared distance to assigned center. Fit uses training points; test points only receive a center. Empty clusters retain their previous center.",
                metrics: metrics, points: colored, centers: centers, grid: cells, history: history))
            if converged { break }
        }
        return SimulationResult(frames: frames, summary: "Lloyd k-means with seeded k-means++ initialization. \(converged ? "Stopped when assignments stabilized or centroid movement reached numerical tolerance." : "Reached the update limit; convergence is not claimed.") Cluster numbers are arbitrary labels; source class labels are not used. Inertia is a sum, so sample count affects its scale.")
    }

    static func dbscan(_ points: [MLPoint], _ config: MLConfiguration) -> SimulationResult {
        let train = points.filter { !$0.isTest }, test = points.filter(\.isTest)
        let epsilon = bounded(config.epsilon, 0.005, 1, fallback: 0.14), eps2 = epsilon * epsilon
        let minPoints = min(40, max(1, config.minPoints))
        var labels = [Int](repeating: -2, count: train.count), visited = [Bool](repeating: false, count: train.count)
        var core = [Bool](repeating: false, count: train.count), visitedCount = 0, cluster = -1
        var frames: [SimulationFrame] = [], history: [Double] = []
        let stride = max(1, Int(ceil(Double(train.count) / 39)))
        var nextCapture = 1
        func neighbors(_ index: Int) -> [Int] { train.indices.filter { squaredDistance(train[index], train[$0]) <= eps2 } }
        func capture(force: Bool = false) {
            guard force || visitedCount >= nextCapture else { return }
            nextCapture = visitedCount + stride
            func assign(_ point: MLPoint) -> Int {
                var bestDistance = eps2 + 1e-12, label = -1
                for i in train.indices where core[i] {
                    let distance = squaredDistance(point, train[i])
                    if distance < bestDistance { bestDistance = distance; label = labels[i] }
                }
                return label
            }
            var trainIndex = 0
            let colored = points.map { point -> MLPoint in
                var point = point
                if point.isTest { point.label = assign(point) }
                else { point.label = labels[trainIndex]; trainIndex += 1 }
                return point
            }
            let noisePercent = 100 * Double(labels.filter { $0 == -1 }.count) / Double(max(1, visitedCount))
            if force && frames.last?.step == visitedCount { _ = history.popLast() }
            history.append(noisePercent)
            var metrics = [MLMetric("Clusters", Double(cluster + 1)), MLMetric("Train noise", noisePercent, unit: "%"), MLMetric("Visited", Double(visitedCount))]
            if !test.isEmpty {
                metrics.append(MLMetric("Test outside ε", 100 * Double(test.filter { assign($0) == -1 }.count) / Double(test.count), unit: "%"))
            }
            let cells = (0..<(24 * 24)).map { i -> GridCell in
                let p = MLPoint(id: -1, x: (Double(i % 24) + 0.5) / 24, y: (Double(i / 24) + 0.5) / 24)
                return GridCell(x: p.x, y: p.y, label: assign(p))
            }
            let corePoints = train.indices.filter { core[$0] }.map { i in MLPoint(id: train[i].id, x: train[i].x, y: train[i].y, label: labels[i]) }
            let frame = SimulationFrame(step: visitedCount, caption: "Visit \(visitedCount) / \(train.count) training samples",
                detail: "A core has at least \(minPoints) training points within ε = \(number(epsilon)), counting itself. Expand density-connected cores; border points join a cluster. Train noise is measured among visited samples. Test/grid colors use nearest fitted core within ε: an explicit extension, since DBSCAN has no native test prediction. Unvisited = −2; noise = −1.",
                metrics: metrics, points: colored, centers: corePoints, grid: cells, history: history)
            if force && frames.last?.step == visitedCount { frames[frames.count - 1] = frame }
            else { frames.append(frame) }
        }
        for start in train.indices where !visited[start] {
            visited[start] = true; visitedCount += 1
            let initial = neighbors(start)
            guard initial.count >= minPoints else {
                labels[start] = -1; capture(); continue
            }
            cluster += 1; core[start] = true; labels[start] = cluster
            var queue = initial, queued = Set(initial), cursor = 0
            capture()
            while cursor < queue.count {
                let index = queue[cursor]; cursor += 1
                if !visited[index] {
                    visited[index] = true; visitedCount += 1
                    let neighborhood = neighbors(index)
                    if neighborhood.count >= minPoints {
                        core[index] = true
                        for candidate in neighborhood where !queued.contains(candidate) {
                            queued.insert(candidate); queue.append(candidate)
                        }
                    }
                }
                if labels[index] < 0 { labels[index] = cluster }
                capture()
            }
        }
        capture(force: true)
        return SimulationResult(frames: frames, summary: "DBSCAN discovers density-connected training groups without choosing k. ε and minPts control density, including the point itself in its neighborhood. Noise can later become a border point. Held-out points do not affect clusters; their nearest-core assignment is an explicit visualization extension, not a native DBSCAN prediction or accuracy score.")
    }

    static func pca(_ points: [MLPoint], _ config: MLConfiguration) -> SimulationResult {
        let train = points.filter { !$0.isTest }, test = points.filter(\.isTest), frameCount = min(12, train.count)
        var frames: [SimulationFrame] = [], history: [Double] = []
        for step in 1...frameCount {
            let count = (step * train.count + frameCount - 1) / frameCount
            let observed = Array(train.prefix(count))
            let mx = average(observed.map(\.x)), my = average(observed.map(\.y)), denominator = Double(max(1, count - 1))
            let xx = observed.reduce(0) { $0 + pow($1.x - mx, 2) } / denominator
            let yy = observed.reduce(0) { $0 + pow($1.y - my, 2) } / denominator
            let xy = observed.reduce(0) { $0 + ($1.x - mx) * ($1.y - my) } / denominator
            let delta = sqrt(max(0, pow(xx - yy, 2) + 4 * xy * xy))
            let eigen1 = max(0, (xx + yy + delta) / 2), eigen2 = max(0, (xx + yy - delta) / 2)
            let angle = 0.5 * atan2(2 * xy, xx - yy), vx = cos(angle), vy = sin(angle)
            let total = eigen1 + eigen2, share = total > 1e-12 ? 100 * eigen1 / total : 0
            func reconstructionMSE(_ samples: [MLPoint]) -> Double {
                average(samples.map { p in
                    let projection = (p.x - mx) * vx + (p.y - my) * vy
                    return pow(p.x - mx - projection * vx, 2) + pow(p.y - my - projection * vy, 2)
                })
            }
            let length1 = min(0.43, 2 * sqrt(eigen1)), length2 = min(0.43, 2 * sqrt(eigen2))
            let axes = [MLLine(x1: mx - vx * length1, y1: my - vy * length1, x2: mx + vx * length1, y2: my + vy * length1, label: 0),
                        MLLine(x1: mx + vy * length2, y1: my - vx * length2, x2: mx - vy * length2, y2: my + vx * length2, label: 1)]
            var metrics = [MLMetric("PC1 variance", share, unit: "%"), MLMetric("Train reconstruction MSE", reconstructionMSE(train))]
            if !test.isEmpty { metrics.append(MLMetric("Test reconstruction MSE", reconstructionMSE(test))) }
            metrics.append(MLMetric("PC1 angle", angle * 180 / .pi, unit: "°"))
            history.append(share)
            frames.append(SimulationFrame(step: step, caption: "Covariance from \(count) training samples",
                detail: "Center the samples, form the sample covariance matrix (n−1 denominator), and solve its two eigenvectors. PC1 maximizes projected variance. Lines span ±2√eigenvalue, capped to fit the plot. Reconstruction projects onto PC1 only. λ₁ = \(number(eigen1)); λ₂ = \(number(eigen2)).",
                metrics: metrics, points: points, centers: [MLPoint(id: 20_000, x: mx, y: my)], vectors: axes, history: history))
        }
        return SimulationResult(frames: frames, summary: "Exact 2D PCA, recomputed from a growing training prefix to show statistical estimation. The final axes are eigenvectors of training covariance. PC1 preserves the largest variance, while held-out reconstruction MSE measures information lost by reducing to one dimension. Input features share a 0…1 scale; no whitening is applied.")
    }

    static func monteCarlo(_ config: MLConfiguration) -> SimulationResult {
        let count = min(10_000, max(20, config.sampleCount))
        let frameCount = min(count, min(40, max(1, config.iterations)))
        var rng = SeededRandom(seed: config.seed), samples: [MLPoint] = [], inside = 0
        var frames: [SimulationFrame] = [], history: [Double] = []
        for step in 1...frameCount {
            let revealed = (step * count + frameCount - 1) / frameCount
            while samples.count < revealed {
                let x = rng.uniform(), y = rng.uniform(), isInside = x * x + y * y <= 1
                if isInside { inside += 1 }
                samples.append(MLPoint(id: samples.count, x: x, y: y, label: isInside ? 0 : 1))
            }
            let estimate = 4 * Double(inside) / Double(samples.count)
            history.append(estimate)
            let proportion = Double(inside) / Double(samples.count)
            let standardError = 4 * sqrt(proportion * (1 - proportion) / Double(samples.count))
            // The visual overlay is the quarter-circle in the unit square, not a trained boundary.
            let arc = (0...80).map { i -> MLPoint in
                let angle = Double(i) / 80 * .pi / 2
                return MLPoint(id: 30_000 + i, x: cos(angle), y: sin(angle))
            }
            frames.append(SimulationFrame(step: samples.count, caption: "Sample \(samples.count) / \(count) random points",
                detail: "Draw uniform x,y ∈ [0,1]. Inside means x²+y² ≤ 1, the quarter-circle of area π/4. π estimate = 4 × \(inside)/\(samples.count). Standard error is a plug-in binomial estimate; a finite random run is not guaranteed to improve monotonically.",
                metrics: [MLMetric("π estimate", estimate), MLMetric("Absolute error", abs(estimate - .pi)), MLMetric("Std. error", standardError), MLMetric("Samples", Double(samples.count))],
                points: samples, curve: arc, history: history))
        }
        return SimulationResult(frames: frames, summary: "Monte Carlo area estimation, not a fitted ML predictor. Seeded uniform draws estimate π from a quarter-circle inside a unit square. Every frame adds real samples; estimate fluctuations and sampling error are retained. No train/test split applies to this experiment.")
    }
}
