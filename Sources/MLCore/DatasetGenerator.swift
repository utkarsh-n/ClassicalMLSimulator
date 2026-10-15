import Foundation

public enum DatasetGenerator {
    public static func generate(_ kind: DatasetKind, configuration: MLConfiguration) -> [MLPoint] {
        let count = min(320, max(5, configuration.sampleCount))
        let noise = bounded(configuration.noise, 0, 0.35, fallback: 0.08)
        var rng = SeededRandom(seed: configuration.seed)
        return (0..<count).map { index in
            var x: Double, y: Double, label: Int
            switch kind {
            case .blobs:
                let centers = [(0.23, 0.28), (0.76, 0.33), (0.49, 0.76)]
                label = index % centers.count
                x = centers[label].0 + rng.normal() * noise
                y = centers[label].1 + rng.normal() * noise
            case .moons:
                label = index % 2
                let angle = .pi * rng.uniform()
                x = (label == 0 ? 0.39 : 0.61) + 0.30 * cos(angle) + rng.normal() * noise * 0.5
                y = (label == 0 ? 0.40 : 0.57) + (label == 0 ? 0.30 : -0.30) * sin(angle) + rng.normal() * noise * 0.5
            case .circles:
                label = index % 2
                let angle = 2 * .pi * rng.uniform()
                let radius = (label == 0 ? 0.17 : 0.36) + rng.normal() * noise * 0.4
                x = 0.5 + radius * cos(angle)
                y = 0.5 + radius * sin(angle)
            case .xor:
                let baseX = 0.06 + 0.88 * rng.uniform(), baseY = 0.06 + 0.88 * rng.uniform()
                label = (baseX > 0.5) != (baseY > 0.5) ? 1 : 0
                x = baseX + rng.normal() * noise * 0.35
                y = baseY + rng.normal() * noise * 0.35
            case .linear:
                x = 0.04 + 0.92 * rng.uniform()
                y = 0.18 + 0.64 * x + rng.normal() * noise
                label = x > 0.5 ? 1 : 0
            case .curve:
                x = 0.04 + 0.92 * rng.uniform()
                y = 0.18 + 0.65 * pow(2 * x - 1, 2) + rng.normal() * noise
                label = abs(x - 0.5) > 0.25 ? 1 : 0
            case .anisotropic:
                let u = rng.normal() * 0.23, v = rng.normal() * (0.015 + noise * 0.45)
                x = 0.5 + 0.8660254038 * u - 0.5 * v
                y = 0.5 + 0.5 * u + 0.8660254038 * v
                label = u > 0 ? 1 : 0
            }
            // IDs and split membership remain fixed while a model trains or the noise changes.
            return MLPoint(id: index, x: unit(x), y: unit(y), label: label, isTest: index % 5 == 4)
        }
    }
}
