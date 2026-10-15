import SwiftUI
import MLCore

struct ModelLibraryView: View {
    let selected: MLAlgorithm
    var onSelect: (MLAlgorithm) -> Void
    @State private var search = ""
    private let families = ["Regression", "Classification", "Clustering", "Projection", "Sampling"]
    private var filtered: [MLAlgorithm] {
        MLAlgorithm.allCases.filter { search.isEmpty || "\($0.title) \($0.rawValue) \($0.family) \($0.tagline)".localizedCaseInsensitiveContains(search) }
    }
    var body: some View {
        ZStack {
            LabBackground()
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    VStack(alignment: .leading, spacing: 8) {
                        Engraving("13 methods / 5 families")
                        Text("Make the math visible.").font(.system(.title2, design: .serif)).italic().foregroundStyle(MLTheme.cyan)
                        Text("Choose a method. Shape its data. Watch each decision emerge.")
                            .font(.subheadline).foregroundStyle(MLTheme.muted)
                    }.padding(.top, 6)
                    ForEach(families, id: \.self) { family in
                        let models = filtered.filter { $0.family == family }
                        if !models.isEmpty {
                            VStack(alignment: .leading, spacing: 10) {
                                HStack { Engraving(family); Spacer(); Engraving(String(format: "%02d", models.count)) }
                                    .padding(.horizontal, 4)
                                ForEach(models) { model in modelRow(model) }
                            }
                        }
                    }
                    if filtered.isEmpty {
                        ContentUnavailableView.search(text: search)
                    }
                    Text("Monte Carlo is a numerical sampling method, included as a foundational probability lab.")
                        .font(.caption).foregroundStyle(MLTheme.muted).padding(.bottom, 16)
                }.frame(maxWidth: 620).padding(16).frame(maxWidth: .infinity)
            }.scrollIndicators(.hidden).accessibilityIdentifier("modelLibraryScroll")
        }
        .navigationTitle("Model library")
        .searchable(text: $search, prompt: "Find a model or family")
    }
    private func modelRow(_ model: MLAlgorithm) -> some View {
        Button { onSelect(model) } label: {
            HStack(spacing: 14) {
                Image(systemName: model.symbol).font(.system(size: 24, weight: .light))
                    .foregroundStyle(MLTheme.classColor(model.accentIndex)).frame(width: 42, height: 48)
                VStack(alignment: .leading, spacing: 5) {
                    Text(model.title).font(.system(.headline, design: .rounded)).foregroundStyle(MLTheme.ink)
                    Text(model.tagline).font(.caption).foregroundStyle(MLTheme.muted).multilineTextAlignment(.leading)
                }
                Spacer(minLength: 2)
                Image(systemName: model == selected ? "checkmark.circle.fill" : "arrow.up.right")
                    .font(.subheadline).foregroundStyle(model == selected ? MLTheme.cyan : MLTheme.muted)
            }.padding(14).frame(maxWidth: .infinity, minHeight: 86, alignment: .leading)
                .background(MLTheme.panel, in: .rect(cornerRadius: 20))
                .overlay { RoundedRectangle(cornerRadius: 20).stroke(model == selected ? MLTheme.cyan.opacity(0.35) : .white.opacity(0.08), lineWidth: 1) }
        }.buttonStyle(.plain).accessibilityLabel("Open \(model.title) lab")
            .accessibilityIdentifier("model-\(model.rawValue)")
    }
}
