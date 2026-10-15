// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ClassicalMLSimulator",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MLCore", targets: ["MLCore"]),
        .library(name: "MLWorkbench", targets: ["MLWorkbench"])
    ],
    targets: [
        .target(name: "MLCore"),
        .target(name: "MLWorkbench", dependencies: ["MLCore"], path: "App",
                exclude: ["Assets.xcassets", "ClassicalMLSimulatorApp.swift", "DesignSystem", "Info.plist", "Views"],
                sources: ["LabSession.swift", "ModelCatalog.swift"]),
        .testTarget(name: "MLCoreTests", dependencies: ["MLCore"]),
        .testTarget(name: "MLWorkbenchTests", dependencies: ["MLWorkbench", "MLCore"])
    ]
)
