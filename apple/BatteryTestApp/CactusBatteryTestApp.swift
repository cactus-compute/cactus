import SwiftUI
import UIKit
import Combine
import Foundation

@main
struct CactusBatteryTestApp: App {
    init() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        UIApplication.shared.isIdleTimerDisabled = true
    }
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

enum Backend: String { case cpu = "CPU", gpu = "GPU", npu = "NPU", idle = "idle" }

@MainActor
final class Runner: ObservableObject {
    @Published var backend: Backend = .idle
    @Published var iterations: Int = 0
    @Published var lastPrefillTps: Double = 0
    @Published var batteryLevel: Float = UIDevice.current.batteryLevel
    @Published var batteryStateText: String = ""
    @Published var elapsedSeconds: Int = 0
    @Published var status: String = "ready"

    private var task: Task<Void, Never>? = nil
    private var startBattery: Float = 0
    private var startTime: Date = .now
    private var model: OpaquePointer? = nil

    private let modelPath: String = {
        let fm = FileManager.default
        if let resPath = Bundle.main.resourcePath {
            let candidate = (resPath as NSString).appendingPathComponent("model")
            if fm.fileExists(atPath: (candidate as NSString).appendingPathComponent("config.txt")) {
                return candidate
            }
        }
        if let p = Bundle.main.path(forResource: "model", ofType: nil) { return p }
        if let p = ProcessInfo.processInfo.environment["CACTUS_MODEL"] { return p }
        return (NSHomeDirectory() as NSString).appendingPathComponent("Documents/model")
    }()

    private let prompt: String = """
    [
      {"role":"system","content":"You are a helpful assistant."},
      {"role":"user","content":"Summarize the following: The history of computing is a long and complex story. The earliest mechanical aids to calculation were the abacus and slide rule. In the 19th century, Charles Babbage designed the Analytical Engine, often considered the first general-purpose mechanical computer, though it was never fully built in his lifetime. Ada Lovelace wrote the first algorithm intended for processing on a machine, making her widely regarded as the first computer programmer. The 20th century brought rapid advances: vacuum tubes, transistors, integrated circuits, and microprocessors. Personal computers in the 1970s and 1980s, followed by networked computing and the internet in the 1990s, transformed society. Today computing is ubiquitous."}
    ]
    """

    private let options: String = "{\"max_tokens\":1,\"telemetry_enabled\":false}"

    func loadModelIfNeeded() -> Bool {
        if model != nil { return true }
        status = "loading model from \(modelPath)..."
        let m = modelPath.withCString { mp in
            cactus_init(mp, nil, false)
        }
        guard let m = m else {
            status = "failed to load model — set CACTUS_MODEL or include 'model' bundle dir"
            return false
        }
        model = OpaquePointer(m)
        status = "model loaded"
        return true
    }

    func start(_ b: Backend) {
        if backend != .idle { stop(); return }
        guard loadModelIfNeeded() else { return }

        switch b {
        case .cpu:
            cactus_npu_set_enabled(false)
            cactus_mps_set_enabled(false)
        case .gpu:
            cactus_npu_set_enabled(false)
            cactus_mps_set_enabled(true)
        case .npu:
            cactus_mps_set_enabled(false)
            cactus_npu_set_enabled(true)
        case .idle: return
        }

        backend = b
        iterations = 0
        lastPrefillTps = 0
        startBattery = UIDevice.current.batteryLevel
        startTime = .now
        status = "running \(b.rawValue)..."

        let modelPtr = self.model
        let promptC = strdup(self.prompt)!
        let optsC = strdup(self.options)!

        task = Task.detached(priority: .userInitiated) {
            var buffer = [CChar](repeating: 0, count: 4096)
            while !Task.isCancelled {
                cactus_reset(modelPtr.map { UnsafeMutableRawPointer($0) })
                let rc = cactus_prefill(modelPtr.map { UnsafeMutableRawPointer($0) },
                                        promptC,
                                        &buffer,
                                        buffer.count,
                                        optsC,
                                        nil, nil, 0)
                if rc <= 0 {
                    await MainActor.run { self.status = "prefill rc=\(rc) (stopping)" }
                    break
                }
                let resp = String(cString: buffer)
                let tps = parsePrefillTps(resp)
                await MainActor.run {
                    self.iterations += 1
                    if tps > 0 { self.lastPrefillTps = tps }
                    self.batteryLevel = UIDevice.current.batteryLevel
                    self.elapsedSeconds = Int(Date().timeIntervalSince(self.startTime))
                    self.batteryStateText = String(format: "%.1f%% drop over %ds",
                        (self.startBattery - self.batteryLevel) * 100, self.elapsedSeconds)
                }
            }
            free(promptC)
            free(optsC)
        }
    }

    func stop() {
        task?.cancel()
        task = nil
        backend = .idle
        status = "stopped"
    }
}

private func parsePrefillTps(_ s: String) -> Double {
    guard let r = s.range(of: "\"prefill_tps\"") else { return 0 }
    let after = s[r.upperBound...]
    let digits = after.drop(while: { !($0.isNumber || $0 == "-") })
    let val = digits.prefix(while: { $0.isNumber || $0 == "." || $0 == "-" })
    return Double(val) ?? 0
}

struct ContentView: View {
    @StateObject var runner = Runner()
    var body: some View {
        VStack(spacing: 18) {
            Text("Cactus prefill battery test").font(.title2).bold().padding(.top, 28)

            VStack(alignment: .leading, spacing: 6) {
                row("backend", runner.backend.rawValue)
                row("iterations", "\(runner.iterations)")
                row("prefill_tps", String(format: "%.1f", runner.lastPrefillTps))
                row("battery", String(format: "%.0f%%", runner.batteryLevel * 100))
                row("Δ", runner.batteryStateText)
                row("elapsed", "\(runner.elapsedSeconds)s")
                row("status", runner.status)
            }
            .font(.system(.body, design: .monospaced))
            .padding()
            .background(Color(.secondarySystemBackground))
            .cornerRadius(10)
            .padding(.horizontal)

            HStack(spacing: 12) {
                Button { runner.start(.cpu) } label: {
                    Text(runner.backend == .cpu ? "STOP CPU" : "CPU")
                        .frame(maxWidth: .infinity).padding().bold()
                }
                .buttonStyle(.borderedProminent)
                .tint(runner.backend == .cpu ? .red : .orange)
                .disabled(runner.backend == .gpu || runner.backend == .npu)

                Button { runner.start(.gpu) } label: {
                    Text(runner.backend == .gpu ? "STOP GPU" : "GPU")
                        .frame(maxWidth: .infinity).padding().bold()
                }
                .buttonStyle(.borderedProminent)
                .tint(runner.backend == .gpu ? .red : .blue)
                .disabled(runner.backend == .cpu || runner.backend == .npu)

                Button { runner.start(.npu) } label: {
                    Text(runner.backend == .npu ? "STOP NPU" : "NPU")
                        .frame(maxWidth: .infinity).padding().bold()
                }
                .buttonStyle(.borderedProminent)
                .tint(runner.backend == .npu ? .red : .green)
                .disabled(runner.backend == .cpu || runner.backend == .gpu)
            }
            .padding(.horizontal)

            Spacer()
        }
    }
    private func row(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k).foregroundStyle(.secondary).frame(width: 100, alignment: .leading)
            Text(v)
            Spacer()
        }
    }
}
