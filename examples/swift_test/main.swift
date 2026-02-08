// =============================================================================
// Cactus Swift Integration Test
// =============================================================================
// Tests core functionality of the Cactus Swift bindings on macOS.
// Built against cactus.framework from apple/build-macos/Release/
//
// Usage:
//   ./build.sh && ./swift_test [model_path]
//   Default model path: ~/cactus/weights/lfm2-350m
// =============================================================================

import Foundation

// MARK: - Test Infrastructure

enum TestStatus {
    case passed(String)
    case failed(String)
}

struct TestReport {
    private(set) var passed = 0
    private(set) var failed = 0
    private var results: [(name: String, status: TestStatus)] = []

    mutating func record(_ name: String, _ status: TestStatus) {
        results.append((name, status))
        switch status {
        case .passed: passed += 1
        case .failed: failed += 1
        }
    }

    func printSummary() {
        let divider = String(repeating: "═", count: 64)
        let thinDiv = String(repeating: "─", count: 64)

        print("\n\(divider)")
        print("  TEST SUMMARY")
        print(divider)

        for (name, status) in results {
            switch status {
            case .passed(let detail):
                print("  ✓ PASS │ \(name)")
                if !detail.isEmpty {
                    print("         │ \(detail)")
                }
            case .failed(let detail):
                print("  ✗ FAIL │ \(name)")
                print("         │ \(detail)")
            }
        }

        print(thinDiv)
        let total = passed + failed
        if failed == 0 {
            print("  ✓ All \(total) tests passed!")
        } else {
            print("  ✗ \(failed) of \(total) tests failed")
        }
        print(divider)
    }
}

// MARK: - Path Resolution

func resolveModelPath(_ explicitPath: String?) -> String {
    if let path = explicitPath {
        return (path as NSString).expandingTildeInPath
    }

    let home = NSHomeDirectory()
    let candidates = [
        "\(home)/cactus/weights/lfm2-350m",
        "./weights/lfm2-350m",
    ]

    for path in candidates {
        if FileManager.default.fileExists(atPath: path) {
            return path
        }
    }
    return candidates[0]
}

// MARK: - Test Cases

/// Test 1: Model initialization and cleanup
func testModelInit(modelPath: String) -> (Cactus?, TestStatus) {
    do {
        let model = try Cactus(modelPath: modelPath)
        return (model, .passed("Loaded from \(modelPath)"))
    } catch {
        return (nil, .failed(error.localizedDescription))
    }
}

/// Test 2: Simple single-prompt completion
func testBasicCompletion(_ model: Cactus) -> TestStatus {
    do {
        model.reset()
        let result = try model.complete("What is 2 + 2?")

        guard !result.text.isEmpty else {
            return .failed("Empty response text")
        }

        let detail = String(
            format: "\"%@\" — %d tok, %.1f tok/s decode, %.0fms TTFT, %.0fms total",
            result.text, result.completionTokens,
            result.decodeTokensPerSecond, result.timeToFirstToken, result.totalTime
        )
        return .passed(detail)
    } catch {
        return .failed(error.localizedDescription)
    }
}

/// Test 3: Multi-turn chat with system prompt and name recall
func testChatMessages(_ model: Cactus) -> TestStatus {
    do {
        model.reset()
        let result = try model.complete(messages: [
            .system("You are a helpful assistant. Keep responses brief."),
            .user("My name is Jay. What is my name?"),
        ])

        guard !result.text.isEmpty else {
            return .failed("Empty response text")
        }

        let recalled = result.text.lowercased().contains("jay")
        let detail = "\"\(result.text)\" — name recalled: \(recalled ? "YES" : "NO")"
        return recalled ? .passed(detail) : .failed(detail)
    } catch {
        return .failed(error.localizedDescription)
    }
}

/// Test 4: Custom generation options (temperature, max tokens, stop sequences)
func testCompletionOptions(_ model: Cactus) -> TestStatus {
    do {
        model.reset()
        let options = Cactus.CompletionOptions(
            temperature: 0.1,
            topP: 0.9,
            topK: 40,
            maxTokens: 32,
            stopSequences: [".", "\n"]
        )
        let result = try model.complete("The capital of France is", options: options)

        guard !result.text.isEmpty else {
            return .failed("Empty response text")
        }

        let detail = "\"\(result.text)\" — \(result.completionTokens) tokens (max 32)"
        return .passed(detail)
    } catch {
        return .failed(error.localizedDescription)
    }
}

/// Test 5: Token streaming callback
func testStreaming(_ model: Cactus) -> TestStatus {
    do {
        model.reset()
        var streamedTokenCount = 0
        var streamedText = ""

        _ = try model.complete(
            messages: [.user("Count from 1 to 5.")],
            onToken: { token, _ in
                streamedTokenCount += 1
                streamedText += token
            }
        )

        guard streamedTokenCount > 0 else {
            return .failed("Zero tokens received via callback")
        }

        let preview = String(streamedText.prefix(80))
        let detail = "\(streamedTokenCount) tokens streamed — \"\(preview)\""
        return .passed(detail)
    } catch {
        return .failed(error.localizedDescription)
    }
}

/// Test 6: Confidence score and cloud handoff flag
func testConfidence(_ model: Cactus) -> TestStatus {
    do {
        model.reset()
        let result = try model.complete("Hello, how are you?")

        let detail = String(
            format: "confidence=%.4f, cloud_handoff=%@",
            result.confidence,
            result.needsCloudHandoff ? "true" : "false"
        )
        return .passed(detail)
    } catch {
        return .failed(error.localizedDescription)
    }
}

/// Test 7: All performance metrics populated
func testPerformanceMetrics(_ model: Cactus) -> TestStatus {
    do {
        model.reset()
        let options = Cactus.CompletionOptions(maxTokens: 50)
        let result = try model.complete(
            messages: [.user("Write a short paragraph about artificial intelligence.")],
            options: options
        )

        guard result.prefillTokensPerSecond > 0, result.decodeTokensPerSecond > 0 else {
            return .failed(String(
                format: "Missing metrics — prefill=%.1f tok/s, decode=%.1f tok/s",
                result.prefillTokensPerSecond, result.decodeTokensPerSecond
            ))
        }

        let detail = String(
            format: "prefill: %.1f tok/s (%d tok) | decode: %.1f tok/s (%d tok) | TTFT: %.0fms | total: %.0fms",
            result.prefillTokensPerSecond, result.promptTokens,
            result.decodeTokensPerSecond, result.completionTokens,
            result.timeToFirstToken, result.totalTime
        )
        return .passed(detail)
    } catch {
        return .failed(error.localizedDescription)
    }
}

/// Test 8: KV cache reset between independent completions
func testResetBetweenCompletions(_ model: Cactus) -> TestStatus {
    do {
        model.reset()
        let r1 = try model.complete("Say hello.")
        model.reset()
        let r2 = try model.complete("Say goodbye.")

        guard !r1.text.isEmpty, !r2.text.isEmpty else {
            return .failed("Empty response after reset")
        }

        let detail = "r1=\"\(r1.text.prefix(40))\" → reset → r2=\"\(r2.text.prefix(40))\""
        return .passed(detail)
    } catch {
        return .failed(error.localizedDescription)
    }
}

/// Test 9: Tokenizer roundtrip
func testTokenization(_ model: Cactus) -> TestStatus {
    do {
        let tokens = try model.tokenize("Hello, world!")

        guard !tokens.isEmpty else {
            return .failed("Empty token array")
        }

        let ids = tokens.prefix(10).map { String($0) }.joined(separator: ", ")
        let detail = "\(tokens.count) tokens: [\(ids)]"
        return .passed(detail)
    } catch {
        return .failed(error.localizedDescription)
    }
}

// MARK: - Main

let modelPath = resolveModelPath(
    CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : nil
)
var report = TestReport()

let divider = String(repeating: "═", count: 64)
print(divider)
print("  🌵 CACTUS SWIFT INTEGRATION TEST")
print(divider)
print("  Model:    \(modelPath)")
print("  Platform: macOS \(ProcessInfo.processInfo.operatingSystemVersionString)")
print("  Date:     \(ISO8601DateFormatter().string(from: Date()))")
print(String(repeating: "─", count: 64))

// --- Test execution ---

print("\n▶ [1/9] Model Initialization...")
let (maybeModel, initResult) = testModelInit(modelPath: modelPath)
report.record("Model Init", initResult)

guard let model = maybeModel else {
    print("\n  ✗ Cannot continue without a loaded model.")
    report.printSummary()
    exit(1)
}

print("▶ [2/9] Basic Completion...")
report.record("Basic Completion", testBasicCompletion(model))

print("▶ [3/9] Chat Messages...")
report.record("Chat Messages", testChatMessages(model))

print("▶ [4/9] Completion Options...")
report.record("Completion Options", testCompletionOptions(model))

print("▶ [5/9] Streaming Tokens...")
report.record("Streaming Tokens", testStreaming(model))

print("▶ [6/9] Confidence & Handoff...")
report.record("Confidence & Handoff", testConfidence(model))

print("▶ [7/9] Performance Metrics...")
report.record("Performance Metrics", testPerformanceMetrics(model))

print("▶ [8/9] Reset Between Completions...")
report.record("Reset Between Completions", testResetBetweenCompletions(model))

print("▶ [9/9] Tokenization...")
report.record("Tokenization", testTokenization(model))

report.printSummary()
exit(Int32(report.failed))
