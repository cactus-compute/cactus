import Foundation
import cactus

public final class Cactus: @unchecked Sendable {


    public struct CompletionResult {
        public let text: String
        public let functionCalls: [[String: Any]]?
        public let promptTokens: Int
        public let completionTokens: Int
        public let timeToFirstToken: Double
        public let totalTime: Double
        public let prefillTokensPerSecond: Double
        public let decodeTokensPerSecond: Double
        public let confidence: Double
        public let needsCloudHandoff: Bool

        init(json: [String: Any]) {
            self.text = json["text"] as? String ?? ""
            self.functionCalls = json["function_calls"] as? [[String: Any]]
            self.promptTokens = json["prompt_tokens"] as? Int ?? 0
            self.completionTokens = json["completion_tokens"] as? Int ?? 0
            self.timeToFirstToken = json["time_to_first_token_ms"] as? Double ?? 0
            self.totalTime = json["total_time_ms"] as? Double ?? 0
            self.prefillTokensPerSecond = json["prefill_tokens_per_second"] as? Double ?? 0
            self.decodeTokensPerSecond = json["decode_tokens_per_second"] as? Double ?? 0
            self.confidence = json["confidence"] as? Double ?? 1.0
            self.needsCloudHandoff = json["cloud_handoff"] as? Bool ?? false
        }
    }

    public struct TranscriptionResult {
        public let text: String
        public let segments: [[String: Any]]?
        public let totalTime: Double

        init(json: [String: Any]) {
            self.text = json["text"] as? String ?? ""
            self.segments = json["segments"] as? [[String: Any]]
            self.totalTime = json["total_time_ms"] as? Double ?? 0
        }
    }

    public struct VADSegment {
        public let start: Int
        public let end: Int

        init(dict: [String: Any]) {
            self.start = dict["start"] as? Int ?? 0
            self.end = dict["end"] as? Int ?? 0
        }
    }

    public struct VADResult {
        public let segments: [VADSegment]
        public let totalTime: Double
        public let ramUsage: Double

        init(json: [String: Any]) {
            let segmentsArray = json["segments"] as? [[String: Any]] ?? []
            self.segments = segmentsArray.map { VADSegment(dict: $0) }
            self.totalTime = json["total_time_ms"] as? Double ?? 0
            self.ramUsage = json["ram_usage_mb"] as? Double ?? 0
        }
    }

    public struct Message {
        public let role: String
        public let content: String

        public init(role: String, content: String) {
            self.role = role
            self.content = content
        }

        public static func system(_ content: String) -> Message {
            Message(role: "system", content: content)
        }

        public static func user(_ content: String) -> Message {
            Message(role: "user", content: content)
        }

        public static func assistant(_ content: String) -> Message {
            Message(role: "assistant", content: content)
        }

        func toDict() -> [String: String] {
            ["role": role, "content": content]
        }
    }

    public struct CompletionOptions {
        public var temperature: Float
        public var topP: Float
        public var topK: Int
        public var maxTokens: Int
        public var stopSequences: [String]
        public var confidenceThreshold: Float

        public init(
            temperature: Float = 0.7,
            topP: Float = 0.9,
            topK: Int = 40,
            maxTokens: Int = 512,
            stopSequences: [String] = [],
            confidenceThreshold: Float = 0.0
        ) {
            self.temperature = temperature
            self.topP = topP
            self.topK = topK
            self.maxTokens = maxTokens
            self.stopSequences = stopSequences
            self.confidenceThreshold = confidenceThreshold
        }

        public static let `default` = CompletionOptions()

        func toJSON() -> String? {
            let dict: [String: Any] = [
                "temperature": temperature,
                "top_p": topP,
                "top_k": topK,
                "max_tokens": maxTokens,
                "stop": stopSequences,
                "confidence_threshold": confidenceThreshold
            ]
            guard let data = try? JSONSerialization.data(withJSONObject: dict),
                  let json = String(data: data, encoding: .utf8) else {
                return nil
            }
            return json
        }
    }

    public struct TranscriptionOptions {
        public var language: String?
        public var translateToEnglish: Bool

        public init(language: String? = nil, translateToEnglish: Bool = false) {
            self.language = language
            self.translateToEnglish = translateToEnglish
        }

        public static let `default` = TranscriptionOptions()

        func toJSON() -> String? {
            var dict: [String: Any] = [
                "translate": translateToEnglish
            ]
            if let lang = language {
                dict["language"] = lang
            }
            guard let data = try? JSONSerialization.data(withJSONObject: dict),
                  let json = String(data: data, encoding: .utf8) else {
                return nil
            }
            return json
        }
    }

    public struct VADOptions {
        public var threshold: Float?
        public var negThreshold: Float?
        public var minSpeechDurationMs: Int?
        public var maxSpeechDurationS: Float?
        public var minSilenceDurationMs: Int?
        public var speechPadMs: Int?
        public var windowSizeSamples: Int?
        public var samplingRate: Int?

        public init(
            threshold: Float? = nil,
            negThreshold: Float? = nil,
            minSpeechDurationMs: Int? = nil,
            maxSpeechDurationS: Float? = nil,
            minSilenceDurationMs: Int? = nil,
            speechPadMs: Int? = nil,
            windowSizeSamples: Int? = nil,
            samplingRate: Int? = nil
        ) {
            self.threshold = threshold
            self.negThreshold = negThreshold
            self.minSpeechDurationMs = minSpeechDurationMs
            self.maxSpeechDurationS = maxSpeechDurationS
            self.minSilenceDurationMs = minSilenceDurationMs
            self.speechPadMs = speechPadMs
            self.windowSizeSamples = windowSizeSamples
            self.samplingRate = samplingRate
        }

        public static let `default` = VADOptions()

        func toJSON() -> String? {
            var dict: [String: Any] = [:]
            if let threshold = threshold { dict["threshold"] = threshold }
            if let negThreshold = negThreshold { dict["neg_threshold"] = negThreshold }
            if let minSpeechDurationMs = minSpeechDurationMs { dict["min_speech_duration_ms"] = minSpeechDurationMs }
            if let maxSpeechDurationS = maxSpeechDurationS { dict["max_speech_duration_s"] = maxSpeechDurationS }
            if let minSilenceDurationMs = minSilenceDurationMs { dict["min_silence_duration_ms"] = minSilenceDurationMs }
            if let speechPadMs = speechPadMs { dict["speech_pad_ms"] = speechPadMs }
            if let windowSizeSamples = windowSizeSamples { dict["window_size_samples"] = windowSizeSamples }
            if let samplingRate = samplingRate { dict["sampling_rate"] = samplingRate }

            guard !dict.isEmpty else { return nil }
            guard let data = try? JSONSerialization.data(withJSONObject: dict),
                  let json = String(data: data, encoding: .utf8) else {
                return nil
            }
            return json
        }
    }

    public enum CactusError: Error, LocalizedError {
        case initializationFailed(String)
        case completionFailed(String)
        case transcriptionFailed(String)
        case vadFailed(String)
        case embeddingFailed(String)
        case invalidResponse

        public var errorDescription: String? {
            switch self {
            case .initializationFailed(let msg): return "Initialization failed: \(msg)"
            case .completionFailed(let msg): return "Completion failed: \(msg)"
            case .transcriptionFailed(let msg): return "Transcription failed: \(msg)"
            case .vadFailed(let msg): return "VAD failed: \(msg)"
            case .embeddingFailed(let msg): return "Embedding failed: \(msg)"
            case .invalidResponse: return "Invalid response from model"
            }
        }
    }


    private let handle: OpaquePointer
    private static let defaultBufferSize = 65536

    public init(modelPath: String, corpusDir: String? = nil, cacheIndex: Bool = false) throws {
        guard let h = cactus_init(modelPath, corpusDir, cacheIndex) else {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.initializationFailed(error.isEmpty ? "Unknown error" : error)
        }
        self.handle = h
    }

    deinit {
        cactus_destroy(handle)
    }

    public func complete(
        messages: [Message],
        options: CompletionOptions = .default,
        tools: [[String: Any]]? = nil,
        onToken: ((String, UInt32) -> Void)? = nil
    ) throws -> CompletionResult {
        let messagesJSON = try serializeMessages(messages)
        let optionsJSON = options.toJSON()
        let toolsJSON = try serializeTools(tools)

        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)

        let callbackContext = onToken.map { TokenCallbackContext(callback: $0) }
        let contextPtr = callbackContext.map { Unmanaged.passUnretained($0).toOpaque() }

        let result = buffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_complete(
                handle,
                messagesJSON,
                bufferPtr.baseAddress,
                bufferPtr.count,
                optionsJSON,
                toolsJSON,
                onToken != nil ? tokenCallbackBridge : nil,
                contextPtr
            )
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.completionFailed(error.isEmpty ? "Unknown error" : error)
        }

        let responseString = String(cString: buffer)
        guard let json = parseJSON(responseString) else {
            throw CactusError.invalidResponse
        }

        if let errorMsg = json["error"] as? String {
            throw CactusError.completionFailed(errorMsg)
        }

        return CompletionResult(json: json)
    }

    public func complete(
        _ prompt: String,
        options: CompletionOptions = .default
    ) throws -> CompletionResult {
        try complete(messages: [.user(prompt)], options: options)
    }

    public func transcribe(
        audioPath: String,
        prompt: String? = nil,
        options: TranscriptionOptions = .default
    ) throws -> TranscriptionResult {
        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)
        let optionsJSON = options.toJSON()

        let result = buffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_transcribe(
                handle,
                audioPath,
                prompt,
                bufferPtr.baseAddress,
                bufferPtr.count,
                optionsJSON,
                nil,
                nil,
                nil,
                0
            )
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.transcriptionFailed(error.isEmpty ? "Unknown error" : error)
        }

        let responseString = String(cString: buffer)
        guard let json = parseJSON(responseString) else {
            throw CactusError.invalidResponse
        }

        return TranscriptionResult(json: json)
    }

    public func transcribe(
        pcmData: Data,
        prompt: String? = nil,
        options: TranscriptionOptions = .default
    ) throws -> TranscriptionResult {
        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)
        let optionsJSON = options.toJSON()

        let result = pcmData.withUnsafeBytes { pcmPtr in
            buffer.withUnsafeMutableBufferPointer { bufferPtr in
                cactus_transcribe(
                    handle,
                    nil,
                    prompt,
                    bufferPtr.baseAddress,
                    bufferPtr.count,
                    optionsJSON,
                    nil,
                    nil,
                    pcmPtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                    pcmData.count
                )
            }
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.transcriptionFailed(error.isEmpty ? "Unknown error" : error)
        }

        let responseString = String(cString: buffer)
        guard let json = parseJSON(responseString) else {
            throw CactusError.invalidResponse
        }

        return TranscriptionResult(json: json)
    }

    public func embed(text: String, normalize: Bool = true) throws -> [Float] {
        var embeddingBuffer = [Float](repeating: 0, count: 4096)
        var embeddingDim: Int = 0

        let result = embeddingBuffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_embed(
                handle,
                text,
                bufferPtr.baseAddress,
                bufferPtr.count,
                &embeddingDim,
                normalize
            )
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.embeddingFailed(error.isEmpty ? "Unknown error" : error)
        }

        return Array(embeddingBuffer.prefix(embeddingDim))
    }

    public func ragQuery(_ query: String, topK: Int = 5) throws -> String {
        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)

        let result = buffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_rag_query(
                handle,
                query,
                bufferPtr.baseAddress,
                bufferPtr.count,
                topK
            )
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.completionFailed(error.isEmpty ? "Unknown error" : error)
        }

        return String(cString: buffer)
    }

    public func tokenize(_ text: String) throws -> [UInt32] {
        var tokenBuffer = [UInt32](repeating: 0, count: 8192)
        var tokenLen: Int = 0

        let result = tokenBuffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_tokenize(
                handle,
                text,
                bufferPtr.baseAddress,
                bufferPtr.count,
                &tokenLen
            )
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.completionFailed(error.isEmpty ? "Unknown error" : error)
        }

        return Array(tokenBuffer.prefix(tokenLen))
    }

    public func scoreWindow(tokens: [UInt32], start: Int, end: Int, context: Int) throws -> String {
        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)

        let result = tokens.withUnsafeBufferPointer { tokenPtr in
            buffer.withUnsafeMutableBufferPointer { bufferPtr in
                cactus_score_window(
                    handle,
                    tokenPtr.baseAddress,
                    tokenPtr.count,
                    start,
                    end,
                    context,
                    bufferPtr.baseAddress,
                    bufferPtr.count
                )
            }
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.completionFailed(error.isEmpty ? "Unknown error" : error)
        }

        return String(cString: buffer)
    }

    public func imageEmbed(_ imagePath: String) throws -> [Float] {
        var embeddingBuffer = [Float](repeating: 0, count: 4096)
        var embeddingDim: Int = 0

        let result = embeddingBuffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_image_embed(
                handle,
                imagePath,
                bufferPtr.baseAddress,
                bufferPtr.count,
                &embeddingDim
            )
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.embeddingFailed(error.isEmpty ? "Unknown error" : error)
        }

        return Array(embeddingBuffer.prefix(embeddingDim))
    }

    public func audioEmbed(_ audioPath: String) throws -> [Float] {
        var embeddingBuffer = [Float](repeating: 0, count: 4096)
        var embeddingDim: Int = 0

        let result = embeddingBuffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_audio_embed(
                handle,
                audioPath,
                bufferPtr.baseAddress,
                bufferPtr.count,
                &embeddingDim
            )
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.embeddingFailed(error.isEmpty ? "Unknown error" : error)
        }

        return Array(embeddingBuffer.prefix(embeddingDim))
    }

    public func vad(
        audioPath: String,
        options: VADOptions = .default
    ) throws -> VADResult {
        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)
        let optionsJSON = options.toJSON()

        let result = buffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_vad(
                handle,
                audioPath,
                bufferPtr.baseAddress,
                bufferPtr.count,
                optionsJSON,
                nil,
                0
            )
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.vadFailed(error.isEmpty ? "Unknown error" : error)
        }

        let responseString = String(cString: buffer)
        guard let json = parseJSON(responseString) else {
            throw CactusError.invalidResponse
        }

        if let errorMsg = json["error"] as? String {
            throw CactusError.vadFailed(errorMsg)
        }

        return VADResult(json: json)
    }

    public func vad(
        pcmData: Data,
        options: VADOptions = .default
    ) throws -> VADResult {
        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)
        let optionsJSON = options.toJSON()

        let result = pcmData.withUnsafeBytes { pcmPtr in
            buffer.withUnsafeMutableBufferPointer { bufferPtr in
                cactus_vad(
                    handle,
                    nil,
                    bufferPtr.baseAddress,
                    bufferPtr.count,
                    optionsJSON,
                    pcmPtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                    pcmData.count
                )
            }
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.vadFailed(error.isEmpty ? "Unknown error" : error)
        }

        let responseString = String(cString: buffer)
        guard let json = parseJSON(responseString) else {
            throw CactusError.invalidResponse
        }

        if let errorMsg = json["error"] as? String {
            throw CactusError.vadFailed(errorMsg)
        }

        return VADResult(json: json)
    }

    public func createStreamTranscriber(options: TranscriptionOptions = .default) throws -> StreamTranscriber {
        guard let streamHandle = cactus_stream_transcribe_start(handle, options.toJSON()) else {
            let error = String(cString: cactus_get_last_error())
            throw CactusError.transcriptionFailed(error.isEmpty ? "Unknown error" : error)
        }
        return StreamTranscriber(handle: streamHandle)
    }

    public func reset() {
        cactus_reset(handle)
    }

    public func stop() {
        cactus_stop(handle)
    }


    private func serializeMessages(_ messages: [Message]) throws -> String {
        let dicts = messages.map { $0.toDict() }
        guard let data = try? JSONSerialization.data(withJSONObject: dicts),
              let json = String(data: data, encoding: .utf8) else {
            throw CactusError.completionFailed("Failed to serialize messages")
        }
        return json
    }

    private func serializeTools(_ tools: [[String: Any]]?) throws -> String? {
        guard let tools = tools else { return nil }
        guard let data = try? JSONSerialization.data(withJSONObject: tools),
              let json = String(data: data, encoding: .utf8) else {
            throw CactusError.completionFailed("Failed to serialize tools")
        }
        return json
    }

    private func parseJSON(_ string: String) -> [String: Any]? {
        guard let data = string.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return json
    }
}


private class TokenCallbackContext {
    let callback: (String, UInt32) -> Void
    init(callback: @escaping (String, UInt32) -> Void) {
        self.callback = callback
    }
}

private func tokenCallbackBridge(token: UnsafePointer<CChar>?, tokenId: UInt32, userData: UnsafeMutableRawPointer?) {
    guard let token = token, let userData = userData else { return }
    let context = Unmanaged<TokenCallbackContext>.fromOpaque(userData).takeUnretainedValue()
    let tokenString = String(cString: token)
    context.callback(tokenString, tokenId)
}


#if os(iOS) || os(macOS) || os(tvOS) || os(watchOS) || os(visionOS)
@available(iOS 13.0, macOS 10.15, tvOS 13.0, watchOS 6.0, *)
#endif
public extension Cactus {

    func complete(
        messages: [Message],
        options: CompletionOptions = .default,
        tools: [[String: Any]]? = nil
    ) async throws -> CompletionResult {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try self.complete(messages: messages, options: options, tools: tools)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func completeStream(
        messages: [Message],
        options: CompletionOptions = .default
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    _ = try self.complete(messages: messages, options: options) { token, _ in
                        continuation.yield(token)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    func transcribe(
        audioPath: String,
        prompt: String? = nil,
        options: TranscriptionOptions = .default
    ) async throws -> TranscriptionResult {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try self.transcribe(audioPath: audioPath, prompt: prompt, options: options)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func embed(text: String, normalize: Bool = true) async throws -> [Float] {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try self.embed(text: text, normalize: normalize)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func vad(
        audioPath: String,
        options: VADOptions = .default
    ) async throws -> VADResult {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try self.vad(audioPath: audioPath, options: options)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func vad(
        pcmData: Data,
        options: VADOptions = .default
    ) async throws -> VADResult {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try self.vad(pcmData: pcmData, options: options)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}


public final class StreamTranscriber: @unchecked Sendable {

    private var handle: OpaquePointer?
    private static let defaultBufferSize = 65536

    init(handle: OpaquePointer) {
        self.handle = handle
    }

    deinit {
        close()
    }

    public func process(pcmData: Data) throws -> Cactus.TranscriptionResult {
        guard let handle = handle else {
            throw Cactus.CactusError.transcriptionFailed("Stream transcriber has been closed")
        }

        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)

        let result = pcmData.withUnsafeBytes { pcmPtr in
            buffer.withUnsafeMutableBufferPointer { bufferPtr in
                cactus_stream_transcribe_process(
                    handle,
                    pcmPtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                    pcmData.count,
                    bufferPtr.baseAddress,
                    bufferPtr.count
                )
            }
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw Cactus.CactusError.transcriptionFailed(error.isEmpty ? "Unknown error" : error)
        }

        let responseString = String(cString: buffer)
        guard let data = responseString.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Cactus.CactusError.invalidResponse
        }

        return Cactus.TranscriptionResult(json: json)
    }

    public func stop() throws -> Cactus.TranscriptionResult {
        guard let handle = handle else {
            throw Cactus.CactusError.transcriptionFailed("Stream transcriber has been closed")
        }

        var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)

        let result = buffer.withUnsafeMutableBufferPointer { bufferPtr in
            cactus_stream_transcribe_stop(
                handle,
                bufferPtr.baseAddress,
                bufferPtr.count
            )
        }

        self.handle = nil  // Stream is now closed

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw Cactus.CactusError.transcriptionFailed(error.isEmpty ? "Unknown error" : error)
        }

        let responseString = String(cString: buffer)
        guard let data = responseString.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Cactus.CactusError.invalidResponse
        }

        return Cactus.TranscriptionResult(json: json)
    }

    public func close() {
        if let handle = handle {
            var buffer = [CChar](repeating: 0, count: Self.defaultBufferSize)
            _ = buffer.withUnsafeMutableBufferPointer { bufferPtr in
                cactus_stream_transcribe_stop(handle, bufferPtr.baseAddress, bufferPtr.count)
            }
            self.handle = nil
        }
    }
}


public final class CactusIndex: @unchecked Sendable {

    public struct IndexResult {
        public let id: Int
        public let score: Float
    }

    public enum IndexError: Error, LocalizedError {
        case initializationFailed(String)
        case operationFailed(String)

        public var errorDescription: String? {
            switch self {
            case .initializationFailed(let msg): return "Index initialization failed: \(msg)"
            case .operationFailed(let msg): return "Index operation failed: \(msg)"
            }
        }
    }

    private var handle: OpaquePointer?

    public init(indexDir: String, embeddingDim: Int) throws {
        guard let h = cactus_index_init(indexDir, embeddingDim) else {
            throw IndexError.initializationFailed("Failed to initialize index")
        }
        self.handle = h
    }

    deinit {
        close()
    }

    public func add(
        ids: [Int],
        documents: [String],
        embeddings: [[Float]],
        metadatas: [String]? = nil
    ) throws {
        guard let handle = handle else {
            throw IndexError.operationFailed("Index has been closed")
        }

        let count = ids.count
        let embeddingDim = embeddings[0].count

        var idArray = ids.map { Int32($0) }
        var docPtrs = documents.map { strdup($0) }
        var metaPtrs: [UnsafeMutablePointer<CChar>?]? = metadatas?.map { strdup($0) }
        var embPtrs = embeddings.map { emb -> UnsafePointer<Float>? in
            let ptr = UnsafeMutablePointer<Float>.allocate(capacity: emb.count)
            ptr.initialize(from: emb, count: emb.count)
            return UnsafePointer(ptr)
        }

        let result = idArray.withUnsafeMutableBufferPointer { idPtr in
            docPtrs.withUnsafeMutableBufferPointer { docPtr in
                embPtrs.withUnsafeMutableBufferPointer { embPtr in
                    if let metaPtrs = metaPtrs {
                        var metaPtrsCopy = metaPtrs
                        return metaPtrsCopy.withUnsafeMutableBufferPointer { metaPtr in
                            cactus_index_add(
                                handle,
                                idPtr.baseAddress,
                                docPtr.baseAddress,
                                metaPtr.baseAddress,
                                embPtr.baseAddress,
                                count,
                                embeddingDim
                            )
                        }
                    } else {
                        return cactus_index_add(
                            handle,
                            idPtr.baseAddress,
                            docPtr.baseAddress,
                            nil,
                            embPtr.baseAddress,
                            count,
                            embeddingDim
                        )
                    }
                }
            }
        }

        docPtrs.forEach { free($0) }
        metaPtrs?.forEach { free($0) }
        embPtrs.forEach { ptr in
            if let ptr = ptr {
                UnsafeMutablePointer(mutating: ptr).deallocate()
            }
        }

        if result < 0 {
            throw IndexError.operationFailed("Failed to add documents to index")
        }
    }

    public func delete(ids: [Int]) throws {
        guard let handle = handle else {
            throw IndexError.operationFailed("Index has been closed")
        }

        var idArray = ids.map { Int32($0) }

        let result = idArray.withUnsafeMutableBufferPointer { idPtr in
            cactus_index_delete(handle, idPtr.baseAddress, ids.count)
        }

        if result < 0 {
            throw IndexError.operationFailed("Failed to delete documents from index")
        }
    }

    public func query(embedding: [Float], topK: Int = 5) throws -> [IndexResult] {
        guard let handle = handle else {
            throw IndexError.operationFailed("Index has been closed")
        }

        var embeddingCopy = embedding
        var idBuffer = [Int32](repeating: 0, count: topK)
        var scoreBuffer = [Float](repeating: 0, count: topK)
        var idBufferSize = topK
        var scoreBufferSize = topK

        let result = embeddingCopy.withUnsafeMutableBufferPointer { embPtr in
            idBuffer.withUnsafeMutableBufferPointer { idPtr in
                scoreBuffer.withUnsafeMutableBufferPointer { scorePtr in
                    var embPtrPtr: UnsafePointer<Float>? = embPtr.baseAddress
                    var idPtrPtr: UnsafeMutablePointer<Int32>? = idPtr.baseAddress
                    var scorePtrPtr: UnsafeMutablePointer<Float>? = scorePtr.baseAddress

                    return withUnsafeMutablePointer(to: &embPtrPtr) { embPtrPtrPtr in
                        withUnsafeMutablePointer(to: &idPtrPtr) { idPtrPtrPtr in
                            withUnsafeMutablePointer(to: &scorePtrPtr) { scorePtrPtrPtr in
                                withUnsafeMutablePointer(to: &idBufferSize) { idSizePtr in
                                    withUnsafeMutablePointer(to: &scoreBufferSize) { scoreSizePtr in
                                        cactus_index_query(
                                            handle,
                                            embPtrPtrPtr,
                                            1,
                                            embedding.count,
                                            nil,
                                            idPtrPtrPtr,
                                            idSizePtr,
                                            scorePtrPtrPtr,
                                            scoreSizePtr
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if result < 0 {
            let error = String(cString: cactus_get_last_error())
            throw IndexError.operationFailed(error.isEmpty ? "Unknown error" : error)
        }

        return (0..<idBufferSize).map { i in
            IndexResult(id: Int(idBuffer[i]), score: scoreBuffer[i])
        }
    }

    public func compact() throws {
        guard let handle = handle else {
            throw IndexError.operationFailed("Index has been closed")
        }

        let result = cactus_index_compact(handle)
        if result < 0 {
            throw IndexError.operationFailed("Failed to compact index")
        }
    }

    public func close() {
        if let handle = handle {
            cactus_index_destroy(handle)
            self.handle = nil
        }
    }
}
