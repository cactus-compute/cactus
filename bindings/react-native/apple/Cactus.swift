import Foundation
import React
import cactus

@objc(Cactus)
final class Cactus: NSObject {
    private let defaultBufferSize = 65_536
    private let largeBufferSize = 1 << 20
    private let defaultEmbeddingBufferSize = 4096
    private let defaultTokenBufferSize = 8192
    private let defaultIndexResultCapacity = 1000
    private let defaultIndexDocBufferSize = 4096
    private let defaultIndexEmbeddingBufferSize = 4096

    @objc
    static func requiresMainQueueSetup() -> Bool {
        false
    }

    private func lastError(_ fallback: String) -> String {
        guard let ptr = cactus_get_last_error() else { return fallback }
        return String(cString: ptr)
    }

    private func reject(_ rejecter: RCTPromiseRejectBlock, _ fallback: String) {
        rejecter("CACTUS_ERROR", lastError(fallback), nil)
    }

    private func decodeBase64(_ value: String?) -> Data? {
        guard let value else { return nil }
        return Data(base64Encoded: value)
    }

    private func stringFromBuffer(_ buffer: [CChar]) -> String {
        String(cString: buffer)
    }

    private func encodeHandle(_ handle: UnsafeMutableRawPointer) -> String {
        String(UInt(bitPattern: handle))
    }

    private func decodeHandle(_ handle: String) -> UnsafeMutableRawPointer? {
        guard let bits = UInt(handle) else { return nil }
        return UnsafeMutableRawPointer(bitPattern: bits)
    }

    @objc(init:corpusDir:cacheIndex:resolver:rejecter:)
    func initialize(_ modelPath: String, corpusDir: String?, cacheIndex: Bool, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        let handle = cactus_init(modelPath, corpusDir, cacheIndex)
        guard let handle else {
            self.reject(reject, "Failed to initialize model")
            return
        }
        resolve(encodeHandle(handle))
    }

    @objc(destroy:resolver:rejecter:)
    func destroy(_ handle: String, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        cactus_destroy(nativeHandle)
        resolve(nil)
    }

    @objc(reset:resolver:rejecter:)
    func reset(_ handle: String, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        cactus_reset(nativeHandle)
        resolve(nil)
    }

    @objc(stop:resolver:rejecter:)
    func stop(_ handle: String, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        cactus_stop(nativeHandle)
        resolve(nil)
    }

    @objc(prefill:messagesJson:optionsJson:toolsJson:pcmDataBase64:resolver:rejecter:)
    func prefill(
        _ handle: String,
        messagesJson: String,
        optionsJson: String?,
        toolsJson: String?,
        pcmDataBase64: String?,
        resolver resolve: RCTPromiseResolveBlock,
        rejecter reject: RCTPromiseRejectBlock
    ) {
        var buffer = [CChar](repeating: 0, count: defaultBufferSize)
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        let rc = decodeBase64(pcmDataBase64)?.withUnsafeBytes { rawBuffer in
            cactus_prefill(nativeHandle, messagesJson, &buffer, buffer.count, optionsJson, toolsJson, rawBuffer.bindMemory(to: UInt8.self).baseAddress, rawBuffer.count)
        } ?? cactus_prefill(nativeHandle, messagesJson, &buffer, buffer.count, optionsJson, toolsJson, nil, 0)
        guard rc >= 0 else {
            self.reject(reject, "Prefill failed")
            return
        }
        resolve(stringFromBuffer(buffer))
    }

    @objc(complete:messagesJson:optionsJson:toolsJson:pcmDataBase64:resolver:rejecter:)
    func complete(
        _ handle: String,
        messagesJson: String,
        optionsJson: String?,
        toolsJson: String?,
        pcmDataBase64: String?,
        resolver resolve: RCTPromiseResolveBlock,
        rejecter reject: RCTPromiseRejectBlock
    ) {
        var buffer = [CChar](repeating: 0, count: defaultBufferSize)
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        let rc = decodeBase64(pcmDataBase64)?.withUnsafeBytes { rawBuffer in
            cactus_complete(nativeHandle, messagesJson, &buffer, buffer.count, optionsJson, toolsJson, nil, nil, rawBuffer.bindMemory(to: UInt8.self).baseAddress, rawBuffer.count)
        } ?? cactus_complete(nativeHandle, messagesJson, &buffer, buffer.count, optionsJson, toolsJson, nil, nil, nil, 0)
        guard rc >= 0 else {
            self.reject(reject, "Completion failed")
            return
        }
        resolve(stringFromBuffer(buffer))
    }

    @objc(tokenize:text:resolver:rejecter:)
    func tokenize(_ handle: String, text: String, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var buffer = [UInt32](repeating: 0, count: defaultTokenBufferSize)
        var outLen: Int = 0
        let rc = cactus_tokenize(nativeHandle, text, &buffer, buffer.count, &outLen)
        guard rc >= 0 else {
            self.reject(reject, "Tokenization failed")
            return
        }
        resolve(buffer.prefix(outLen).map(Int.init))
    }

    @objc(scoreWindow:tokens:start:end:context:resolver:rejecter:)
    func scoreWindow(
        _ handle: String,
        tokens: [NSNumber],
        start: NSNumber,
        end: NSNumber,
        context: NSNumber,
        resolver resolve: RCTPromiseResolveBlock,
        rejecter reject: RCTPromiseRejectBlock
    ) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var tokenValues = tokens.map { UInt32(truncating: $0) }
        var buffer = [CChar](repeating: 0, count: defaultBufferSize)
        let rc = cactus_score_window(nativeHandle, &tokenValues, tokenValues.count, start.intValue, end.intValue, context.intValue, &buffer, buffer.count)
        guard rc >= 0 else {
            self.reject(reject, "Score window failed")
            return
        }
        resolve(stringFromBuffer(buffer))
    }

    @objc(transcribe:audioPath:prompt:optionsJson:pcmDataBase64:resolver:rejecter:)
    func transcribe(
        _ handle: String,
        audioPath: String?,
        prompt: String?,
        optionsJson: String?,
        pcmDataBase64: String?,
        resolver resolve: RCTPromiseResolveBlock,
        rejecter reject: RCTPromiseRejectBlock
    ) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var buffer = [CChar](repeating: 0, count: defaultBufferSize)
        let promptValue = prompt ?? ""
        let rc = decodeBase64(pcmDataBase64)?.withUnsafeBytes { rawBuffer in
            cactus_transcribe(nativeHandle, audioPath, promptValue, &buffer, buffer.count, optionsJson, nil, nil, rawBuffer.bindMemory(to: UInt8.self).baseAddress, rawBuffer.count)
        } ?? cactus_transcribe(nativeHandle, audioPath, promptValue, &buffer, buffer.count, optionsJson, nil, nil, nil, 0)
        guard rc >= 0 else {
            self.reject(reject, "Transcription failed")
            return
        }
        resolve(stringFromBuffer(buffer))
    }

    @objc(embed:text:normalize:resolver:rejecter:)
    func embed(_ handle: String, text: String, normalize: Bool, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var buffer = [Float](repeating: 0, count: defaultEmbeddingBufferSize)
        var dim: Int = 0
        let rc = cactus_embed(nativeHandle, text, &buffer, buffer.count, &dim, normalize)
        guard rc >= 0 else {
            self.reject(reject, "Embedding failed")
            return
        }
        resolve(buffer.prefix(dim).map(Double.init))
    }

    @objc(imageEmbed:imagePath:resolver:rejecter:)
    func imageEmbed(_ handle: String, imagePath: String, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var buffer = [Float](repeating: 0, count: defaultEmbeddingBufferSize)
        var dim: Int = 0
        let rc = cactus_image_embed(nativeHandle, imagePath, &buffer, buffer.count, &dim)
        guard rc >= 0 else {
            self.reject(reject, "Image embedding failed")
            return
        }
        resolve(buffer.prefix(dim).map(Double.init))
    }

    @objc(audioEmbed:audioPath:resolver:rejecter:)
    func audioEmbed(_ handle: String, audioPath: String, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var buffer = [Float](repeating: 0, count: defaultEmbeddingBufferSize)
        var dim: Int = 0
        let rc = cactus_audio_embed(nativeHandle, audioPath, &buffer, buffer.count, &dim)
        guard rc >= 0 else {
            self.reject(reject, "Audio embedding failed")
            return
        }
        resolve(buffer.prefix(dim).map(Double.init))
    }

    @objc(vad:audioPath:optionsJson:pcmDataBase64:resolver:rejecter:)
    func vad(
        _ handle: String,
        audioPath: String?,
        optionsJson: String?,
        pcmDataBase64: String?,
        resolver resolve: RCTPromiseResolveBlock,
        rejecter reject: RCTPromiseRejectBlock
    ) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var buffer = [CChar](repeating: 0, count: defaultBufferSize)
        let rc = decodeBase64(pcmDataBase64)?.withUnsafeBytes { rawBuffer in
            cactus_vad(nativeHandle, audioPath, &buffer, buffer.count, optionsJson, rawBuffer.bindMemory(to: UInt8.self).baseAddress, rawBuffer.count)
        } ?? cactus_vad(nativeHandle, audioPath, &buffer, buffer.count, optionsJson, nil, 0)
        guard rc >= 0 else {
            self.reject(reject, "VAD failed")
            return
        }
        resolve(stringFromBuffer(buffer))
    }

    @objc(diarize:audioPath:optionsJson:pcmDataBase64:resolver:rejecter:)
    func diarize(
        _ handle: String,
        audioPath: String?,
        optionsJson: String?,
        pcmDataBase64: String?,
        resolver resolve: RCTPromiseResolveBlock,
        rejecter reject: RCTPromiseRejectBlock
    ) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var buffer = [CChar](repeating: 0, count: largeBufferSize)
        let rc = decodeBase64(pcmDataBase64)?.withUnsafeBytes { rawBuffer in
            cactus_diarize(nativeHandle, audioPath, &buffer, buffer.count, optionsJson, rawBuffer.bindMemory(to: UInt8.self).baseAddress, rawBuffer.count)
        } ?? cactus_diarize(nativeHandle, audioPath, &buffer, buffer.count, optionsJson, nil, 0)
        guard rc >= 0 else {
            self.reject(reject, "Diarize failed")
            return
        }
        resolve(stringFromBuffer(buffer))
    }

    @objc(embedSpeaker:audioPath:optionsJson:pcmDataBase64:maskWeights:resolver:rejecter:)
    func embedSpeaker(
        _ handle: String,
        audioPath: String?,
        optionsJson: String?,
        pcmDataBase64: String?,
        maskWeights: [NSNumber]?,
        resolver resolve: RCTPromiseResolveBlock,
        rejecter reject: RCTPromiseRejectBlock
    ) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var buffer = [CChar](repeating: 0, count: defaultBufferSize)
        var mask = maskWeights?.map { Float(truncating: $0) }
        let rc = decodeBase64(pcmDataBase64)?.withUnsafeBytes { rawBuffer in
            if var mask {
                return cactus_embed_speaker(nativeHandle, audioPath, &buffer, buffer.count, optionsJson, rawBuffer.bindMemory(to: UInt8.self).baseAddress, rawBuffer.count, &mask, mask.count)
            }
            return cactus_embed_speaker(nativeHandle, audioPath, &buffer, buffer.count, optionsJson, rawBuffer.bindMemory(to: UInt8.self).baseAddress, rawBuffer.count, nil, 0)
        } ?? {
            if var mask {
                return cactus_embed_speaker(nativeHandle, audioPath, &buffer, buffer.count, optionsJson, nil, 0, &mask, mask.count)
            }
            return cactus_embed_speaker(nativeHandle, audioPath, &buffer, buffer.count, optionsJson, nil, 0, nil, 0)
        }()
        guard rc >= 0 else {
            self.reject(reject, "EmbedSpeaker failed")
            return
        }
        resolve(stringFromBuffer(buffer))
    }

    @objc(ragQuery:query:topK:resolver:rejecter:)
    func ragQuery(_ handle: String, query: String, topK: NSNumber, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var buffer = [CChar](repeating: 0, count: defaultBufferSize)
        let rc = cactus_rag_query(nativeHandle, query, &buffer, buffer.count, topK.intValue)
        guard rc >= 0 else {
            self.reject(reject, "RAG query failed")
            return
        }
        resolve(stringFromBuffer(buffer))
    }

    @objc(indexInit:embeddingDim:resolver:rejecter:)
    func indexInit(_ indexDir: String, embeddingDim: NSNumber, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        let handle = cactus_index_init(indexDir, embeddingDim.intValue)
        guard let handle else {
            self.reject(reject, "Failed to initialize index")
            return
        }
        resolve(encodeHandle(handle))
    }

    @objc(indexAdd:ids:documents:embeddings:metadatas:resolver:rejecter:)
    func indexAdd(
        _ handle: String,
        ids: [NSNumber],
        documents: [String],
        embeddings: [[NSNumber]],
        metadatas: [String]?,
        resolver resolve: RCTPromiseResolveBlock,
        rejecter reject: RCTPromiseRejectBlock
    ) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var idValues = ids.map { Int32(truncating: $0) }
        var docStrings = documents.map { strdup($0) }
        var docPointers = docStrings.map { UnsafePointer<CChar>($0) }
        var metaStrings = metadatas?.map { strdup($0) }
        var metaPointers = metaStrings?.map { UnsafePointer<CChar>($0) }
        let embeddingDim = embeddings.first?.count ?? 0
        var embeddingPointers = embeddings.map { row -> UnsafeMutablePointer<Float> in
            let ptr = UnsafeMutablePointer<Float>.allocate(capacity: row.count)
            for (index, value) in row.enumerated() {
                ptr[index] = Float(truncating: value)
            }
            return ptr
        }
        var embeddingConstPointers: [UnsafePointer<Float>?] = embeddingPointers.map { UnsafePointer<Float>($0) }
        defer {
            docStrings.forEach { free($0) }
            metaStrings?.forEach { free($0) }
            embeddingPointers.forEach { $0.deallocate() }
        }
        var idsCopy = idValues
        let rc = idsCopy.withUnsafeMutableBufferPointer { idBuffer in
            docPointers.withUnsafeMutableBufferPointer { docBuffer in
                embeddingConstPointers.withUnsafeMutableBufferPointer { embBuffer in
                    if var metaPointers {
                        return metaPointers.withUnsafeMutableBufferPointer { metaBuffer in
                            cactus_index_add(
                                nativeHandle,
                                idBuffer.baseAddress,
                                docBuffer.baseAddress,
                                metaBuffer.baseAddress,
                                embBuffer.baseAddress,
                                idValues.count,
                                embeddingDim
                            )
                        }
                    }
                    return cactus_index_add(
                        nativeHandle,
                        idBuffer.baseAddress,
                        docBuffer.baseAddress,
                        nil,
                        embBuffer.baseAddress,
                        idValues.count,
                        embeddingDim
                    )
                }
            }
        }
        guard rc >= 0 else {
            self.reject(reject, "Failed to add to index")
            return
        }
        resolve(nil)
    }

    @objc(indexDelete:ids:resolver:rejecter:)
    func indexDelete(_ handle: String, ids: [NSNumber], resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var idValues = ids.map { Int32(truncating: $0) }
        let rc = cactus_index_delete(nativeHandle, &idValues, idValues.count)
        guard rc >= 0 else {
            self.reject(reject, "Failed to delete from index")
            return
        }
        resolve(nil)
    }

    @objc(indexGet:ids:resolver:rejecter:)
    func indexGet(_ handle: String, ids: [NSNumber], resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var idValues = ids.map { Int32(truncating: $0) }
        let count = idValues.count
        var docPointers: [UnsafeMutablePointer<CChar>?] = (0..<count).map { _ in UnsafeMutablePointer<CChar>.allocate(capacity: defaultIndexDocBufferSize) }
        var docSizes = Array(repeating: defaultIndexDocBufferSize, count: count)
        var metaPointers: [UnsafeMutablePointer<CChar>?] = (0..<count).map { _ in UnsafeMutablePointer<CChar>.allocate(capacity: defaultIndexDocBufferSize) }
        var metaSizes = Array(repeating: defaultIndexDocBufferSize, count: count)
        var embPointers: [UnsafeMutablePointer<Float>?] = (0..<count).map { _ in UnsafeMutablePointer<Float>.allocate(capacity: defaultIndexEmbeddingBufferSize) }
        var embSizes = Array(repeating: defaultIndexEmbeddingBufferSize, count: count)
        defer {
            docPointers.forEach { $0?.deallocate() }
            metaPointers.forEach { $0?.deallocate() }
            embPointers.forEach { $0?.deallocate() }
        }
        for i in 0..<count {
            docPointers[i]?.initialize(repeating: 0, count: defaultIndexDocBufferSize)
            metaPointers[i]?.initialize(repeating: 0, count: defaultIndexDocBufferSize)
            embPointers[i]?.initialize(repeating: 0, count: defaultIndexEmbeddingBufferSize)
        }
        let rc = idValues.withUnsafeMutableBufferPointer { idBuffer in
            docPointers.withUnsafeMutableBufferPointer { docBuffer in
                docSizes.withUnsafeMutableBufferPointer { docSizeBuffer in
                    metaPointers.withUnsafeMutableBufferPointer { metaBuffer in
                        metaSizes.withUnsafeMutableBufferPointer { metaSizeBuffer in
                            embPointers.withUnsafeMutableBufferPointer { embBuffer in
                                embSizes.withUnsafeMutableBufferPointer { embSizeBuffer in
                                    cactus_index_get(
                                        nativeHandle,
                                        idBuffer.baseAddress,
                                        count,
                                        docBuffer.baseAddress,
                                        docSizeBuffer.baseAddress,
                                        metaBuffer.baseAddress,
                                        metaSizeBuffer.baseAddress,
                                        embBuffer.baseAddress,
                                        embSizeBuffer.baseAddress
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
        guard rc >= 0 else {
            self.reject(reject, "Failed to get from index")
            return
        }
        var results = [[String: Any]]()
        for i in 0..<count {
            let document = docPointers[i].map { String(cString: $0) } ?? ""
            let metadataString = metaPointers[i].map { String(cString: $0) } ?? ""
            let metadata: Any = metadataString.isEmpty ? NSNull() : metadataString
            let embedding = (0..<embSizes[i]).map { index in Double(embPointers[i]![index]) }
            results.append(["document": document, "metadata": metadata, "embedding": embedding])
        }
        if let data = try? JSONSerialization.data(withJSONObject: ["results": results]),
           let json = String(data: data, encoding: .utf8) {
            resolve(json)
        } else {
            self.reject(reject, "Failed to encode index get results")
        }
    }

    @objc(indexQuery:embedding:optionsJson:resolver:rejecter:)
    func indexQuery(_ handle: String, embedding: [NSNumber], optionsJson: String?, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        var embValues = embedding.map { Float(truncating: $0) }
        var idStorage = [Int32](repeating: 0, count: defaultIndexResultCapacity)
        var scoreStorage = [Float](repeating: 0, count: defaultIndexResultCapacity)
        var idSizes = [defaultIndexResultCapacity]
        var scoreSizes = [defaultIndexResultCapacity]
        var embCopy = embValues
        let rc = embCopy.withUnsafeMutableBufferPointer { embBuffer in
            var embPointer: UnsafePointer<Float>? = UnsafePointer(embBuffer.baseAddress)
            return idStorage.withUnsafeMutableBufferPointer { idBuffer in
                var idPointer: UnsafeMutablePointer<Int32>? = idBuffer.baseAddress
                return scoreStorage.withUnsafeMutableBufferPointer { scoreBuffer in
                    var scorePointer: UnsafeMutablePointer<Float>? = scoreBuffer.baseAddress
                    return cactus_index_query(
                        nativeHandle,
                        &embPointer,
                        1,
                        embValues.count,
                        optionsJson,
                        &idPointer,
                        &idSizes,
                        &scorePointer,
                        &scoreSizes
                    )
                }
            }
        }
        guard rc >= 0 else {
            self.reject(reject, "Index query failed")
            return
        }
        let results = (0..<idSizes[0]).map { index in
            ["id": Int(idStorage[index]), "score": Double(scoreStorage[index])]
        }
        if let data = try? JSONSerialization.data(withJSONObject: ["results": results]),
           let json = String(data: data, encoding: .utf8) {
            resolve(json)
        } else {
            self.reject(reject, "Failed to encode index query results")
        }
    }

    @objc(indexCompact:resolver:rejecter:)
    func indexCompact(_ handle: String, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        let rc = cactus_index_compact(nativeHandle)
        guard rc >= 0 else {
            self.reject(reject, "Failed to compact index")
            return
        }
        resolve(nil)
    }

    @objc(indexDestroy:resolver:rejecter:)
    func indexDestroy(_ handle: String, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
        guard let nativeHandle = decodeHandle(handle) else {
            self.reject(reject, "Invalid native handle")
            return
        }
        cactus_index_destroy(nativeHandle)
        resolve(nil)
    }

    @objc(logSetLevel:resolver:rejecter:)
    func logSetLevel(_ level: NSNumber, resolver resolve: RCTPromiseResolveBlock, rejecter _: RCTPromiseRejectBlock) {
        cactus_log_set_level(level.int32Value)
        resolve(nil)
    }

    @objc(setTelemetryEnvironment:cacheLocation:version:resolver:rejecter:)
    func setTelemetryEnvironment(
        _ framework: String?,
        cacheLocation: String?,
        version: String?,
        resolver resolve: RCTPromiseResolveBlock,
        rejecter _: RCTPromiseRejectBlock
    ) {
        cactus_set_telemetry_environment(framework, cacheLocation, version)
        resolve(nil)
    }

    @objc(setAppId:resolver:rejecter:)
    func setAppId(_ appId: String, resolver resolve: RCTPromiseResolveBlock, rejecter _: RCTPromiseRejectBlock) {
        cactus_set_app_id(appId)
        resolve(nil)
    }

    @objc(telemetryFlush:rejecter:)
    func telemetryFlush(_ resolve: RCTPromiseResolveBlock, rejecter _: RCTPromiseRejectBlock) {
        cactus_telemetry_flush()
        resolve(nil)
    }

    @objc(telemetryShutdown:rejecter:)
    func telemetryShutdown(_ resolve: RCTPromiseResolveBlock, rejecter _: RCTPromiseRejectBlock) {
        cactus_telemetry_shutdown()
        resolve(nil)
    }

    @objc(getLastError:rejecter:)
    func getLastError(_ resolve: RCTPromiseResolveBlock, rejecter _: RCTPromiseRejectBlock) {
        guard let ptr = cactus_get_last_error() else {
            resolve(nil)
            return
        }
        resolve(String(cString: ptr))
    }
}
