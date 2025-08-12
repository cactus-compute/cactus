package com.cactus

class CactusLM(
    private val threads: Int = 4,
    private val contextSize: Int = 2048,
    private val batchSize: Int = 512,
    private val gpuLayers: Int = 0
) {
    private var handle: Long? = null
    private var lastDownloadedFilename: String? = null
    private val historyManager = ConversationHistoryManager()
    
    suspend fun download(
        url: String = "https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
        filename: String? = null
    ): Boolean {
        val actualFilename = filename ?: url.substringAfterLast("/")
        val success = downloadModel(url, actualFilename)
        if (success) {
            lastDownloadedFilename = actualFilename
        }
        return success
    }
    
    suspend fun init(filename: String = lastDownloadedFilename ?: "Qwen3-0.6B-Q8_0.gguf"): Boolean {
        handle = loadModel(filename, threads, contextSize, batchSize, gpuLayers)
        return handle != null
    }
    
    suspend fun completion(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): String? {
        val processedMessages = historyManager.processNewMessages(listOf(ChatMessage(prompt, "user")))
        if (processedMessages.requiresReset) {
            historyManager.reset()
        }
        val response = handle?.let { h ->
            generateCompletion(h, prompt, maxTokens, temperature, topP)
        } ?: ""
        historyManager.update(processedMessages.newMessages, ChatMessage(response, "assistant"))
        return response
    }
    
    fun unload() {
        handle?.let { h ->
            unloadModel(h)
            handle = null
        }
    }
    
    fun isLoaded(): Boolean = handle != null
}

expect suspend fun downloadModel(url: String, filename: String): Boolean
expect suspend fun loadModel(path: String, threads: Int, contextSize: Int, batchSize: Int, gpuLayers: Int): Long?
expect suspend fun generateCompletion(handle: Long, prompt: String, maxTokens: Int, temperature: Float, topP: Float): String?
expect fun unloadModel(handle: Long) 