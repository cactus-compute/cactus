package com.cactus

import com.cactus.CactusContext.getFormattedChatWithJinja
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import com.cactus.downloadModel
import com.cactus.generateCompletion
import com.cactus.loadModel
import com.cactus.unloadModel

@Serializable
data class CompletionResult(
    val content: String?,
    val toolCalls: List<String>? = null,
)

class CactusAgent(
    private val threads: Int = 4,
    private val contextSize: Int = 2048,
    private val batchSize: Int = 512,
    private val gpuLayers: Int = 0
) {
    private var handle: Long? = null
    private var lastDownloadedFilename: String? = null
    private val tools = Tools()

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
        return handle?.let { h ->
            generateCompletion(h, prompt, maxTokens, temperature, topP)
        }
    }

    fun unload() {
        handle?.let { h ->
            unloadModel(h)
            handle = null
        }
    }

    fun isLoaded(): Boolean = handle != null

    fun addTool(
        name: String,
        function: ToolExecutor,
        description: String,
        parameters: Map<String, Parameter>
    ) {
        tools.add(name, function, description, parameters)
    }

    suspend fun completionWithTools(
        message: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
        callback: ((String) -> Unit)? = null
    ): CompletionResult {
        if (tools.isEmpty()) {
            val response = completion(message, maxTokens, temperature, topP)
            return CompletionResult(
                content = response ?: "",
                toolCalls = null
            )
        }
        if (handle == null) {
            return CompletionResult("Handle is null", null)
        }

        val toolsJson = Json.encodeToString(ListSerializer(ToolSchema.serializer()), tools.getSchemas())

        val formattedPromptResult = handle?.let {
            getFormattedChatWithJinja(
                handle = it,
                messages = message,
                chatTemplate = "chatml",
                jsonSchema = null,
                tools = toolsJson,
                parallelToolCalls = true,
                toolChoice = null
            )
        }

        if (formattedPromptResult == null) {
            return CompletionResult("Failed to format prompt", null)
        }

        // Generate completion using the formatted prompt
        val modelResponse = handle?.let { h ->
            val response = generateCompletion(h, formattedPromptResult.prompt, maxTokens, temperature, topP)
            response
        }

        val toolResult: ToolCallResult = parseAndExecuteTool(modelResponse, tools)

        return CompletionResult(
            content = if (toolResult.toolCalled) toolResult.toolOutput else modelResponse,
            toolCalls = if (toolResult.toolCalled) listOf(toolResult.toolName!!) else null
        )
    }
}