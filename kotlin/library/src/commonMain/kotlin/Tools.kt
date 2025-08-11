package com.cactus

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive

@Serializable
data class Parameter(
    val type: String,
    val description: String,
    val required: Boolean = false
)

interface ToolExecutor {
    suspend fun execute(args: Map<String, Any>): Any
}

@Serializable
data class ToolSchema(
    val type: String,
    val function: FunctionSchema
)

@Serializable
data class FunctionSchema(
    val name: String,
    val description: String,
    val parameters: ParametersSchema
)

@Serializable
data class ParametersSchema(
    val type: String,
    val properties: Map<String, Parameter>,
    val required: List<String>
)

data class Tool(
    val func: ToolExecutor,
    val description: String,
    val parameters: Map<String, Parameter>,
    val required: List<String>
)

class Tools {
    private val tools = mutableMapOf<String, Tool>()

    fun add(
        name: String,
        func: ToolExecutor,
        description: String,
        parameters: Map<String, Parameter>
    ) {
        val required = parameters.entries
            .filter { (_, param) -> param.required }
            .map { (key, _) -> key }

        tools[name] = Tool(
            func = func,
            description = description,
            parameters = parameters,
            required = required
        )
    }

    fun getSchemas(): List<ToolSchema> {
        return tools.entries.map { (name, tool) ->
            ToolSchema(
                type = "function",
                function = FunctionSchema(
                    name = name,
                    description = tool.description,
                    parameters = ParametersSchema(
                        type = "object",
                        properties = tool.parameters,
                        required = tool.required
                    )
                )
            )
        }
    }

    suspend fun execute(name: String, args: Map<String, Any>): Any {
        val tool = tools[name] ?: throw IllegalArgumentException("Tool $name not found")
        return tool.func.execute(args)
    }

    fun isEmpty(): Boolean = tools.isEmpty()
}

@Serializable
data class ToolCallResult(
    val toolCalled: Boolean,
    val toolName: String? = null,
    val toolInput: Map<String, String>? = null,
    val toolOutput: String? = null
)

@Serializable
data class ModelToolCall(
    val name: String,
    val arguments: Map<String, JsonElement>
)

@Serializable
data class ModelResponse(
    val tool_calls: List<ModelToolCall>? = null,
    val tool_call: List<ModelToolCall>? = null
)

suspend fun parseAndExecuteTool(
    modelResponse: String?,
    tools: Tools
): ToolCallResult {
    if (modelResponse.isNullOrBlank()) {
        return ToolCallResult(toolCalled = false)
    }

    try {
        val json = Json { ignoreUnknownKeys = true }

        val possibleJsonBlocks = extractJsonBlocks(modelResponse)

        for (jsonBlock in possibleJsonBlocks) {
            try {
                val response = json.decodeFromString<ModelResponse>(jsonBlock)

                if (!response.tool_calls.isNullOrEmpty()) {
                    val toolCall = response.tool_calls.first()
                    val toolName = toolCall.name
                    val toolInput = convertJsonElementsToAny(toolCall.arguments)

                    val toolOutput = tools.execute(toolName, toolInput)

                    return ToolCallResult(
                        toolCalled = true,
                        toolName = toolName,
                        toolInput = toolInput.mapValues { it.value.toString() },
                        toolOutput = toolOutput.toString()
                    )
                } else if (!response.tool_call.isNullOrEmpty()) {
                    val toolCall = response.tool_call.first()
                    val toolName = toolCall.name
                    val toolInput = convertJsonElementsToAny(toolCall.arguments)

                    val toolOutput = tools.execute(toolName, toolInput)

                    return ToolCallResult(
                        toolCalled = true,
                        toolName = toolName,
                        toolInput = toolInput.mapValues { it.value.toString() },
                        toolOutput = toolOutput.toString()
                    )
                }
            } catch (e: Exception) {
                continue
            }
        }

        return ToolCallResult(toolCalled = false)
    } catch (error: Exception) {
        return ToolCallResult(toolCalled = false)
    }
}

private fun extractJsonBlocks(response: String): List<String> {
    val jsonBlocks = mutableListOf<String>()

    if (response.contains("\"tool_calls\"") || response.contains("\"tool_call\"")) {
        var braceCount = 0
        var startIndex = -1
        var endIndex = -1

        for (i in response.indices) {
            when (response[i]) {
                '{' -> {
                    if (braceCount == 0) startIndex = i
                    braceCount++
                }
                '}' -> {
                    braceCount--
                    if (braceCount == 0 && startIndex != -1) {
                        endIndex = i
                        val candidate = response.substring(startIndex, endIndex + 1)
                        if ((candidate.contains("\"tool_calls\"") || candidate.contains("\"tool_call\""))
                            && !jsonBlocks.contains(candidate)) {
                            jsonBlocks.add(candidate)
                        }
                        startIndex = -1
                    }
                }
            }
        }
    }

    return jsonBlocks
}

private fun convertJsonElementsToAny(jsonElements: Map<String, JsonElement>): Map<String, Any> {
    return jsonElements.mapValues { (_, jsonElement) ->
        convertJsonElementToAny(jsonElement)
    }
}

private fun convertJsonElementToAny(jsonElement: JsonElement): Any {
    if (jsonElement !is JsonPrimitive) {
        return jsonElement.toString()
    }

    val primitive = jsonElement.jsonPrimitive

    return when {
        primitive.booleanOrNull != null -> primitive.boolean
        primitive.intOrNull != null -> primitive.int
        primitive.doubleOrNull != null -> primitive.double
        else -> primitive.content
    }
}
