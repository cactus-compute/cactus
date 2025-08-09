package com.cactus

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

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
    val arguments: Map<String, String>
)

@Serializable
data class ModelResponse(
    val tool_calls: List<ModelToolCall>? = null
)

suspend fun parseAndExecuteTool(
    modelResponse: String?,
    tools: Tools
): ToolCallResult {
    if (modelResponse.isNullOrBlank()) {
        return ToolCallResult(toolCalled = false)
    }

    try {
        // Extract JSON from the response (it might be wrapped with thinking tags or other text)
        val jsonStart = modelResponse.indexOf("{")
        val jsonEnd = modelResponse.lastIndexOf("}")

        if (jsonStart == -1 || jsonEnd == -1 || jsonStart >= jsonEnd) {
            return ToolCallResult(toolCalled = false)
        }

        val jsonPart = modelResponse.substring(jsonStart, jsonEnd + 1)

        val json = Json { ignoreUnknownKeys = true }
        val response = json.decodeFromString<ModelResponse>(jsonPart)

        if (response.tool_calls.isNullOrEmpty()) {
            return ToolCallResult(toolCalled = false)
        }

        val toolCall = response.tool_calls.first()
        val toolName = toolCall.name
        val toolInput = toolCall.arguments

        val toolOutput = tools.execute(toolName, toolInput)

        return ToolCallResult(
            toolCalled = true,
            toolName = toolName,
            toolInput = toolInput,
            toolOutput = toolOutput.toString()
        )
    } catch (error: Exception) {
        return ToolCallResult(toolCalled = false)
    }
}
