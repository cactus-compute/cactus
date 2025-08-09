package com.cactus.example

import com.cactus.ToolExecutor

class WeatherTool : ToolExecutor {
    override suspend fun execute(args: Map<String, Any>): Any {
        val location = args["location"] as? String ?: "unknown"
        return "The weather in $location is sunny, 72°F"
    }
}