package com.cactus

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.URL

private var currentHandle: Long? = null

private val applicationContext: Context by lazy {
    CactusContextInitializer.getApplicationContext()
}

actual suspend fun downloadModel(url: String, filename: String): Boolean {
    return withContext(Dispatchers.IO) {
        try {
            val modelsDir = File(applicationContext.cacheDir, "models")
            if (!modelsDir.exists()) modelsDir.mkdirs()
            
            val modelFile = File(modelsDir, filename)
            if (modelFile.exists()) return@withContext true
            
            val urlConnection = URL(url).openConnection()
            urlConnection.getInputStream().use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            modelFile.exists()
        } catch (e: Exception) {
            false
        }
    }
}

actual suspend fun loadModel(path: String, threads: Int, contextSize: Int, batchSize: Int, gpuLayers: Int): Long? {
    return withContext(Dispatchers.Default) {
        try {
            Log.d("CactusLM", "Loading model: $path")
            val modelsDir = File(applicationContext.cacheDir, "models")
            val modelFile = File(modelsDir, path)
            
            if (!modelFile.exists()) {
                Log.e("CactusLM", "Model file not found: ${modelFile.absolutePath}")
                return@withContext null
            }
            
            Log.d("CactusLM", "Model file found: ${modelFile.absolutePath}")
            
            val actualGpuLayers = 0
            if (gpuLayers == 99) {
                Log.w("CactusLM", "GPU acceleration requested but not supported on Android. Using CPU.")
            }
            
            Log.d("CactusLM", "GPU layers requested: $gpuLayers, actual: $actualGpuLayers (Android always uses CPU)")
            Log.d("CactusLM", "DEBUGGING: About to create CactusInitParams with useMmap = true")
            val params = CactusInitParams(
                modelPath = modelFile.absolutePath,
                nCtx = contextSize,
                nThreads = threads,
                nBatch = batchSize,
                nGpuLayers = actualGpuLayers,
                useMmap = true
            )
            Log.d("CactusLM", "DEBUGGING: Created CactusInitParams, useMmap = ${params.useMmap}")
            Log.d("CactusLM", "DEBUGGING: useMmap type = ${params.useMmap::class.simpleName}")
            Log.d("CactusLM", "DEBUGGING: useMmap toString() = ${params.useMmap.toString()}")
            Log.d("CactusLM", "DEBUGGING: useMmap === true = ${params.useMmap === true}")
            Log.d("CactusLM", "DEBUGGING: useMmap == true = ${params.useMmap == true}")
            
            Log.d("CactusLM", "Created CactusInitParams:")
            Log.d("CactusLM", "  modelPath: ${params.modelPath}")
            Log.d("CactusLM", "  nCtx: ${params.nCtx}")
            Log.d("CactusLM", "  nThreads: ${params.nThreads}")
            Log.d("CactusLM", "  nBatch: ${params.nBatch}")
            Log.d("CactusLM", "  nGpuLayers: ${params.nGpuLayers}")
            Log.d("CactusLM", "  useMmap: ${params.useMmap}")
            Log.d("CactusLM", "  useMlock: ${params.useMlock}")
            
            Log.d("CactusLM", "Initializing context...")
            val handle = CactusContext.initContext(params)
            if (handle != null) {
                currentHandle = handle
                Log.d("CactusLM", "Model loaded successfully with handle: $handle")
                handle
            } else {
                Log.e("CactusLM", "Failed to initialize context")
                null
            }
        } catch (e: Exception) {
            Log.e("CactusLM", "Exception loading model: ${e.message}", e)
            null
        }
    }
}

actual suspend fun generateCompletion(handle: Long, prompt: String, maxTokens: Int, temperature: Float, topP: Float): CactusCompletionResult? {
    return withContext(Dispatchers.Default) {
        try {
            Log.d("CactusLM", "Generating completion for prompt: ${prompt.take(50)}...")
            val params = CactusCompletionParams(
                prompt = prompt,
                nPredict = maxTokens,
                temperature = temperature.toDouble(),
                topP = topP.toDouble()
            )
            
            Log.d("CactusLM", "Calling CactusContext.completion...")
            val result = CactusContext.completion(handle, params)
            Log.d("CactusLM", "Completion result: ${result.text?.take(50)}...")
            result
        } catch (e: Exception) {
            Log.e("CactusLM", "Exception generating completion: ${e.message}", e)
            null
        }
    }
}

actual suspend fun generateStreamingCompletion(handle: Long, prompt: String, maxTokens: Int, temperature: Float, topP: Float, onNewToken: (String) -> Boolean): CactusCompletionResult? {
    return withContext(Dispatchers.Default) {
        try {
            Log.d("CactusLM", "Generating streaming completion for prompt: ${prompt.take(50)}...")
            val params = CactusCompletionParams(
                prompt = prompt,
                nPredict = maxTokens,
                temperature = temperature.toDouble(),
                topP = topP.toDouble(),
                onNewToken = onNewToken
            )
            
            Log.d("CactusLM", "Calling CactusContext.completion with streaming...")
            val result = CactusContext.completion(handle, params)
            Log.d("CactusLM", "Streaming completion finished: ${result.text?.take(50)}...")
            result
        } catch (e: Exception) {
            Log.e("CactusLM", "Exception generating streaming completion: ${e.message}", e)
            null
        }
    }
}

actual fun unloadModel(handle: Long) {
    try {
        CactusContext.freeContext(handle)
        if (currentHandle == handle) {
            currentHandle = null
        }
    } catch (e: Exception) {
    }
} 