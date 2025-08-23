package com.cactus

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import platform.Foundation.*

private var currentHandle: Long? = null

@OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
actual suspend fun downloadModel(url: String, filename: String): Boolean {
    return withContext(Dispatchers.Default) {
        try {
            val documentsDir = NSSearchPathForDirectoriesInDomains(
                NSDocumentDirectory, NSUserDomainMask, true
            ).firstOrNull() as? String ?: return@withContext false
            
            val modelsDir = "$documentsDir/models"
            NSFileManager.defaultManager.createDirectoryAtPath(
                modelsDir, true, null, null
            )
            
            val modelPath = "$modelsDir/$filename"
            if (NSFileManager.defaultManager.fileExistsAtPath(modelPath)) {
                return@withContext true
            }
            
            val nsUrl = NSURL(string = url) ?: return@withContext false
            val data = NSData.dataWithContentsOfURL(nsUrl) ?: return@withContext false
            
            data.writeToFile(modelPath, true)
        } catch (e: Exception) {
            false
        }
    }
}

actual suspend fun loadModel(path: String, threads: Int, contextSize: Int, batchSize: Int, gpuLayers: Int): Long? {
    return withContext(Dispatchers.Default) {
        try {
            val documentsDir = NSSearchPathForDirectoriesInDomains(
                NSDocumentDirectory, NSUserDomainMask, true
            ).firstOrNull() as? String ?: return@withContext null
            
            val fullModelPath = "$documentsDir/models/$path"
            if (!NSFileManager.defaultManager.fileExistsAtPath(fullModelPath)) {
                return@withContext null
            }
            
            val actualGpuLayers = when (gpuLayers) {
                0 -> 0
                99 -> 99
                else -> 0
            }
            
            val params = CactusInitParams(
                modelPath = fullModelPath,
                nCtx = contextSize,
                nThreads = threads,
                nBatch = batchSize,
                nGpuLayers = actualGpuLayers
            )
            
            val handle = CactusContext.initContext(params)
            if (handle != null) {
                currentHandle = handle
                handle
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }
    }
}

actual suspend fun generateCompletion(handle: Long, prompt: String, maxTokens: Int, temperature: Float, topP: Float): CactusCompletionResult? {
    return withContext(Dispatchers.Default) {
        try {
            val params = CactusCompletionParams(
                prompt = prompt,
                nPredict = maxTokens,
                temperature = temperature.toDouble(),
                topP = topP.toDouble()
            )
            
            val result = CactusContext.completion(handle, params)
            result
        } catch (e: Exception) {
            null
        }
    }
}

actual suspend fun generateStreamingCompletion(handle: Long, prompt: String, maxTokens: Int, temperature: Float, topP: Float, onNewToken: (String) -> Boolean): CactusCompletionResult? {
    return withContext(Dispatchers.Default) {
        try {
            val params = CactusCompletionParams(
                prompt = prompt,
                nPredict = maxTokens,
                temperature = temperature.toDouble(),
                topP = topP.toDouble(),
                onNewToken = onNewToken
            )
            
            val result = CactusContext.completion(handle, params)
            result
        } catch (e: Exception) {
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