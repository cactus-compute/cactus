package com.cactus

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import platform.Foundation.*

@OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
actual suspend fun downloadSTTModel(url: String, filename: String): Boolean {
    return withContext(Dispatchers.Default) {
        try {
            val documentsDir = NSSearchPathForDirectoriesInDomains(
                NSDocumentDirectory, NSUserDomainMask, true
            ).firstOrNull() as? String ?: return@withContext false
            
            val modelsDir = "$documentsDir/models/vosk"
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

actual suspend fun initializeSTT(): Boolean {
    return try {
        val initialized = initializeSpeechRecognition()
        if (initialized) {
            println("iOS Speech recognition initialized successfully")
        } else {
            println("Failed to initialize iOS speech recognition")
        }
        initialized
    } catch (e: Exception) {
        println("Error initializing iOS speech recognition: $e")
        false
    }
}

actual suspend fun performSTT(language: String, maxDuration: Int): STTResult? {
    return try {
        println("iOS performSTT called with language=$language, maxDuration=$maxDuration")
        
        if (!isSpeechRecognitionAuthorized()) {
            println("Requesting speech permissions...")
            val permissionGranted = requestSpeechPermissions()
            if (!permissionGranted) {
                println("Speech recognition permission not granted")
                return null
            }
            println("Speech recognition permission granted")
        }
        
        if (!isSpeechRecognitionAvailable()) {
            println("Speech recognition not available")
            return null
        }
        
        if (!isVoskModelLoaded()) {
            println("Vosk model not loaded on this device")
            return STTResult(
                text = "",
                confidence = 0.0f,
                isPartial = false
            )
        }
        
        println("Creating speech recognition params (on-device mode)...")
        val params = SpeechRecognitionParams(
            language = language,
            enablePartialResults = false,
            maxDuration = maxDuration
        )
        
        println("Calling performSpeechRecognition...")
        val speechResult = performSpeechRecognition(params)
        
        println("performSpeechRecognition returned: $speechResult")
        
        if (speechResult == null) {
            println("❌ speechResult is null")
            return null
        }
        
        println("📱 Converting to STTResult - text: '${speechResult.text}', confidence: ${speechResult.confidence}")
        val result = STTResult(
            text = speechResult.text,
            confidence = speechResult.confidence,
            isPartial = speechResult.isPartial
        )
        
        println("Returning STTResult: $result")
        result
    } catch (e: Exception) {
        println("STT error: $e")
        e.printStackTrace()
        null
    }
}

actual suspend fun performFileSTT(audioPath: String, language: String): STTResult? {
    return try {
        val params = SpeechRecognitionParams(
            language = language,
            enablePartialResults = false,
            maxDuration = 60
        )
        val speechResult = recognizeSpeechFromFile(audioPath, params)
        speechResult?.let {
            STTResult(
                text = it.text,
                confidence = it.confidence,
                isPartial = it.isPartial
            )
        }
    } catch (e: Exception) {
        null
    }
}

actual fun stopSTT() {
    stopSpeechRecognition()
} 