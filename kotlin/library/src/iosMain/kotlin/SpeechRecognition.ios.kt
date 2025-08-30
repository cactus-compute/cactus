package com.cactus

import cnames.structs.VoskModel
import cnames.structs.VoskRecognizer
import cnames.structs.VoskSpkModel
import platform.Foundation.*
import platform.AVFAudio.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlinx.cinterop.*
import kotlinx.serialization.json.*
import com.vosk.native.*

// Vosk components - using COpaquePointer to match the C API
@OptIn(ExperimentalForeignApi::class)
private var voskModel: CValuesRef<VoskModel>? = null
@OptIn(ExperimentalForeignApi::class)
private var voskSpkModel: CValuesRef<VoskSpkModel>? = null
@OptIn(ExperimentalForeignApi::class)
private var voskRecognizer: CValuesRef<VoskRecognizer>? = null
private var audioEngine: AVAudioEngine? = null
private var isInitialized = false
private var isRecording = false

@OptIn(ExperimentalForeignApi::class)
actual suspend fun initializeSpeechRecognition(): Boolean = withContext(Dispatchers.Main) {
    return@withContext try {
        // Set log level (similar to Swift VoskModel init)
        vosk_set_log_level(0)
        
        // Initialize Vosk model paths from the main app bundle
        val bundlePath = NSBundle.mainBundle.resourcePath
        if (bundlePath != null) {
            val modelPath = "$bundlePath/vosk-model-small-en-us-0.15"
            val spkModelPath = "$bundlePath/vosk-model-spk-0.4"
            
            println("Looking for Vosk models at: $bundlePath")
            println("Model path: $modelPath")
            println("Speaker model path: $spkModelPath")
            
            // Check if model directories exist
            val fileManager = NSFileManager.defaultManager
            val modelExists = fileManager.fileExistsAtPath(modelPath)
            val spkModelExists = fileManager.fileExistsAtPath(spkModelPath)
            
            println("Model exists: $modelExists")
            println("Speaker model exists: $spkModelExists")
            
            if (!modelExists) {
                println("ERROR: Model directory not found at $modelPath")
                return@withContext false
            }
            
            voskModel = vosk_model_new(modelPath)
            voskSpkModel = vosk_spk_model_new(spkModelPath)
            
            if (voskModel != null) {
                println("Vosk model loaded successfully")
                
                // Initialize audio engine
                audioEngine = AVAudioEngine()
                
                val audioSession = AVAudioSession.sharedInstance()
                audioSession.setCategory(AVAudioSessionCategoryRecord, null)
                audioSession.setActive(true, null)
                
                isInitialized = true
                println("Vosk speech recognition initialized successfully")
                true
            } else {
                println("Failed to load Vosk model")
                false
            }
        } else {
            println("Could not get bundle resource path")
            false
        }
    } catch (e: Exception) {
        println("Failed to initialize Vosk speech recognition: $e")
        false
    }
}

actual suspend fun requestSpeechPermissions(): Boolean = suspendCancellableCoroutine { continuation ->
    // For Vosk, we only need microphone permission, no speech framework permission needed
    val audioSession = AVAudioSession.sharedInstance()
    audioSession.requestRecordPermission { granted ->
        continuation.resume(granted)
    }
}

@OptIn(ExperimentalForeignApi::class)
actual suspend fun performSpeechRecognition(params: SpeechRecognitionParams): SpeechRecognitionResult? = 
    suspendCancellableCoroutine { continuation ->
        
    if (!isInitialized || voskModel == null || audioEngine == null) {
        println("Vosk speech recognition not initialized")
        continuation.resume(null)
        return@suspendCancellableCoroutine
    }
    
    if (isRecording) {
        println("Already recording")
        continuation.resume(null)
        return@suspendCancellableCoroutine
    }
    
    try {
        stopCurrentRecognition()
        
        // Create recognizer with speaker model (similar to Swift Vosk init)
        voskRecognizer = vosk_recognizer_new_spk(voskModel, 16000.0f, voskSpkModel)
        
        if (voskRecognizer == null) {
            continuation.resume(null)
            return@suspendCancellableCoroutine
        }
        
        println("Starting Vosk speech recognition")
        
        val inputNode = audioEngine!!.inputNode
        val recordingFormat = inputNode.outputFormatForBus(0u)
        
        // Debug audio format
        println("🎤 Audio format - Sample rate: ${recordingFormat.sampleRate}, Channels: ${recordingFormat.channelCount}")
        println("🎤 Audio format - Bits per sample: ${recordingFormat.streamDescription?.pointed?.mBitsPerChannel ?: 0}")
        
        var finalResult: String? = null
        var hasFinalResult = false
        var timeoutTimer: NSTimer? = null
        var silenceTimer: NSTimer? = null
        var hasDetectedSpeech = false
        
        inputNode.installTapOnBus(
            bus = 0u,
            bufferSize = 4800u, // Larger buffer for better processing
            format = recordingFormat
        ) { buffer, _ ->
            buffer?.let { audioBuffer ->
                try {
                    // Debug buffer info
                    val frameLength = audioBuffer.frameLength
                    val channelCount = audioBuffer.format.channelCount
                    println("🎤 Buffer: ${frameLength} frames, ${channelCount} channels")
                    
                    if (frameLength > 0u) {
                        // For now, let's try to process the audio at the original sample rate
                        // and let Vosk handle the mismatch, or downsample manually
                        val floatChannelData = audioBuffer.floatChannelData
                        if (floatChannelData != null) {
                            val originalSampleRate = audioBuffer.format.sampleRate
                            val targetSampleRate = 16000.0
                            
                            // Simple downsampling: take every Nth sample
                            val downsampleRatio = (originalSampleRate / targetSampleRate).toInt()
                            val outputSamples = (frameLength.toInt() / downsampleRatio)
                            
                            println("🎤 Downsampling from ${originalSampleRate}Hz to ${targetSampleRate}Hz (ratio: ${downsampleRatio})")
                            println("🎤 Input samples: ${frameLength}, Output samples: ${outputSamples}")
                            
                            if (outputSamples > 0) {
                                // Create buffer for downsampled 16-bit data
                                val int16Buffer = ByteArray(outputSamples * 2) // 2 bytes per 16-bit sample
                                
                                // Get the first (left) channel data
                                val channelData = floatChannelData[0]
                                
                                // Downsample and convert to 16-bit
                                for (i in 0 until outputSamples) {
                                    val inputIndex = i * downsampleRatio
                                    if (inputIndex < frameLength.toInt()) {
                                        // Get float sample (-1.0 to 1.0) and convert to 16-bit int (-32768 to 32767)
                                        val floatSample = channelData!![inputIndex]
                                        val int16Sample = (floatSample * 32767.0f).toInt().coerceIn(-32768, 32767).toShort()
                                        
                                        // Store as little-endian bytes
                                        val byteIndex = i * 2
                                        int16Buffer[byteIndex] = (int16Sample.toInt() and 0xFF).toByte()
                                        int16Buffer[byteIndex + 1] = ((int16Sample.toInt() shr 8) and 0xFF).toByte()
                                    }
                                }
                                
                                val dataLength = int16Buffer.size
                                println("🎤 Sending ${dataLength} bytes to Vosk")
                                
                                // Send to Vosk using the downsampled data
                                int16Buffer.usePinned { pinned ->
                                    // Convert byte data to short pointer for vosk_recognizer_accept_waveform_s
                                    val shortPtr = pinned.addressOf(0).reinterpret<ShortVar>()
                                    val sampleCount = dataLength / 2 // Convert bytes to sample count
                                    
                                    val endOfSpeech = vosk_recognizer_accept_waveform_s(
                                        recognizer = voskRecognizer!!, 
                                        data = shortPtr, 
                                        length = sampleCount
                                    )
                                    
                                    val resultJson = if (endOfSpeech == 1) {
                                        // End of speech detected
                                        vosk_recognizer_result(voskRecognizer)
                                    } else {
                                        // Partial result
                                        vosk_recognizer_partial_result(voskRecognizer)
                                    }
                                    
                                    resultJson?.let { jsonCString ->
                                        val jsonString = jsonCString.toKString()
                                        println("🎤 Vosk result: $jsonString")
                                        
                                        try {
                                            val json = Json.parseToJsonElement(jsonString).jsonObject
                                            
                                            if (endOfSpeech == 1) {
                                                // Final result
                                                val text = json["text"]?.jsonPrimitive?.content ?: ""
                                                if (text.isNotBlank() && !hasFinalResult) {
                                                    finalResult = text
                                                    hasDetectedSpeech = true
                                                    hasFinalResult = true
                                                    stopCurrentRecognition()
                                                    timeoutTimer?.invalidate()
                                                    silenceTimer?.invalidate()
                                                    
                                                    val speechResult = SpeechRecognitionResult(
                                                        text = text,
                                                        confidence = 0.9f,
                                                        isPartial = false,
                                                        alternatives = emptyList()
                                                    )
                                                    println("Vosk final result: $speechResult")
                                                    continuation.resume(speechResult)
                                                }
                                            } else {
                                                // Partial result
                                                val partialText = json["partial"]?.jsonPrimitive?.content ?: ""
                                                if (partialText.isNotBlank()) {
                                                    hasDetectedSpeech = true
                                                    println("Vosk partial result: $partialText")
                                                    
                                                    // Reset silence timer when we get speech
                                                    silenceTimer?.invalidate()
                                                    silenceTimer = NSTimer.scheduledTimerWithTimeInterval(
                                                        interval = 2.0, // 2 second silence timeout
                                                        repeats = false
                                                    ) { _ ->
                                                        if (!hasFinalResult && hasDetectedSpeech) {
                                                            println("Silence detected, getting final result...")
                                                            val finalJson = vosk_recognizer_final_result(voskRecognizer)
                                                            finalJson?.let { finalCString ->
                                                                val finalJsonString = finalCString.toKString()
                                                                try {
                                                                    val finalJsonObj = Json.parseToJsonElement(finalJsonString).jsonObject
                                                                    val finalText = finalJsonObj["text"]?.jsonPrimitive?.content ?: partialText
                                                                    
                                                                    hasFinalResult = true
                                                                    stopCurrentRecognition()
                                                                    timeoutTimer?.invalidate()
                                                                    
                                                                    val speechResult = SpeechRecognitionResult(
                                                                        text = finalText,
                                                                        confidence = 0.8f,
                                                                        isPartial = false,
                                                                        alternatives = emptyList()
                                                                    )
                                                                    println("Vosk silence-triggered result: $speechResult")
                                                                    continuation.resume(speechResult)
                                                                } catch (e: Exception) {
                                                                    println("Error parsing final result: $e")
                                                                    continuation.resume(SpeechRecognitionResult(
                                                                        text = partialText,
                                                                        confidence = 0.8f,
                                                                        isPartial = false,
                                                                        alternatives = emptyList()
                                                                    ))
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        } catch (e: Exception) {
                                            println("Error parsing Vosk result JSON: $e")
                                        }
                                    }
                                }
                            } else {
                                println("❌ No samples after downsampling")
                            }
                        } else {
                            println("❌ No float channel data in buffer")
                        }
                    }
                } catch (e: Exception) {
                    println("❌ Error processing audio buffer: $e")
                }
            }
        }
        
        audioEngine!!.prepare()
        audioEngine!!.startAndReturnError(null)
        isRecording = true
        
        println("🎙️ Started Vosk listening... (speak now)")
        
        // Overall timeout timer
        timeoutTimer = NSTimer.scheduledTimerWithTimeInterval(
            interval = params.maxDuration.toDouble(),
            repeats = false
        ) { _ ->
            if (!hasFinalResult) {
                if (hasDetectedSpeech) {
                    println("Timeout reached, getting final result")
                    val finalJson = vosk_recognizer_final_result(voskRecognizer)
                    val finalText = finalJson?.toKString()?.let { jsonString ->
                        try {
                            val json = Json.parseToJsonElement(jsonString).jsonObject
                            json["text"]?.jsonPrimitive?.content ?: ""
                        } catch (e: Exception) {
                            ""
                        }
                    } ?: ""
                    
                    stopCurrentRecognition()
                    silenceTimer?.invalidate()
                    val timeoutResult = SpeechRecognitionResult(
                        text = finalText,
                        confidence = 0.7f,
                        isPartial = false,
                        alternatives = emptyList()
                    )
                    println("Vosk timeout result: $timeoutResult")
                    continuation.resume(timeoutResult)
                } else {
                    println("Timeout reached, no speech detected")
                    stopCurrentRecognition()
                    silenceTimer?.invalidate()
                    val noSpeechResult = SpeechRecognitionResult(
                        text = "",
                        confidence = 0.0f,
                        isPartial = false,
                        alternatives = emptyList()
                    )
                    println("Vosk no speech result: $noSpeechResult")
                    continuation.resume(noSpeechResult)
                }
            }
        }
        
        // Initial silence timer
        silenceTimer = NSTimer.scheduledTimerWithTimeInterval(
            interval = 5.0, 
            repeats = false
        ) { _ ->
            if (!hasFinalResult && !hasDetectedSpeech) {
                println("No speech detected in 5 seconds, stopping...")
                hasFinalResult = true
                stopCurrentRecognition()
                timeoutTimer?.invalidate()
                val initialSilenceResult = SpeechRecognitionResult(
                    text = "",
                    confidence = 0.0f,
                    isPartial = false,
                    alternatives = emptyList()
                )
                println("Vosk initial silence result: $initialSilenceResult")
                continuation.resume(initialSilenceResult)
            }
        }
        
        continuation.invokeOnCancellation {
            timeoutTimer?.invalidate()
            silenceTimer?.invalidate()
            stopCurrentRecognition()
        }
        
    } catch (e: Exception) {
        println("Failed to start Vosk speech recognition: $e")
        stopCurrentRecognition()
        continuation.resume(null)
    }
}

@OptIn(ExperimentalForeignApi::class)
actual suspend fun recognizeSpeechFromFile(audioFilePath: String, params: SpeechRecognitionParams): SpeechRecognitionResult? = 
    withContext(Dispatchers.Default) {
        
    if (!isInitialized || voskModel == null) {
        return@withContext null
    }
        
    // TODO: Implement file recognition - for now return empty result
    return@withContext SpeechRecognitionResult(
        text = "",
        confidence = 0.0f,
        isPartial = false,
        alternatives = emptyList()
    )
}

@OptIn(ExperimentalForeignApi::class)
private fun stopCurrentRecognition() {
    if (isRecording) {
        println("Stopping Vosk speech recognition...")
        audioEngine?.stop()
        audioEngine?.inputNode?.removeTapOnBus(0u)
        isRecording = false
    }
    
    voskRecognizer?.let {
        vosk_recognizer_free(it)
        voskRecognizer = null
    }
}

actual fun stopSpeechRecognition() {
    stopCurrentRecognition()
}

@OptIn(ExperimentalForeignApi::class)
actual fun isSpeechRecognitionAvailable(): Boolean {
    return isInitialized && voskModel != null
}

actual fun isSpeechRecognitionAuthorized(): Boolean {
    val audioSession = AVAudioSession.sharedInstance()
    return audioSession.recordPermission() == AVAudioSessionRecordPermissionGranted
}

@OptIn(ExperimentalForeignApi::class)
fun isVoskModelLoaded(): Boolean {
    return voskModel != null
}

// Test function to verify model files exist in bundle
fun testVoskModelPaths(): String {
    val bundlePath = NSBundle.mainBundle.resourcePath
    if (bundlePath != null) {
        val modelPath = "$bundlePath/vosk-model-small-en-us-0.15"
        val spkModelPath = "$bundlePath/vosk-model-spk-0.4"
        val testAudioPath = "$bundlePath/10001-90210-01803.wav"
        
        val modelExists = NSFileManager.defaultManager.fileExistsAtPath(modelPath)
        val spkModelExists = NSFileManager.defaultManager.fileExistsAtPath(spkModelPath)
        val testAudioExists = NSFileManager.defaultManager.fileExistsAtPath(testAudioPath)
        
        return "Bundle path: $bundlePath\n" +
               "Model exists: $modelExists at $modelPath\n" +
               "Speaker model exists: $spkModelExists at $spkModelPath\n" +
               "Test audio exists: $testAudioExists at $testAudioPath"
    } else {
        return "Could not get bundle resource path"
    }
} 