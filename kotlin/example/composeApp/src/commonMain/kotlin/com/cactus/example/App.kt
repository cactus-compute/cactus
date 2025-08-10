package com.cactus.example

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.cactus.CactusAgent
import com.cactus.CactusLM
import com.cactus.CactusVLM
import com.cactus.CactusSTT

import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import com.cactus.Parameter

import org.jetbrains.compose.resources.ExperimentalResourceApi
import kmp_app_template.composeapp.generated.resources.Res

@Composable
fun App() {
    val scope = rememberCoroutineScope()
    var lm by remember { mutableStateOf(CactusLM()) }
    var vlm by remember { mutableStateOf(CactusVLM()) }
    val stt = remember { CactusSTT() }
    var agent by remember { mutableStateOf(CactusAgent()) }
    
    var logs by remember { mutableStateOf(listOf<String>()) }
    var currentGpuMode by remember { mutableStateOf("CPU") }
    
    fun addLog(message: String) {
        logs = logs + "${logs.size + 1}. $message"
    }
    
    LaunchedEffect(Unit) {
        addLog("App started - Cactus Modular Demo")
        addLog("Available: Language Model, Vision, Speech-to-Text, Text-to-Speech")
    }
    
    MaterialTheme {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp)
                    .windowInsetsPadding(WindowInsets.safeDrawing),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Cactus Modular Demo",
                    style = MaterialTheme.typography.headlineMedium,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                
                // Language Model Section
                Card(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Language Model - Mode: $currentGpuMode", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        // GPU Mode Selection
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Switching to CPU mode (GPU=0)...")
                                        lm.unload()
                                        lm = CactusLM(gpuLayers = 0)
                                        currentGpuMode = "CPU"
                                        addLog("Switched to CPU mode")
                                    }
                                }
                            ) { Text("CPU Mode") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Switching to GPU mode (GPU=99)...")
                                        lm.unload()
                                        lm = CactusLM(gpuLayers = 99)
                                        currentGpuMode = "GPU"
                                        addLog("Switched to GPU mode")
                                    }
                                }
                            ) { Text("GPU Mode") }
                        }
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        // Model Operations
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Downloading LM model...")
                                        val success = lm.download()
                                        addLog(if (success) "LM model downloaded" else "LM download failed")
                                    }
                                }
                            ) { Text("Download") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Loading LM model in $currentGpuMode mode...")
                                        val success = lm.init()
                                        addLog(if (success) "LM model loaded in $currentGpuMode mode" else "LM load failed")
                                    }
                                }
                            ) { Text("Load") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Generating text with $currentGpuMode...")
                                        val result = lm.completion("What is AI?", maxTokens = 50)
                                        addLog("LM ($currentGpuMode): ${result ?: "No response"}")
                                    }
                                }
                            ) { Text("Generate") }
                        }
                    }
                }
                
                // Vision Language Model Section
                Card(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Vision Language Model - Mode: $currentGpuMode", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        // GPU Mode Selection for VLM  
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Switching VLM to CPU mode (GPU=0)...")
                                        vlm.unload()
                                        vlm = CactusVLM(gpuLayers = 0)
                                        currentGpuMode = "CPU"
                                        addLog("VLM switched to CPU mode")
                                    }
                                }
                            ) { Text("VLM CPU") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Switching VLM to GPU mode (GPU=99)...")
                                        vlm.unload()
                                        vlm = CactusVLM(gpuLayers = 99)
                                        currentGpuMode = "GPU"
                                        addLog("VLM switched to GPU mode")
                                    }
                                }
                            ) { Text("VLM GPU") }
                        }
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Downloading VLM models...")
                                        val success = vlm.download()
                                        addLog(if (success) "VLM models downloaded" else "VLM download failed")
                                    }
                                }
                            ) { Text("Download") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Loading VLM model in $currentGpuMode mode...")
                                        val success = vlm.init()
                                        addLog(if (success) "VLM model loaded in $currentGpuMode mode" else "VLM load failed")
                                    }
                                }
                            ) { Text("Load") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Analyzing image with $currentGpuMode...")
                                        val result = withContext(Dispatchers.Default) {
                                            try {
                                                @OptIn(ExperimentalResourceApi::class)
                                                val imageBytes = Res.readBytes("files/image.jpg")
                                                val imagePath = saveImageToTempFile(imageBytes)
                                                if (imagePath != null) {
                                                    vlm.completion("Describe this", imagePath, maxTokens = 50)
                                                } else {
                                                    null
                                                }
                                            } catch (e: Exception) {
                                                null
                                            }
                                        }
                                        addLog("VLM ($currentGpuMode): ${result ?: "No response"}")
                                    }
                                }
                            ) { Text("Analyze") }
                        }
                    }
                }
                
                // Speech to Text Section
                Card(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Speech to Text", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Downloading STT model...")
                                        val downloadSuccess = stt.download()
                                        if (downloadSuccess) {
                                            addLog("Initializing STT...")
                                            val initSuccess = stt.init()
                                            addLog(if (initSuccess) "STT ready" else "STT init failed")
                                        } else {
                                            addLog("STT download failed")
                                        }
                                    }
                                }
                            ) { Text("Setup") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        if (!stt.isReady()) {
                                            addLog("Setting up STT first...")
                                            val downloadSuccess = stt.download()
                                            if (downloadSuccess) {
                                                val initSuccess = stt.init()
                                                if (!initSuccess) {
                                                    addLog("STT initialization failed")
                                                    return@launch
                                                }
                                            } else {
                                                addLog("STT download failed")
                                                return@launch
                                            }
                                        }
                                        
                                        addLog("Listening...")
                                        val result = stt.transcribe()
                                        addLog("STT: ${result?.text ?: "No speech detected"}")
                                    }
                                }
                            ) { Text("Listen") }

                            Button(
                                onClick = {
                                    scope.launch {
                                        if (!stt.isReady()) {
                                            addLog("Setting up STT first...")
                                            val downloadSuccess = stt.download()
                                            if (downloadSuccess) {
                                                val initSuccess = stt.init()
                                                if (!initSuccess) {
                                                    addLog("STT initialization failed")
                                                    return@launch
                                                }
                                            } else {
                                                addLog("STT download failed")
                                                return@launch
                                            }
                                        }

                                        addLog("Transcribing audio file...")
                                        val result = withContext(Dispatchers.Default) {
                                            try {
                                                @OptIn(ExperimentalResourceApi::class)
                                                val audioBytes = Res.readBytes("files/audio.wav")
                                                val audioPath = saveAudioToTempFile(audioBytes)
                                                if (audioPath != null) {
                                                    stt.transcribeFile(audioPath)
                                                } else {
                                                    null
                                                }
                                            } catch (e: Exception) {
                                                null
                                            }
                                        }
                                        addLog("STT: ${result?.text ?: "No speech detected"}")
                                    }
                                }
                            ) { Text("File") }
                        }
                    }
                }

                // Agent with Tools Section
                Card(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("AI Agent with Tools - Mode: $currentGpuMode", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))

                        // GPU Mode Selection
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Switching to CPU mode (GPU=0)...")
                                        agent.unload()
                                        agent = CactusAgent(gpuLayers = 0)
                                        currentGpuMode = "CPU"
                                        addLog("Switched to CPU mode")
                                    }
                                }
                            ) { Text("CPU Mode") }

                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Switching to GPU mode (GPU=99)...")
                                        agent.unload()
                                        agent = CactusAgent(gpuLayers = 99)
                                        currentGpuMode = "GPU"
                                        addLog("Switched to GPU mode")
                                    }
                                }
                            ) { Text("GPU Mode") }

                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Downloading Agent model...")
                                        val success =agent.download()
                                        addLog(if (success) "Agent model downloaded" else "Agent download failed")
                                    }
                                }
                            ) { Text("Download") }
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Setting up Agent with tools...")
                                        val success = agent.init()
                                        addLog(if (success) "Agent model loaded in $currentGpuMode mode" else "Agent load failed")
                                        agent.addTool(
                                            "get_weather",
                                            WeatherTool(),
                                            "Get current weather for a location",
                                            mapOf(
                                                "location" to Parameter(
                                                    "string",
                                                    "The city or location",
                                                    true
                                                )
                                            )
                                        )
                                        addLog("Agent created")
                                    }
                                }
                            ) { Text("Setup Tools") }

                            Button(
                                onClick = {
                                    scope.launch {
                                        agent.let { ag ->
                                            addLog("Agent: What's the weather in New York?")
                                            if (!ag.isLoaded()) {
                                                addLog("Agent not loaded, loading now...")
                                                val loaded = ag.init()
                                                addLog(if (loaded) "Agent loaded" else "Agent load failed")
                                            }
                                            val result = ag.completionWithTools(
                                                message = """
                                                    [
                                                        {"role": "user", "content": "What's the weather in New York?"}
                                                    ]
                                                """.trimIndent(),
                                                maxTokens = 200,
                                                temperature = 0.1f,
                                                callback = { token ->
                                                    // Optional: show real-time generation
                                                }
                                            )
                                            addLog("Agent response: ${result.content}")
                                            result.toolCalls?.let { calls ->
                                                addLog("Tools used: ${calls.joinToString(", ") { it }}")
                                            }
                                        }
                                    }
                                }
                            ) { Text("Weather") }
                        }
                    }
                }
                

                
                Spacer(modifier = Modifier.height(16.dp))
                HorizontalDivider()
                Spacer(modifier = Modifier.height(16.dp))
                
                Text("Logs:", style = MaterialTheme.typography.titleMedium)
                
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    items(logs) { log ->
                        Text(
                            text = log,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(4.dp)
                        )
                    }
                }
            }
        }
    }
}