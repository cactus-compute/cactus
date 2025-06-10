# Cactus React Native

A powerful React Native library for running Large Language Models (LLMs) and Vision Language Models (VLMs) directly on mobile devices, with full support for chat completions, multimodal inputs, embeddings, text-to-speech and advanced features.

## Installation

```bash
npm install cactus-react-native react-native-fs
# or
yarn add cactus-react-native react-native-fs
```

**Additional Setup:**
- For iOS: `cd ios && npx pod-install` or `yarn pod-install`
- For Android: Ensure your `minSdkVersion` is 24 or higher

> **Important**: `react-native-fs` is required for file system access to download and manage model files locally.

## Quick Start

### Basic Text Completion

```typescript
import { initLlama } from 'cactus-react-native';

// Initialize a context
const context = await initLlama({
  model: '/path/to/your/model.gguf',
  n_ctx: 2048,
  n_threads: 4,
});

// Generate text
const result = await context.completion({
  messages: [
    { role: 'user', content: 'Hello, how are you?' }
  ],
  n_predict: 100,
  temperature: 0.7,
});

console.log(result.text);
```

### Complete Chat App Example

```typescript
import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity } from 'react-native';
import { initLlama, LlamaContext } from 'cactus-react-native';
import RNFS from 'react-native-fs';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function ChatApp() {
  const [context, setContext] = useState<LlamaContext | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    initializeModel();
  }, []);

  async function initializeModel() {
    try {
      // Download model (example URL)
      const modelUrl = 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf';
      const modelPath = `${RNFS.DocumentDirectoryPath}/model.gguf`;
      
      // Download if not exists
      if (!(await RNFS.exists(modelPath))) {
        await RNFS.downloadFile({
          fromUrl: modelUrl,
          toFile: modelPath,
        }).promise;
      }

      // Initialize context
      const llamaContext = await initLlama({
        model: modelPath,
        n_ctx: 2048,
        n_threads: 4,
        n_gpu_layers: 99, // Use GPU acceleration
      });

      setContext(llamaContext);
      setLoading(false);
    } catch (error) {
      console.error('Failed to initialize model:', error);
    }
  }

  async function sendMessage() {
    if (!context || !input.trim()) return;

    const userMessage: Message = { role: 'user', content: input };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');

    try {
      const result = await context.completion({
        messages: newMessages,
        n_predict: 256,
        temperature: 0.7,
        stop: ['</s>', '<|end|>'],
      });

      const assistantMessage: Message = { 
        role: 'assistant', 
        content: result.text 
      };
      setMessages([...newMessages, assistantMessage]);
    } catch (error) {
      console.error('Completion failed:', error);
    }
  }

  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <Text>Loading model...</Text>
      </View>
    );
  }

  return (
    <View style={{ flex: 1, padding: 16 }}>
      {/* Messages */}
      <View style={{ flex: 1 }}>
        {messages.map((msg, index) => (
          <Text key={index} style={{ 
            backgroundColor: msg.role === 'user' ? '#007AFF' : '#f0f0f0',
            color: msg.role === 'user' ? 'white' : 'black',
            padding: 8,
            margin: 4,
            borderRadius: 8,
          }}>
            {msg.content}
          </Text>
        ))}
      </View>

      {/* Input */}
      <View style={{ flexDirection: 'row' }}>
        <TextInput
          style={{ flex: 1, borderWidth: 1, padding: 8, borderRadius: 4 }}
          value={input}
          onChangeText={setInput}
          placeholder="Type a message..."
        />
        <TouchableOpacity 
          onPress={sendMessage}
          style={{ backgroundColor: '#007AFF', padding: 8, borderRadius: 4, marginLeft: 8 }}
        >
          <Text style={{ color: 'white' }}>Send</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}
```

## File Path Requirements

**Critical**: Cactus requires **absolute local file paths**, not Metro bundler URLs or asset references.

### ❌ Won't Work
```typescript
// Metro bundler URLs
'http://localhost:8081/assets/model.gguf'

// React Native asset requires  
require('./assets/model.gguf')

// Relative paths
'./models/model.gguf'
```

### ✅ Will Work
```typescript
import RNFS from 'react-native-fs';

// Absolute paths in app directories
const modelPath = `${RNFS.DocumentDirectoryPath}/model.gguf`;
const imagePath = `${RNFS.DocumentDirectoryPath}/image.jpg`;

// Downloaded/copied files
const downloadModel = async () => {
  const modelUrl = 'https://example.com/model.gguf';
  const localPath = `${RNFS.DocumentDirectoryPath}/model.gguf`;
  
  await RNFS.downloadFile({
    fromUrl: modelUrl,
    toFile: localPath,
  }).promise;
  
  return localPath; // Use this path with Cactus
};
```

### Image Assets
For images, you need to copy them to local storage first:

```typescript
// Copy bundled asset to local storage
const copyAssetToLocal = async (assetName: string): Promise<string> => {
  const assetPath = `${RNFS.MainBundlePath}/${assetName}`;
  const localPath = `${RNFS.DocumentDirectoryPath}/${assetName}`;
  
  if (!(await RNFS.exists(localPath))) {
    await RNFS.copyFile(assetPath, localPath);
  }
  
  return localPath;
};

// Usage
const imagePath = await copyAssetToLocal('demo.jpg');
const result = await multimodalCompletion(contextId, prompt, [imagePath], params);
```

### External Images
Download external images to local storage:

```typescript
const downloadImage = async (imageUrl: string): Promise<string> => {
  const localPath = `${RNFS.DocumentDirectoryPath}/temp_image.jpg`;
  
  await RNFS.downloadFile({
    fromUrl: imageUrl,
    toFile: localPath,
  }).promise;
  
  return localPath;
};
```

## Core Concepts

### Context Initialization

The `LlamaContext` is the main interface for interacting with models:

```typescript
import { initLlama, ContextParams } from 'cactus-react-native';

const params: ContextParams = {
  model: '/path/to/model.gguf',
  n_ctx: 4096,           // Context window size
  n_batch: 512,          // Batch size for processing
  n_threads: 4,          // Number of threads
  n_gpu_layers: 99,      // GPU layers (0 = CPU only)
  temperature: 0.7,      // Default temperature
  use_mlock: true,       // Lock model in memory
  use_mmap: true,        // Use memory mapping
};

const context = await initLlama(params);
```

### Messages Format

Cactus uses OpenAI-compatible message format:

```typescript
interface CactusOAICompatibleMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What is the capital of France?' },
  { role: 'assistant', content: 'The capital of France is Paris.' },
  { role: 'user', content: 'What about Germany?' },
];
```

## Text Completion

### Basic Completion

```typescript
const result = await context.completion({
  messages: [
    { role: 'user', content: 'Write a short poem about coding' }
  ],
  n_predict: 200,
  temperature: 0.8,
  top_p: 0.9,
  stop: ['</s>', '\n\n'],
});

console.log(result.text);
console.log(`Tokens: ${result.tokens_predicted}`);
console.log(`Speed: ${result.timings.predicted_per_second.toFixed(2)} tokens/sec`);
```

### Streaming Completion

```typescript
const result = await context.completion({
  messages: [{ role: 'user', content: 'Tell me a story' }],
  n_predict: 500,
  temperature: 0.7,
}, (token) => {
  // Called for each generated token
  console.log('Token:', token.token);
  updateUI(token.token);
});
```

### Advanced Parameters

```typescript
const result = await context.completion({
  messages: [...],
  
  // Generation control
  n_predict: 256,        // Max tokens to generate
  temperature: 0.7,      // Randomness (0.0 - 2.0)
  top_p: 0.9,           // Nucleus sampling
  top_k: 40,            // Top-k sampling
  min_p: 0.05,          // Minimum probability
  
  // Repetition control
  penalty_repeat: 1.1,   // Repetition penalty
  penalty_freq: 0.0,     // Frequency penalty
  penalty_present: 0.0,  // Presence penalty
  
  // Stop conditions
  stop: ['</s>', '<|end|>', '\n\n'],
  ignore_eos: false,
  
  // Sampling methods
  mirostat: 0,          // Mirostat sampling (0=disabled)
  mirostat_tau: 5.0,    // Target entropy
  mirostat_eta: 0.1,    // Learning rate
  
  // Advanced
  seed: -1,             // Random seed (-1 = random)
  n_probs: 0,           // Return token probabilities
});
```

## Multimodal (Vision)

### Setup Vision Model

```typescript
import { initLlama, initMultimodal, multimodalCompletion } from 'cactus-react-native';

// 1. Initialize base model
const context = await initLlama({
  model: '/path/to/vision-model.gguf',
  n_ctx: 4096,
});

// 2. Initialize multimodal capabilities
const success = await initMultimodal(
  context.id, 
  '/path/to/mmproj.gguf',  // Multimodal projector
  false                     // Use GPU for projector
);

if (!success) {
  throw new Error('Failed to initialize multimodal capabilities');
}
```

### Image Analysis

```typescript
// Analyze single image
const result = await multimodalCompletion(
  context.id,
  'Describe this image in detail',
  ['/path/to/image.jpg'],
  {
    prompt: 'Describe this image in detail',
    n_predict: 200,
    temperature: 0.3,
    emit_partial_completion: false,
  }
);

console.log(result.text);
```

### Multi-Image Analysis

```typescript
const imagePaths = [
  '/path/to/image1.jpg',
  '/path/to/image2.jpg',
  '/path/to/image3.jpg'
];

const result = await multimodalCompletion(
  context.id,
  'Compare these images and explain the differences',
  imagePaths,
  {
    prompt: 'Compare these images and explain the differences',
    n_predict: 300,
    temperature: 0.4,
    emit_partial_completion: false,
  }
);
```

### Conversation with Images

```typescript
// Build conversation context with proper chat formatting
const conversation = [
  { role: 'user', content: 'What do you see in this image?' }
];

const formattedPrompt = await context.getFormattedChat(conversation);
const promptString = typeof formattedPrompt === 'string' 
  ? formattedPrompt 
  : formattedPrompt.prompt;

const result = await multimodalCompletion(
  context.id,
  promptString,
  ['/path/to/image.jpg'],
  {
    prompt: promptString,
    n_predict: 256,
    temperature: 0.3,
    emit_partial_completion: false,
  }
);
```

## Embeddings

### Text Embeddings

```typescript
// Enable embeddings during initialization
const context = await initLlama({
  model: '/path/to/embedding-model.gguf',
  embedding: true,        // Enable embedding mode
  n_ctx: 512,            // Smaller context for embeddings
});

// Generate embeddings
const result = await context.embedding('Your text here');
console.log('Embedding vector:', result.embedding);
console.log('Dimensions:', result.embedding.length);
```

### Batch Embeddings

```typescript
const texts = [
  'The quick brown fox',
  'Machine learning is fascinating',
  'React Native development'
];

const embeddings = await Promise.all(
  texts.map(text => context.embedding(text))
);

// Calculate similarity
function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

const similarity = cosineSimilarity(
  embeddings[0].embedding,
  embeddings[1].embedding
);
```

## Text-to-Speech (TTS)

Cactus supports text-to-speech through vocoder models, allowing you to generate speech from text.

### Setup TTS Model

```typescript
import { initLlama, initVocoder, isVocoderEnabled } from 'cactus-react-native';

// 1. Initialize base model (text generation)
const context = await initLlama({
  model: '/path/to/text-model.gguf',
  n_ctx: 2048,
});

// 2. Initialize vocoder for TTS
const success = await initVocoder(
  context.id,
  '/path/to/vocoder-model.gguf'
);

if (!success) {
  throw new Error('Failed to initialize vocoder');
}

// 3. Verify TTS is enabled
const ttsEnabled = await isVocoderEnabled(context.id);
console.log('TTS enabled:', ttsEnabled);
```

### Basic Text-to-Speech

```typescript
import { getFormattedAudioCompletion, getTTSType } from 'cactus-react-native';

// Get TTS type information
const ttsType = await getTTSType(context.id);
console.log('TTS Type:', ttsType);

// Generate speech from text
const speakerConfig = JSON.stringify({
  voice_id: 0,        // Speaker voice ID
  temperature: 0.7,   // Speech variation
  speed: 1.0,         // Speech speed
});

const result = await getFormattedAudioCompletion(
  context.id,
  speakerConfig,
  'Hello, this is a test of text-to-speech functionality.'
);

console.log('Audio data:', result.audio_data);
console.log('Sample rate:', result.sample_rate);
console.log('Audio format:', result.format);
```

### Advanced TTS with Token Control

```typescript
import { 
  getAudioCompletionGuideTokens, 
  decodeAudioTokens 
} from 'cactus-react-native';

// Get guide tokens for precise control
const tokensResult = await getAudioCompletionGuideTokens(
  context.id,
  'This text will be converted to speech tokens.'
);

console.log('Guide tokens:', tokensResult.tokens);
console.log('Token count:', tokensResult.tokens.length);

// Decode tokens to audio
const audioResult = await decodeAudioTokens(
  context.id,
  tokensResult.tokens
);

console.log('Decoded audio:', audioResult.audio_data);
console.log('Duration:', audioResult.duration_seconds);
```

### Complete TTS Example

```typescript
import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, Alert } from 'react-native';
import { Audio } from 'expo-av';
import RNFS from 'react-native-fs';
import { 
  initLlama, 
  initVocoder, 
  getFormattedAudioCompletion,
  LlamaContext 
} from 'cactus-react-native';

export default function TTSDemo() {
  const [context, setContext] = useState<LlamaContext | null>(null);
  const [text, setText] = useState('Hello, this is a test of speech synthesis.');
  const [isGenerating, setIsGenerating] = useState(false);
  const [sound, setSound] = useState<Audio.Sound | null>(null);

  useEffect(() => {
    initializeTTS();
    return () => {
      if (sound) {
        sound.unloadAsync();
      }
    };
  }, []);

  async function initializeTTS() {
    try {
      // Download and initialize models
      const modelPath = await downloadModel();
      const vocoderPath = await downloadVocoder();

      const ctx = await initLlama({
        model: modelPath,
        n_ctx: 1024,
        n_threads: 4,
      });

      await initVocoder(ctx.id, vocoderPath);
      setContext(ctx);
    } catch (error) {
      console.error('Failed to initialize TTS:', error);
      Alert.alert('Error', 'Failed to initialize TTS');
    }
  }

  async function generateSpeech() {
    if (!context || !text.trim()) return;

    setIsGenerating(true);
    try {
      const speakerConfig = JSON.stringify({
        voice_id: 0,
        temperature: 0.7,
        speed: 1.0,
      });

      const result = await getFormattedAudioCompletion(
        context.id,
        speakerConfig,
        text
      );

      // Save audio to file
      const audioPath = `${RNFS.DocumentDirectoryPath}/speech.wav`;
      await RNFS.writeFile(audioPath, result.audio_data, 'base64');

      // Play audio
      const { sound: audioSound } = await Audio.Sound.createAsync({
        uri: `file://${audioPath}`,
      });
      
      setSound(audioSound);
      await audioSound.playAsync();

      console.log(`Generated speech: ${result.duration_seconds}s`);
    } catch (error) {
      console.error('Speech generation failed:', error);
      Alert.alert('Error', 'Failed to generate speech');
    } finally {
      setIsGenerating(false);
    }
  }

  async function downloadModel(): Promise<string> {
    // Your model download logic here
    const modelUrl = 'https://example.com/tts-model.gguf';
    const modelPath = `${RNFS.DocumentDirectoryPath}/tts-model.gguf`;
    
    if (!(await RNFS.exists(modelPath))) {
      await RNFS.downloadFile({
        fromUrl: modelUrl,
        toFile: modelPath,
      }).promise;
    }
    
    return modelPath;
  }

  async function downloadVocoder(): Promise<string> {
    // Your vocoder download logic here
    const vocoderUrl = 'https://example.com/vocoder.gguf';
    const vocoderPath = `${RNFS.DocumentDirectoryPath}/vocoder.gguf`;
    
    if (!(await RNFS.exists(vocoderPath))) {
      await RNFS.downloadFile({
        fromUrl: vocoderUrl,
        toFile: vocoderPath,
      }).promise;
    }
    
    return vocoderPath;
  }

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 18, marginBottom: 16 }}>
        Text-to-Speech Demo
      </Text>
      
      <TextInput
        style={{
          borderWidth: 1,
          borderColor: '#ddd',
          borderRadius: 8,
          padding: 12,
          marginBottom: 16,
          minHeight: 100,
        }}
        value={text}
        onChangeText={setText}
        placeholder="Enter text to convert to speech..."
        multiline
      />
      
      <TouchableOpacity
        onPress={generateSpeech}
        disabled={isGenerating || !context}
        style={{
          backgroundColor: isGenerating ? '#ccc' : '#007AFF',
          padding: 16,
          borderRadius: 8,
          alignItems: 'center',
        }}
      >
        <Text style={{ color: 'white', fontSize: 16, fontWeight: 'bold' }}>
          {isGenerating ? 'Generating...' : 'Generate Speech'}
        </Text>
      </TouchableOpacity>
    </View>
  );
}
```

### TTS with Real-time Generation

Generate speech as you type text:

```typescript
const generateRealtimeSpeech = async (inputText: string) => {
  if (!context || inputText.length < 10) return;

  // Generate speech for complete sentences
  const sentences = inputText.split(/[.!?]+/).filter(s => s.trim());
  const lastSentence = sentences[sentences.length - 1];
  
  if (lastSentence && lastSentence.trim().length > 20) {
    const result = await getFormattedAudioCompletion(
      context.id,
      JSON.stringify({ voice_id: 0, speed: 1.0 }),
      lastSentence.trim()
    );
    
    // Play the generated audio
    playAudio(result.audio_data);
  }
};
```

### Cleanup

```typescript
// Clean up TTS resources
const cleanup = async () => {
  if (context) {
    await releaseVocoder(context.id);
    await context.release();
  }
};
```

## Advanced Features

### Session Management

```typescript
// Save session
const tokensKept = await context.saveSession('/path/to/session.bin', {
  tokenSize: 1024  // Number of tokens to keep
});

// Load session
const sessionInfo = await context.loadSession('/path/to/session.bin');
console.log(`Loaded ${sessionInfo.tokens_loaded} tokens`);
console.log('Original prompt:', sessionInfo.prompt);
```

### LoRA Adapters

```typescript
// Apply LoRA adapters
await context.applyLoraAdapters([
  { path: '/path/to/lora1.gguf', scaled: 1.0 },
  { path: '/path/to/lora2.gguf', scaled: 0.8 }
]);

// Get loaded adapters
const adapters = await context.getLoadedLoraAdapters();
console.log('Loaded adapters:', adapters);

// Remove adapters
await context.removeLoraAdapters();
```

### Custom Chat Templates

```typescript
const context = await initLlama({
  model: '/path/to/model.gguf',
  chat_template: `{%- for message in messages %}
{%- if message.role == 'user' %}
User: {{ message.content }}
{%- elif message.role == 'assistant' %}
Assistant: {{ message.content }}
{%- endif %}
{%- endfor %}
Assistant:`,
});
```

### Structured Output (JSON)

```typescript
const result = await context.completion({
  messages: [
    { role: 'user', content: 'Extract information about this person: John Doe, 30 years old, software engineer from San Francisco' }
  ],
  response_format: {
    type: 'json_object',
    schema: {
      type: 'object',
      properties: {
        name: { type: 'string' },
        age: { type: 'number' },
        profession: { type: 'string' },
        location: { type: 'string' }
      },
      required: ['name', 'age']
    }
  }
});

const person = JSON.parse(result.text);
console.log(person.name); // "John Doe"
```

### Performance Monitoring

```typescript
const result = await context.completion({
  messages: [{ role: 'user', content: 'Hello' }],
  n_predict: 100,
});

console.log('Performance metrics:');
console.log(`Prompt tokens: ${result.timings.prompt_n}`);
console.log(`Generated tokens: ${result.timings.predicted_n}`);
console.log(`Prompt speed: ${result.timings.prompt_per_second.toFixed(2)} tokens/sec`);
console.log(`Generation speed: ${result.timings.predicted_per_second.toFixed(2)} tokens/sec`);
console.log(`Total time: ${(result.timings.prompt_ms + result.timings.predicted_ms).toFixed(0)}ms`);
```

## Best Practices

### Model Management

```typescript
class ModelManager {
  private contexts = new Map<string, LlamaContext>();

  async loadModel(name: string, params: ContextParams): Promise<LlamaContext> {
    if (this.contexts.has(name)) {
      return this.contexts.get(name)!;
    }

    const context = await initLlama(params);
    this.contexts.set(name, context);
    return context;
  }

  async unloadModel(name: string): Promise<void> {
    const context = this.contexts.get(name);
    if (context) {
      await context.release();
      this.contexts.delete(name);
    }
  }

  async unloadAll(): Promise<void> {
    await Promise.all(
      Array.from(this.contexts.values()).map(ctx => ctx.release())
    );
    this.contexts.clear();
  }
}
```

### Error Handling

```typescript
async function safeCompletion(context: LlamaContext, messages: any[]) {
  try {
    const result = await context.completion({
      messages,
      n_predict: 256,
      temperature: 0.7,
    });
    return { success: true, data: result };
  } catch (error) {
    if (error.message.includes('Context is busy')) {
      // Handle concurrent requests
      await new Promise(resolve => setTimeout(resolve, 100));
      return safeCompletion(context, messages);
    } else if (error.message.includes('Context not found')) {
      // Handle context cleanup
      throw new Error('Model context was released');
    } else {
      // Handle other errors
      console.error('Completion failed:', error);
      return { success: false, error: error.message };
    }
  }
}
```

### Memory Management

```typescript
// Monitor memory usage
const checkMemory = () => {
  if (Platform.OS === 'android') {
    // Android-specific memory monitoring
    console.log('Memory warning - consider releasing unused contexts');
  }
};

// Release contexts when app goes to background
import { AppState } from 'react-native';

AppState.addEventListener('change', (nextAppState) => {
  if (nextAppState === 'background') {
    // Release non-essential contexts
    modelManager.unloadAll();
  }
});
```

## API Reference

### Core Functions

- `initLlama(params: ContextParams): Promise<LlamaContext>` - Initialize a model context
- `releaseAllLlama(): Promise<void>` - Release all contexts
- `setContextLimit(limit: number): Promise<void>` - Set maximum contexts
- `toggleNativeLog(enabled: boolean): Promise<void>` - Enable/disable native logging

### Multimodal Functions

- `initMultimodal(contextId: number, mmprojPath: string, useGpu?: boolean): Promise<boolean>`
- `multimodalCompletion(contextId: number, prompt: string, mediaPaths: string[], params: NativeCompletionParams): Promise<NativeCompletionResult>`
- `isMultimodalEnabled(contextId: number): Promise<boolean>`
- `isMultimodalSupportVision(contextId: number): Promise<boolean>`
- `isMultimodalSupportAudio(contextId: number): Promise<boolean>`
- `releaseMultimodal(contextId: number): Promise<void>`

### Text-to-Speech Functions

- `initVocoder(contextId: number, vocoderModelPath: string): Promise<boolean>`
- `isVocoderEnabled(contextId: number): Promise<boolean>`
- `getTTSType(contextId: number): Promise<NativeTTSType>`
- `getFormattedAudioCompletion(contextId: number, speakerJsonStr: string, textToSpeak: string): Promise<NativeAudioCompletionResult>`
- `getAudioCompletionGuideTokens(contextId: number, textToSpeak: string): Promise<NativeAudioTokensResult>`
- `decodeAudioTokens(contextId: number, tokens: number[]): Promise<NativeAudioDecodeResult>`
- `releaseVocoder(contextId: number): Promise<void>`

### Context Methods

- `completion(params: CompletionParams, callback?: (token: TokenData) => void): Promise<NativeCompletionResult>`
- `embedding(text: string, params?: EmbeddingParams): Promise<NativeEmbeddingResult>`
- `tokenize(text: string): Promise<NativeTokenizeResult>`
- `detokenize(tokens: number[]): Promise<string>`
- `saveSession(filepath: string, options?: { tokenSize: number }): Promise<number>`
- `loadSession(filepath: string): Promise<NativeSessionLoadResult>`
- `release(): Promise<void>`

## Troubleshooting

### Common Issues

**Model Loading Fails**
```typescript
// Check file exists and is accessible
if (!(await RNFS.exists(modelPath))) {
  throw new Error('Model file not found');
}

// Check file size
const stats = await RNFS.stat(modelPath);
console.log('Model size:', stats.size);
```

**Out of Memory**
```typescript
// Reduce context size
const context = await initLlama({
  model: '/path/to/model.gguf',
  n_ctx: 1024,     // Reduce from 4096
  n_batch: 128,    // Reduce batch size
});
```

**GPU Issues**
```typescript
// Disable GPU if having issues
const context = await initLlama({
  model: '/path/to/model.gguf',
  n_gpu_layers: 0,  // Use CPU only
});
```

### Performance Tips

1. **Use appropriate context sizes** - Larger contexts use more memory
2. **Optimize batch sizes** - Balance between speed and memory
3. **Cache models** - Don't reload models unnecessarily
4. **Use GPU acceleration** - When available and stable
5. **Monitor memory usage** - Release contexts when not needed

This documentation covers the essential usage patterns for cactus-react-native. For more examples, check the [example apps](../examples/) in the repository.
