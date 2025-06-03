# Cactus for React Native

A lightweight, high-performance framework for running AI models on mobile devices with React Native.

## Installation

```bash
# Using npm
npm install react-native-fs
npm install cactus-react-native

# Using yarn
yarn add react-native-fs
yarn add cactus-react-native

# For iOS, install pods if not on Expo
npx pod-install
```

## Basic Usage

### Initialize a Model

```typescript
import { initLlama, LlamaContext } from 'cactus-react-native';

// Initialize the model
const context = await initLlama({
  model: 'models/llama-2-7b-chat.gguf', // Path to your model
  n_ctx: 2048,                          // Context size
  n_batch: 512,                         // Batch size for prompt processing
  n_threads: 4                          // Number of threads to use
});
```

### Text Completion

```typescript
// Generate text completion
const result = await context.completion({
  prompt: "Explain quantum computing in simple terms",
  temperature: 0.7,
  top_k: 40,
  top_p: 0.95,
  n_predict: 512
}, (token) => {
  // Process each token as it's generated
  console.log(token.token);
});

// Clean up when done
await context.release();
```

### Chat Completion

```typescript
// Chat messages following OpenAI format
const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is machine learning?" }
];

// Generate chat completion
const result = await context.completion({
  messages: messages,
  temperature: 0.7,
  top_k: 40,
  top_p: 0.95,
  n_predict: 512
}, (token) => {
  // Process each token
  console.log(token.token);
});
```

## Advanced Features

### JSON Mode with Schema Validation

```typescript
// Define a JSON schema
const schema = {
  type: "object",
  properties: {
    name: { type: "string" },
    age: { type: "number" },
    hobbies: { 
      type: "array",
      items: { type: "string" }
    }
  },
  required: ["name", "age"]
};

// Generate JSON-structured output
const result = await context.completion({
  prompt: "Generate a profile for a fictional person",
  response_format: {
    type: "json_schema",
    json_schema: {
      schema: schema,
      strict: true
    }
  },
  temperature: 0.7,
  n_predict: 512
});

// The result will be valid JSON according to the schema
const jsonData = JSON.parse(result.text);
```

### Working with Embeddings

```typescript
// Generate embeddings for text
const embedding = await context.embedding("This is a sample text", {
  pooling_type: "mean" // Options: "none", "mean", "cls", "last", "rank"
});

console.log(`Embedding dimensions: ${embedding.embedding.length}`);
// Use the embedding for similarity comparison, clustering, etc.
```

### Session Management

```typescript
// Save the current session state
const tokenCount = await context.saveSession("session.bin", { tokenSize: 1024 });
console.log(`Saved session with ${tokenCount} tokens`);

// Load a saved session
const loadResult = await context.loadSession("session.bin");
console.log(`Loaded session: ${loadResult.success}`);
```

### Working with LoRA Adapters

```typescript
// Apply LoRA adapters to the model
await context.applyLoraAdapters([
  { path: "models/lora_adapter.bin", scaled: 0.8 }
]);

// Get currently loaded adapters
const loadedAdapters = await context.getLoadedLoraAdapters();

// Remove all LoRA adapters
await context.removeLoraAdapters();
```

### Model Benchmarking

```typescript
// Benchmark the model performance
const benchResult = await context.bench(
  32,  // pp: prompt processing tests
  32,  // tg: token generation tests
  512, // pl: prompt length
  5    // nr: number of runs
);

console.log(`Average token generation speed: ${benchResult.tgAvg} tokens/sec`);
console.log(`Model size: ${benchResult.modelSize} bytes`);
```

### Native Logging

```typescript
import { addNativeLogListener, toggleNativeLog } from 'cactus-react-native';

// Enable native logging
await toggleNativeLog(true);

// Add a listener for native logs
const logListener = addNativeLogListener((level, text) => {
  console.log(`[${level}] ${text}`);
});

// Remove the listener when no longer needed
logListener.remove();
```

## Error Handling

```typescript
try {
  const context = await initLlama({
    model: 'models/non-existent-model.gguf',
    n_ctx: 2048,
    n_threads: 4
  });
} catch (error) {
  console.error('Failed to initialize model:', error);
}
```

## Best Practices

1. **Model Management**
   - Store models in the app's document directory
   - Consider model size when targeting specific devices
   - Smaller models like SmolLM (135M) work well on most devices

2. **Performance Optimization**
   - Adjust `n_threads` based on the device's capabilities
   - Use a smaller `n_ctx` for memory-constrained devices
   - Consider INT8 or INT4 quantized models for better performance

3. **Battery Efficiency**
   - Release the model context when not in use
   - Process inference in smaller batches
   - Consider background processing for long generations

4. **Memory Management**
   - Always call `context.release()` when done with a model
   - Use `releaseAllLlama()` when switching between multiple models

## Example App

For a complete working example, check out the [React Native example app](https://github.com/cactus-compute/cactus/tree/main/examples/react-example) in the repository. (Note: This link might need updating if a specific voice chat example is added, e.g., `examples/react-native-voice-chat/`).

## Voice-to-Text (STT)

The `VoiceToText` class provides an interface for speech-to-text functionality using the device microphone.

### Basic STT Usage

```typescript
import { VoiceToText } from 'cactus-react-native';
import { Platform, PermissionsAndroid } from 'react-native'; // For Android permissions

const voiceToText = new VoiceToText();

// 1. Initialize the STT Engine (e.g., with a model path)
//    Ensure the model file exists at the specified path on the device.
//    This path can be from your app's assets, documents directory, or cache.
const modelPath = Platform.OS === 'ios' ? 'path/to/your_stt_model.bin' : '/data/user/0/com.yourapp/files/your_stt_model.bin';
// Note: Actual model path will vary based on how you bundle/download it.

async function initSTTEngine() {
  try {
    // Make sure the model path is correct and the model file is accessible
    // You might need to bundle the model with your app or download it first.
    // For this example, we assume `modelPath` is valid.
    await voiceToText.initSTT(modelPath);
    console.log('STT Engine Initialized');
  } catch (e) {
    console.error('Failed to initialize STT:', e);
  }
}

// Call initialization (e.g., in your component's useEffect or an init function)
// initSTTEngine();


// 2. Request Microphone Permissions
async function requestMicPermission() {
  if (Platform.OS === 'android') {
    try {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
        {
          title: 'Microphone Permission',
          message: 'This app needs access to your microphone for voice transcription.',
          buttonPositive: 'OK',
        },
      );
      return granted === PermissionsAndroid.RESULTS.GRANTED;
    } catch (err) {
      console.warn(err);
      return false;
    }
  } else if (Platform.OS === 'ios') {
    return await voiceToText.requestPermissions();
  }
  return false;
}

// 3. Start and Stop Voice Capture
let currentTranscription = '';
let sttError = '';

// Setup event listeners (typically in useEffect or similar)
// This requires access to the NativeEventEmitter instance used by VoiceToText,
// or VoiceToText should expose its own event handling mechanism.
// The example App.tsx in `examples/react-native-voice-chat` shows direct NativeEventEmitter usage.
// For simplicity here, we assume VoiceToText might have its own event subscription methods
// or you'd adapt the direct NativeEventEmitter pattern.

// voiceToText.on('onTranscription', (event) => { // Hypothetical event listener on VoiceToText
//   if (event.transcription) {
//     currentTranscription = event.transcription;
//     console.log('Transcription:', currentTranscription);
//   }
// });
// voiceToText.on('onError', (error) => {
//    sttError = error.message || JSON.stringify(error);
//    console.error('STT Error:', sttError);
// });


async function toggleRecording(isRecordingCurrently: boolean) {
  const hasPermission = await requestMicPermission();
  if (!hasPermission) {
    console.error('Microphone permission denied.');
    sttError = 'Microphone permission denied.';
    return;
  }

  // Ensure STT is initialized before starting
  // if (!voiceToText.isInitialized()) { // Assuming an isInitialized() method
  //   await initSTTEngine(); // Or handle error if init failed previously
  // }


  if (isRecordingCurrently) {
    try {
      await voiceToText.stop();
      console.log('Recording stopped.');
      // Transcription result might come via an event or be part of the stop() promise.
      // If `voiceToText.stop()` returns the final audio path, `voiceToText.processAudio(path)`
      // would be called, and its result (or an event it triggers) would provide the transcription.
    } catch (e) {
      console.error('Failed to stop recording:', e);
      sttError = 'Failed to stop: ' + (e as Error).message;
    }
  } else {
    try {
      currentTranscription = ''; // Clear previous
      sttError = '';
      await voiceToText.start();
      console.log('Recording started...');
    } catch (e) {
      console.error('Failed to start recording:', e);
      sttError = 'Failed to start: ' + (e as Error).message;
    }
  }
  // Update your UI state for isRecording, currentTranscription, sttError
}

// 4. Process a pre-existing audio file
async function transcribeAudioFile(filePath: string) {
  try {
    // Ensure STT is initialized
    // if (!voiceToText.isInitialized()) { await initSTTEngine(); }
    const transcription = await voiceToText.processAudio(filePath);
    console.log('File Transcription:', transcription);
    currentTranscription = transcription;
  } catch (e) {
    console.error('Failed to process audio file:', e);
    sttError = 'File processing error: ' + (e as Error).message;
  }
}

// 5. Set User-Specific Vocabulary (Placeholder)
// This section is replaced by the one below.

// 6. Release STT resources when no longer needed (e.g., on component unmount)
// voiceToText.release();

```

### Setting User-Specific Vocabulary (Initial Prompt)

You can provide a string of custom vocabulary or context to improve transcription accuracy for specific terms. This is often referred to as setting an 'initial prompt'.

```typescript
import { VoiceToText } from 'cactus-react'; // Or your specific import

// ... Assuming voiceToText instance is created and STT is initialized:
// const voiceToText = new VoiceToText();
// await voiceToText.initSTT('path/to/your/model.gguf'); // Language defaults to 'en' or as per your initSTT

const customVocabulary = "EyeRIS Doctor Smith patient zero"; // Example as a single string
try {
  // Ensure STT is initialized in voiceToText before calling this
  if (voiceToText.modelPath) { // modelPath is set in VoiceToText after successful initSTT
    await voiceToText.setUserVocabulary(customVocabulary);
    console.log("User vocabulary set.");
  } else {
    console.log("STT not initialized, cannot set vocabulary.");
  }
} catch (e) {
  console.error("Failed to set user vocabulary", e);
}

// Proceed with voice processing...
```
**Note:** The Android implementation for STT features, including `setUserVocabulary`, is currently facing integration challenges due to issues with locating the core Java Native Module. Functionality on Android may be limited until these are resolved.

### Required Permissions

**Android (`AndroidManifest.xml`):**
You need to add the `RECORD_AUDIO` permission to your `android/app/src/main/AndroidManifest.xml`:
```xml
<manifest ...>
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
    ...
</manifest>
```

**iOS (`Info.plist`):**
Add the `NSMicrophoneUsageDescription` key to your `ios/[YourAppName]/Info.plist` file with a description of why your app needs microphone access:
```xml
<key>NSMicrophoneUsageDescription</key>
<string>This app uses the microphone to capture your voice for transcription.</string>
```

Make sure to handle permission requests appropriately within your app flow. The examples above show basic permission requests.

## Example App

This example demonstrates:
- Loading and initializing models
- Building a chat interface
- Streaming responses
- Proper resource management

## License

This project is licensed under the Apache 2.0 License.
