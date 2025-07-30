# Cactus React Native

Run LLMs, VLMs, and TTS models directly on mobile devices.

## Installation

```json
{
  "dependencies": {
    "cactus-react-native": "^0.2.4",
    "react-native-fs": "^2.20.0"
  }
}
```

**Setup:**
- iOS: `cd ios && npx pod-install`
- Android: Ensure `minSdkVersion` 24+

## Quick Start

```typescript
import { CactusLM } from 'cactus-react-native';
import RNFS from 'react-native-fs';

const modelPath = `${RNFS.DocumentDirectoryPath}/model.gguf`;

const { lm, error } = await CactusLM.init({
  model: modelPath,
  n_ctx: 2048,
  n_threads: 4,
});

if (error) throw error;

const messages = [{ role: 'user', content: 'Hello!' }];
const result = await lm.completion(messages, { n_predict: 100 });
console.log(result.text);
lm.release();
```

## Streaming Chat

```typescript
import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, ActivityIndicator } from 'react-native';
import { CactusLM } from 'cactus-react-native';
import RNFS from 'react-native-fs';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function ChatScreen() {
  const [lm, setLM] = useState<CactusLM | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    initializeModel();
    return () => {
      lm?.release();
    };
  }, []);

  const initializeModel = async () => {
    try {
      const modelUrl = 'https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf';
      const modelPath = await downloadModel(modelUrl, 'qwen-600m.gguf');

      const { lm: model, error } = await CactusLM.init({
        model: modelPath,
        n_ctx: 2048,
        n_threads: 4,
        n_gpu_layers: 99,
      });

      if (error) throw error;
      setLM(model);
    } catch (error) {
      console.error('Failed to initialize model:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadModel = async (url: string, filename: string): Promise<string> => {
    const path = `${RNFS.DocumentDirectoryPath}/${filename}`;
    
    if (await RNFS.exists(path)) return path;
    
    console.log('Downloading model...');
    await RNFS.downloadFile({
      fromUrl: url,
      toFile: path,
      progress: (res) => {
        const progress = res.bytesWritten / res.contentLength;
        console.log(`Download progress: ${(progress * 100).toFixed(1)}%`);
      },
    }).promise;
    
    return path;
  };

  const sendMessage = async () => {
    if (!lm || !input.trim() || isGenerating) return;

    const userMessage: Message = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMessage];
    setMessages([...newMessages, { role: 'assistant', content: '' }]);
    setInput('');
    setIsGenerating(true);

    try {
      let response = '';
      await lm.completion(newMessages, {
        n_predict: 200,
        temperature: 0.7,
        stop: ['</s>', '<|end|>'],
      }, (token) => {
        response += token.token;
        setMessages(prev => [
          ...prev.slice(0, -1),
          { role: 'assistant', content: response }
        ]);
      });
    } catch (error) {
      console.error('Generation failed:', error);
      setMessages(prev => [
        ...prev.slice(0, -1),
        { role: 'assistant', content: 'Error generating response' }
      ]);
    } finally {
      setIsGenerating(false);
    }
  };

  if (isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator size="large" />
        <Text style={{ marginTop: 16 }}>Loading model...</Text>
      </View>
    );
  }

  return (
    <View style={{ flex: 1, backgroundColor: '#f5f5f5' }}>
      <ScrollView style={{ flex: 1, padding: 16 }}>
        {messages.map((msg, index) => (
          <View
            key={index}
            style={{
              backgroundColor: msg.role === 'user' ? '#007AFF' : '#ffffff',
              padding: 12,
              marginVertical: 4,
              borderRadius: 12,
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '80%',
              shadowColor: '#000',
              shadowOffset: { width: 0, height: 1 },
              shadowOpacity: 0.2,
              shadowRadius: 2,
              elevation: 2,
            }}
          >
            <Text style={{ 
              color: msg.role === 'user' ? '#ffffff' : '#000000',
              fontSize: 16,
            }}>
              {msg.content}
            </Text>
          </View>
        ))}
      </ScrollView>
      
      <View style={{
        flexDirection: 'row',
        padding: 16,
        backgroundColor: '#ffffff',
        borderTopWidth: 1,
        borderTopColor: '#e0e0e0',
      }}>
        <TextInput
          style={{
            flex: 1,
            borderWidth: 1,
            borderColor: '#e0e0e0',
            borderRadius: 20,
            paddingHorizontal: 16,
            paddingVertical: 10,
            fontSize: 16,
            backgroundColor: '#f8f8f8',
          }}
          value={input}
          onChangeText={setInput}
          placeholder="Type a message..."
          multiline
          onSubmitEditing={sendMessage}
        />
        <TouchableOpacity
          onPress={sendMessage}
          disabled={isGenerating || !input.trim()}
          style={{
            backgroundColor: isGenerating ? '#cccccc' : '#007AFF',
            borderRadius: 20,
            paddingHorizontal: 16,
            paddingVertical: 10,
            marginLeft: 8,
            justifyContent: 'center',
          }}
        >
          <Text style={{ color: '#ffffff', fontWeight: 'bold' }}>
            {isGenerating ? '...' : 'Send'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}
```

## Core APIs

### CactusLM

```typescript
import { CactusLM } from 'cactus-react-native';

const { lm, error } = await CactusLM.init({
  model: '/path/to/model.gguf',
  n_ctx: 2048,
  n_threads: 4,
  n_gpu_layers: 99,
  embedding: true,
});

const messages = [{ role: 'user', content: 'What is AI?' }];
const result = await lm.completion(messages, {
  n_predict: 200,
  temperature: 0.7,
  stop: ['</s>'],
});

const embedding = await lm.embedding('Your text here');
await lm.rewind();
await lm.release();
```

### CactusVLM

```typescript
import { CactusVLM } from 'cactus-react-native';

const { vlm, error } = await CactusVLM.init({
  model: '/path/to/vision-model.gguf',
  mmproj: '/path/to/mmproj.gguf',
  n_ctx: 2048,
});

const messages = [{ role: 'user', content: 'Describe this image' }];
const result = await vlm.completion(messages, {
  images: ['/path/to/image.jpg'],
  n_predict: 200,
  temperature: 0.3,
});

await vlm.release();
```

### CactusTTS

```typescript
import { CactusTTS, initLlama } from 'cactus-react-native';

const context = await initLlama({
  model: '/path/to/tts-model.gguf',
  n_ctx: 1024,
});

const tts = await CactusTTS.init(context, '/path/to/vocoder.gguf');

const audio = await tts.generate(
  'Hello, this is text-to-speech',
  '{"speaker_id": 0}'
);

await tts.release();
```

## Advanced Usage

### Model Manager

```typescript
class ModelManager {
  private models = new Map<string, CactusLM | CactusVLM>();
  
  async loadLM(name: string, modelPath: string): Promise<CactusLM> {
    if (this.models.has(name)) {
      return this.models.get(name) as CactusLM;
    }
    
    const { lm, error } = await CactusLM.init({
      model: modelPath,
      n_ctx: 2048,
    });
    
    if (error) throw error;
    this.models.set(name, lm);
    return lm;
  }
  
  async loadVLM(name: string, modelPath: string, mmprojPath: string): Promise<CactusVLM> {
    if (this.models.has(name)) {
      return this.models.get(name) as CactusVLM;
    }
    
    const { vlm, error } = await CactusVLM.init({
      model: modelPath,
      mmproj: mmprojPath,
    });
    
    if (error) throw error;
    this.models.set(name, vlm);
    return vlm;
  }
  
  async releaseModel(name: string): Promise<void> {
    const model = this.models.get(name);
    if (model) {
      await model.release();
      this.models.delete(name);
    }
  }
  
  async releaseAll(): Promise<void> {
    await Promise.all(
      Array.from(this.models.values()).map(model => model.release())
    );
    this.models.clear();
  }
}

const modelManager = new ModelManager();
```

### File Management Hook

```typescript
import { useState, useCallback } from 'react';
import RNFS from 'react-native-fs';

interface DownloadProgress {
  progress: number;
  isDownloading: boolean;
  error: string | null;
}

export const useModelDownload = () => {
  const [downloads, setDownloads] = useState<Map<string, DownloadProgress>>(new Map());
  
  const downloadModel = useCallback(async (url: string, filename: string): Promise<string> => {
    const path = `${RNFS.DocumentDirectoryPath}/${filename}`;
    
    if (await RNFS.exists(path)) {
      const stats = await RNFS.stat(path);
      if (stats.size > 0) return path;
    }
    
    setDownloads(prev => new Map(prev.set(filename, {
      progress: 0,
      isDownloading: true,
      error: null,
    })));
    
    try {
      await RNFS.downloadFile({
        fromUrl: url,
        toFile: path,
        progress: (res) => {
          const progress = res.bytesWritten / res.contentLength;
          setDownloads(prev => new Map(prev.set(filename, {
            progress,
            isDownloading: true,
            error: null,
          })));
        },
      }).promise;
      
      setDownloads(prev => new Map(prev.set(filename, {
        progress: 1,
        isDownloading: false,
        error: null,
      })));
      
      return path;
    } catch (error) {
      setDownloads(prev => new Map(prev.set(filename, {
        progress: 0,
        isDownloading: false,
        error: error.message,
      })));
      throw error;
    }
  }, []);
  
  return { downloadModel, downloads };
};
```

### Vision Chat Component

```typescript
import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, Image, Alert } from 'react-native';
import { launchImageLibrary } from 'react-native-image-picker';
import { CactusVLM } from 'cactus-react-native';
import RNFS from 'react-native-fs';

export default function VisionChat() {
  const [vlm, setVLM] = useState<CactusVLM | null>(null);
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    initializeVLM();
    return () => {
      vlm?.release();
    };
  }, []);

  const initializeVLM = async () => {
    try {
      const modelUrl = 'https://huggingface.co/Cactus-Compute/SmolVLM2-500m-Instruct-GGUF/resolve/main/SmolVLM2-500M-Video-Instruct-Q8_0.gguf';
      const mmprojUrl = 'https://huggingface.co/Cactus-Compute/SmolVLM2-500m-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf';
      
      const [modelPath, mmprojPath] = await Promise.all([
        downloadFile(modelUrl, 'smolvlm-model.gguf'),
        downloadFile(mmprojUrl, 'smolvlm-mmproj.gguf'),
      ]);

      const { vlm: model, error } = await CactusVLM.init({
        model: modelPath,
        mmproj: mmprojPath,
        n_ctx: 2048,
      });

      if (error) throw error;
      setVLM(model);
    } catch (error) {
      console.error('Failed to initialize VLM:', error);
      Alert.alert('Error', 'Failed to initialize vision model');
    } finally {
      setIsLoading(false);
    }
  };

  const downloadFile = async (url: string, filename: string): Promise<string> => {
    const path = `${RNFS.DocumentDirectoryPath}/${filename}`;
    
    if (await RNFS.exists(path)) return path;
    
    await RNFS.downloadFile({ fromUrl: url, toFile: path }).promise;
    return path;
  };

  const pickImage = () => {
    launchImageLibrary(
      {
        mediaType: 'photo',
        quality: 0.8,
        includeBase64: false,
      },
      (response) => {
        if (response.assets && response.assets[0]) {
          setImagePath(response.assets[0].uri!);
          setResponse('');
        }
      }
    );
  };

  const analyzeImage = async () => {
    if (!vlm || !imagePath) return;

    setIsAnalyzing(true);
    try {
      const messages = [{ role: 'user', content: 'Describe this image in detail' }];
      
      let analysisResponse = '';
      const result = await vlm.completion(messages, {
        images: [imagePath],
        n_predict: 300,
        temperature: 0.3,
      }, (token) => {
        analysisResponse += token.token;
        setResponse(analysisResponse);
      });

      setResponse(analysisResponse || result.text);
    } catch (error) {
      console.error('Analysis failed:', error);
      Alert.alert('Error', 'Failed to analyze image');
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <Text>Loading vision model...</Text>
      </View>
    );
  }

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 24, fontWeight: 'bold', marginBottom: 20 }}>
        Vision Chat
      </Text>
      
      {imagePath && (
        <Image
          source={{ uri: imagePath }}
          style={{
            width: '100%',
            height: 200,
            borderRadius: 8,
            marginBottom: 16,
          }}
          resizeMode="contain"
        />
      )}
      
      <View style={{ flexDirection: 'row', marginBottom: 16 }}>
        <TouchableOpacity
          onPress={pickImage}
          style={{
            backgroundColor: '#007AFF',
            padding: 12,
            borderRadius: 8,
            marginRight: 8,
            flex: 1,
          }}
        >
          <Text style={{ color: 'white', textAlign: 'center', fontWeight: 'bold' }}>
            Pick Image
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          onPress={analyzeImage}
          disabled={!imagePath || isAnalyzing}
          style={{
            backgroundColor: !imagePath || isAnalyzing ? '#cccccc' : '#34C759',
            padding: 12,
            borderRadius: 8,
            flex: 1,
          }}
        >
          <Text style={{ color: 'white', textAlign: 'center', fontWeight: 'bold' }}>
            {isAnalyzing ? 'Analyzing...' : 'Analyze'}
          </Text>
        </TouchableOpacity>
      </View>
      
      <View style={{
        flex: 1,
        backgroundColor: '#f8f8f8',
        borderRadius: 8,
        padding: 16,
      }}>
        <Text style={{ fontSize: 16, lineHeight: 24 }}>
          {response || 'Select an image and tap Analyze to get started'}
        </Text>
      </View>
    </View>
  );
}
```

### Cloud Fallback

```typescript
const { lm } = await CactusLM.init({
  model: '/path/to/model.gguf',
  n_ctx: 2048,
}, undefined, 'your_cactus_token');

// Try local first, fallback to cloud if local fails
const embedding = await lm.embedding('text', undefined, 'localfirst');

// Vision models also support cloud fallback
const { vlm } = await CactusVLM.init({
  model: '/path/to/model.gguf',
  mmproj: '/path/to/mmproj.gguf',
}, undefined, 'your_cactus_token');

const result = await vlm.completion(messages, {
  images: ['/path/to/image.jpg'],
  mode: 'localfirst',
});
```

### Embeddings & Similarity

```typescript
const { lm } = await CactusLM.init({
  model: '/path/to/model.gguf',
  embedding: true,
});

const embedding1 = await lm.embedding('machine learning');
const embedding2 = await lm.embedding('artificial intelligence');

function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

const similarity = cosineSimilarity(embedding1.embedding, embedding2.embedding);
console.log('Similarity:', similarity);
```

## Error Handling & Performance

### Production Error Handling

```typescript
async function safeModelInit(modelPath: string): Promise<CactusLM> {
  const configs = [
    { model: modelPath, n_ctx: 4096, n_gpu_layers: 99 },
    { model: modelPath, n_ctx: 2048, n_gpu_layers: 99 },
    { model: modelPath, n_ctx: 2048, n_gpu_layers: 0 },
    { model: modelPath, n_ctx: 1024, n_gpu_layers: 0 },
  ];

  for (const config of configs) {
    try {
      const { lm, error } = await CactusLM.init(config);
      if (error) throw error;
      return lm;
    } catch (error) {
      console.warn('Config failed:', config, error.message);
      if (configs.indexOf(config) === configs.length - 1) {
        throw new Error(`All configurations failed. Last error: ${error.message}`);
      }
    }
  }
  
  throw new Error('Model initialization failed');
}

async function safeCompletion(lm: CactusLM, messages: any[], retries = 3): Promise<any> {
  for (let i = 0; i < retries; i++) {
    try {
      return await lm.completion(messages, { n_predict: 200 });
    } catch (error) {
      if (error.message.includes('Context is busy') && i < retries - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        continue;
      }
      throw error;
    }
  }
}
```

### Memory Management

```typescript
import { AppState, AppStateStatus } from 'react-native';

class AppModelManager {
  private modelManager = new ModelManager();
  
  constructor() {
    AppState.addEventListener('change', this.handleAppStateChange);
  }
  
  private handleAppStateChange = (nextAppState: AppStateStatus) => {
    if (nextAppState === 'background') {
      // Release non-essential models when app goes to background
      this.modelManager.releaseAll();
    }
  };
  
  async getModel(name: string, modelPath: string): Promise<CactusLM> {
    try {
      return await this.modelManager.loadLM(name, modelPath);
    } catch (error) {
      // Handle low memory by releasing other models
      await this.modelManager.releaseAll();
      return await this.modelManager.loadLM(name, modelPath);
    }
  }
}
```

### Performance Optimization

```typescript
// Optimize for device capabilities
const getOptimalConfig = () => {
  const { OS } = Platform;
  const isHighEndDevice = true; // Implement device detection logic
  
  return {
    n_ctx: isHighEndDevice ? 4096 : 2048,
    n_gpu_layers: OS === 'ios' ? 99 : 0, // iOS generally has better GPU support
    n_threads: isHighEndDevice ? 6 : 4,
    n_batch: isHighEndDevice ? 512 : 256,
  };
};

const config = getOptimalConfig();
const { lm } = await CactusLM.init({
  model: modelPath,
  ...config,
});
```

## Tool Calling with CactusAgent

The `CactusAgent` class extends `CactusLM` with built-in tool calling capabilities:

```typescript
import { CactusAgent } from 'cactus-react-native';

const { agent, error } = await CactusAgent.init({
  model: '/path/to/model.gguf', // we recommend Qwen 3 family, 0.6B is great
  n_ctx: 2048,
});

const weatherTool = agent.addTool(
  (location: string) => `Weather in ${location}: 72°F, sunny`,
  'Get current weather for a location',
  {
    location: { type: 'string', description: 'City name', required: true }
  }
);

const messages = [{ role: 'user', content: 'What\'s the weather in NYC?' }];
const result = await agent.completionWithTools(messages, {
  n_predict: 200,
  temperature: 0.7,
});

await agent.release();
```

### Custom Tools

```typescript
// Math calculator tool
const calculator = agent.addTool(
  (expression: string) => {
    try {
      return `Result: ${eval(expression)}`;
    } catch (e) {
      return 'Invalid expression';
    }
  },
  'Evaluate mathematical expressions',
  {
    expression: { type: 'string', description: 'Math expression to evaluate', required: true }
  }
);
```

## API Reference

### CactusLM

**init(params, onProgress?, cactusToken?)**
- `model: string` - Path to GGUF model file
- `n_ctx?: number` - Context size (default: 2048)
- `n_threads?: number` - CPU threads (default: 4)
- `n_gpu_layers?: number` - GPU layers (default: 99)
- `embedding?: boolean` - Enable embeddings (default: false)
- `n_batch?: number` - Batch size (default: 512)

**completion(messages, params?, callback?)**
- `messages: Array<{role: string, content: string}>` - Chat messages
- `n_predict?: number` - Max tokens (default: -1)
- `temperature?: number` - Randomness 0.0-2.0 (default: 0.8)
- `top_p?: number` - Nucleus sampling (default: 0.95)
- `top_k?: number` - Top-k sampling (default: 40)
- `stop?: string[]` - Stop sequences
- `callback?: (token) => void` - Streaming callback

**embedding(text, params?, mode?)**
- `text: string` - Text to embed
- `mode?: string` - 'local' | 'localfirst' | 'remotefirst' | 'remote'

### CactusVLM

**init(params, onProgress?, cactusToken?)**
- All CactusLM params plus:
- `mmproj: string` - Path to multimodal projector

**completion(messages, params?, callback?)**
- All CactusLM completion params plus:
- `images?: string[]` - Array of image paths
- `mode?: string` - Cloud fallback mode

### Types

```typescript
interface CactusOAICompatibleMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface NativeCompletionResult {
  text: string;
  tokens_predicted: number;
  tokens_evaluated: number;
  timings: {
    predicted_per_second: number;
    prompt_per_second: number;
  };
}

interface NativeEmbeddingResult {
  embedding: number[];
}
```
