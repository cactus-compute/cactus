import { Platform } from 'react-native';
import { LlamaContext, multimodalCompletion, initLlama, initMultimodal } from 'cactus-react-native-3';
import type { CactusOAICompatibleMessage } from 'cactus-react-native-3';
import RNFS from 'react-native-fs';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  images?: string[];
}

// VLM Model URLs
const modelUrl = 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf';
const mmprojUrl = 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf';
const modelFileName = 'SmolVLM-256M-Instruct-Q8_0.gguf';
const mmprojFileName = 'mmproj-SmolVLM-256M-Instruct-Q8_0.gguf';
const demoImageUrl = 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop';

const stopWords = ['<|end_of_text|>', '<|endoftext|>', '</s>', '<end_of_utterance>'];

class CactusManager {
  private context: LlamaContext | null = null;
  private isInitialized = false;
  private conversationHistory: CactusOAICompatibleMessage[] = [];
  private demoImagePath: string | null = null;

  async downloadFile(url: string, fileName: string, progressCallback: (progress: number, file: string) => void): Promise<string> {
    const documentsPath = RNFS.DocumentDirectoryPath;
    const filePath = `${documentsPath}/${fileName}`;

    if (await RNFS.exists(filePath)) {
      console.log(`${fileName} already exists at: ${filePath}`);
      return filePath;
    }

    console.log(`Downloading ${fileName}...`);
    progressCallback(0, fileName);

    const { promise } = RNFS.downloadFile({
      fromUrl: url,
      toFile: filePath,
      progress: (res) => {
        const progress = res.bytesWritten / res.contentLength;
        progressCallback(progress, fileName);
      },
    });

    const result = await promise;
    
    if (result.statusCode === 200) {
      console.log(`${fileName} downloaded successfully`);
      return filePath;
    } else {
      throw new Error(`Failed to download ${fileName}`);
    }
  }

  async downloadDemoImage(): Promise<string> {
    const documentsPath = RNFS.DocumentDirectoryPath;
    const targetPath = `${documentsPath}/demo_image.jpg`;
    
    if (await RNFS.exists(targetPath)) {
      console.log('Demo image already exists at:', targetPath);
      return targetPath;
    }
    
    console.log('Downloading demo image...');
    const { promise } = RNFS.downloadFile({
      fromUrl: demoImageUrl,
      toFile: targetPath,
    });

    const result = await promise;

    if (result.statusCode === 200) {
      console.log('Demo image downloaded to:', targetPath);
      return targetPath;
    } else {
      throw new Error(`Failed to download demo image. Status code: ${result.statusCode}`);
    }
  }

  async initialize(progressCallback: (progress: number, file: string) => void): Promise<void> {
    if (this.isInitialized) return;

    console.log('Downloading VLM model, multimodal projector, and demo image...');

    // Download model, multimodal projector, and demo image
    const [modelPath, mmprojPath, demoPath] = await Promise.all([
      this.downloadFile(modelUrl, modelFileName, progressCallback),
      this.downloadFile(mmprojUrl, mmprojFileName, progressCallback),
      this.downloadDemoImage()
    ]);

    this.demoImagePath = demoPath;

    console.log('Initializing VLM context...');

    // Initialize llama context with VLM model
    this.context = await initLlama({
      model: modelPath,
      n_ctx: 2048,
      n_batch: 32,
      n_gpu_layers: 99, // Use GPU acceleration for main model
      n_threads: 4,
      embedding: false,
    });

    console.log('Initializing multimodal capabilities...');
    console.log('Context ID:', this.context.id);
    console.log('mmproj path:', mmprojPath);
    console.log('mmproj file exists:', await RNFS.exists(mmprojPath));

    // Import the initMultimodal function and initialize multimodal support
    const multimodalSuccess = await initMultimodal(this.context.id, mmprojPath, false); // Disable GPU for iOS simulator compatibility
    console.log('initMultimodal result:', multimodalSuccess);
    
    if (!multimodalSuccess) {
      console.warn('Failed to initialize multimodal capabilities');
    } else {
      console.log('VLM context initialized successfully with multimodal support');
    }

    this.isInitialized = true;
  }

  async generateResponse(userMessage: Message): Promise<string> {
    if (!this.context) {
      throw new Error('Cactus context not initialized');
    }

    // Add user message to conversation history
    if (userMessage.images && userMessage.images.length > 0) {
      // For multimodal messages, use the demo image path
      const localImagePaths = userMessage.images.map(() => this.demoImagePath!).filter(Boolean);
      
      console.log('Using multimodal completion with local images:', localImagePaths);
      
      // Build conversation context from history
      let conversationContext = '';
      this.conversationHistory.forEach(msg => {
        conversationContext += `${msg.role === 'user' ? 'Human' : 'Assistant'}: ${msg.content}\n`;
      });
      conversationContext += `Human: ${userMessage.content}`;
      
      const result = await multimodalCompletion(
        this.context.id,
        conversationContext,
        localImagePaths,
        {
          prompt: conversationContext,
          n_predict: 256,
          stop: stopWords,
          emit_partial_completion: false
        }
      );

      const responseText = result.text || 'No response generated';
      
      // Add both user and assistant messages to history
      this.conversationHistory.push({
        role: 'user',
        content: userMessage.content + (userMessage.images ? ' [image attached]' : '')
      });
      this.conversationHistory.push({
        role: 'assistant', 
        content: responseText
      });

      return responseText;
    } else {
      // For text-only messages, use proper chat template formatting
      
      // Add user message to conversation history
      this.conversationHistory.push({
        role: 'user',
        content: userMessage.content
      });

      console.log('Using chat template completion for text-only message');
      console.log('Conversation history length:', this.conversationHistory.length);

      const result = await this.context.completion({
        messages: this.conversationHistory,
        n_predict: 256,
        stop: stopWords
      });

      const responseText = result.text || 'No response generated';
      
      // Add assistant response to history
      this.conversationHistory.push({
        role: 'assistant',
        content: responseText
      });

      return responseText;
    }
  }

  clearConversation(): void {
    this.conversationHistory = [];
    console.log('Conversation history cleared');
  }

  getDemoImageUri(): string {
    if (this.demoImagePath) {
      return this.demoImagePath.startsWith('file://') ? this.demoImagePath : `file://${this.demoImagePath}`;
    }
    return demoImageUrl; // Fallback to external URL
  }

  getIsInitialized(): boolean {
    return this.isInitialized;
  }

  getConversationLength(): number {
    return this.conversationHistory.length;
  }
}

// Export singleton instance
export const cactus = new CactusManager(); 