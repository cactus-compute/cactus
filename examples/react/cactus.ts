import { Platform } from 'react-native';
import { LlamaContext, multimodalCompletion, initLlama, initMultimodal } from 'cactus-react-native';
import type { CactusOAICompatibleMessage } from 'cactus-react-native';
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
  private lastInteractionWasMultimodal = false;

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

    console.log('🚀 generateResponse called');

    const isCurrentlyMultimodal = !!(userMessage.images && userMessage.images.length > 0);
    
    // Clear conversation history when switching between modes to prevent context bleeding
    if (this.lastInteractionWasMultimodal !== isCurrentlyMultimodal) {
      console.log(`🔄 Switching from ${this.lastInteractionWasMultimodal ? 'multimodal' : 'text'} to ${isCurrentlyMultimodal ? 'multimodal' : 'text'} mode - clearing history`);
      this.conversationHistory = [];
    }

    this.lastInteractionWasMultimodal = isCurrentlyMultimodal;

    if (userMessage.images && userMessage.images.length > 0) {
      // For multimodal messages, use the demo image path
      const localImagePaths = userMessage.images.map(() => this.demoImagePath!).filter(Boolean);
      
      console.log('🖼️ Using multimodal completion with local images:', localImagePaths);
      
      // Add user message to conversation history first
      this.conversationHistory.push({
        role: 'user',
        content: userMessage.content
      });

      // Use proper chat template formatting for multimodal - get formatted prompt first
      const formattedPrompt = await this.context.getFormattedChat(
        this.conversationHistory,
        undefined, // Let the model use its default chat template
        { jinja: true }
      );

      // Extract prompt string from formatted result
      const promptString = typeof formattedPrompt === 'string' ? formattedPrompt : formattedPrompt.prompt;

      console.log('📝 Formatted multimodal prompt:', promptString.substring(0, 100) + '...');

      // Use multimodal completion with properly formatted prompt
      const multimodalResult = await multimodalCompletion(
        this.context.id,
        promptString,
        localImagePaths,
        {
          prompt: promptString,
          n_predict: 256,
          stop: stopWords,
          temperature: 0.3, // Slightly higher to reduce repetition but still focused
          top_p: 0.9,
          penalty_repeat: 1.1, // Prevent repetitive outputs
          emit_partial_completion: false,
        }
      );

      const responseText = multimodalResult.text || 'No response generated';
      
      // Add assistant response to history
      this.conversationHistory.push({
        role: 'assistant', 
        content: responseText
      });

      console.log('🖼️ Multimodal completion finished');
      return responseText;
    } else {
      // For text-only messages, use pure text completion to avoid visual interference
      
      // Add user message to conversation history
      this.conversationHistory.push({
        role: 'user',
        content: userMessage.content
      });

      console.log('🔤 Using pure text completion (no image context)');
      console.log('📊 Conversation history length:', this.conversationHistory.length);
      console.log('🏁 About to call context.completion...');

      try {
        // Use text-only completion with explicit empty image array to prevent visual context
        const result = await this.context.completion({
          messages: this.conversationHistory,
          n_predict: 256,
          stop: stopWords,
          temperature: 0.7, // Higher temperature for creative text responses
          top_p: 0.9,
          penalty_repeat: 1.05, // Light repetition penalty for text
        });

        console.log('✅ Text completion successful, result received');
        console.log('📝 Response length:', result.text?.length || 0);

        const responseText = result.text || 'No response generated';
        
        // Add assistant response to history
        this.conversationHistory.push({
          role: 'assistant',
          content: responseText
        });

        console.log('💾 Text response added to history');
        return responseText;
      } catch (error) {
        console.error('❌ Text completion failed:', error);
        throw error;
      }
    }
  }

  clearConversation(): void {
    this.conversationHistory = [];
    this.lastInteractionWasMultimodal = false;
    console.log('Conversation history cleared and interaction mode reset');
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