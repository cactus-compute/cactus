import { Platform } from "react-native";
import { initLlama, LlamaContext, initMultimodal } from "cactus-react-native";
import RNFS from 'react-native-fs';

// VLM Model URLs from the C++ example
const modelUrl = 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf';
const mmprojUrl = 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf';

const modelFileName = 'SmolVLM-256M-Instruct-Q8_0.gguf';
const mmprojFileName = 'mmproj-SmolVLM-256M-Instruct-Q8_0.gguf';

async function downloadFile(url: string, fileName: string, progressCallback: (progress: number, file: string) => void): Promise<string> {
    const documentsPath = RNFS.DocumentDirectoryPath;
    const filePath = `${documentsPath}/${fileName}`;
    
    // Check if file already exists
    const fileExists = await RNFS.exists(filePath);
    if (fileExists) {
        console.log(`${fileName} already exists at:`, filePath);
        return filePath;
    }
    
    console.log(`Downloading ${fileName} to:`, filePath);
    
    const downloadResult = RNFS.downloadFile({
        fromUrl: url,
        toFile: filePath,
        progress: (res) => {
            const progress = (res.bytesWritten / res.contentLength) * 100;
            progressCallback(Math.round(progress), fileName);
        },
    });
    
    await downloadResult.promise;
    console.log(`${fileName} downloaded successfully`);
    return filePath;
}

export async function initVLMContext(progressCallback: (progress: number, file: string) => void): Promise<LlamaContext> {
    // Download both model and mmproj files
    console.log('Downloading VLM model and multimodal projector...');
    
    const modelPath = await downloadFile(modelUrl, modelFileName, progressCallback);
    const mmprojPath = await downloadFile(mmprojUrl, mmprojFileName, progressCallback);
    
    console.log('Initializing VLM context...');
    const context = await initLlama({
        model: modelPath,
        use_mlock: true,
        n_ctx: 2048,
        n_gpu_layers: Platform.OS === 'ios' ? 99 : 0
    });
    
    console.log('Initializing multimodal capabilities...');
    console.log('Context ID:', context.id);
    console.log('mmproj path:', mmprojPath);
    
    // Check file exists before calling
    const mmprojExists = await RNFS.exists(mmprojPath);
    console.log('mmproj file exists:', mmprojExists);
    
    if (!mmprojExists) {
        throw new Error(`mmproj file does not exist at path: ${mmprojPath}`);
    }
    
    try {
        // Note: GPU acceleration disabled for multimodal projector due to iOS Simulator compatibility issues
        const multimodalSuccess = await initMultimodal(context.id, mmprojPath, false);
        console.log('initMultimodal result:', multimodalSuccess);
        
        if (!multimodalSuccess) {
            console.warn('Failed to initialize multimodal capabilities');
        } else {
            console.log('VLM context initialized successfully with multimodal support');
        }
    } catch (error) {
        console.error('Error during initMultimodal:', error);
        throw error;
    }
    
    return context;
}

// Keep the old function for backwards compatibility
export async function initLlamaContext(progressCallback: (progress: number) => void): Promise<LlamaContext> {
    return initVLMContext((progress, file) => progressCallback(progress));
}