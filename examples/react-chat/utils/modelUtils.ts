import { Platform } from "react-native";
import { initLlama, LlamaContext } from "cactus-react-native-2";
import RNFS from 'react-native-fs';

const modelUrl = 'https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/main/SmolLM-135M.Q6_K.gguf';
const modelFileName = 'SmolLM-135M.Q6_K.gguf';

async function downloadModel(progressCallback: (progress: number) => void): Promise<string> {
    const documentsPath = RNFS.DocumentDirectoryPath;
    const modelPath = `${documentsPath}/${modelFileName}`;
    
    // Check if model already exists
    const fileExists = await RNFS.exists(modelPath);
    if (fileExists) {
        console.log('Model already exists at:', modelPath);
        return modelPath;
    }
    
    console.log('Downloading model to:', modelPath);
    
    const downloadResult = RNFS.downloadFile({
        fromUrl: modelUrl,
        toFile: modelPath,
        progress: (res) => {
            const progress = (res.bytesWritten / res.contentLength) * 100;
            progressCallback(Math.round(progress));
        },
    });
    
    await downloadResult.promise;
    console.log('Model downloaded successfully');
    return modelPath;
}

export async function initLlamaContext(progressCallback: (progress: number) => void): Promise<LlamaContext> {
    const modelPath = await downloadModel(progressCallback);
    
    return await initLlama({
        model: modelPath,
        use_mlock: true,
        n_ctx: 2048,
        n_gpu_layers: Platform.OS === 'ios' ? 99 : 0
    });
}