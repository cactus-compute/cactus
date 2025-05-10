import { initLlama, LlamaContext } from "cactus-react-native";
import { Platform } from "react-native";
import { downloadModelIfNotExists } from "cactus-react-native";

const modelUrl = 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf';

export async function initLlamaContext(progressCallback: (progress: number) => void): Promise<LlamaContext> {

    const fullModelPath = await downloadModelIfNotExists({
        modelUrl,
        modelFolderName: 'local-models',
        onProgress: progressCallback,
        onSuccess: (_) => {
            console.log('Model downloaded successfully');
        }
    });

    return await initLlama({
        model: fullModelPath,
        use_mlock: true,
        n_ctx: 2048,
        n_gpu_layers: Platform.OS === 'ios' ? 99 : 0
    });
}