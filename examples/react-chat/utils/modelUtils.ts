import { Platform } from "react-native";
import { initLlama, LlamaContext, downloadModelIfNotExists } from "cactus-react-native";

const modelUrl = 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf';

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