import { initLlama, LlamaContext } from "cactus-react-native";
import { Platform } from "react-native";
import * as FileSystem from 'expo-file-system';

const modelName = 'qwen2.5-1.5b-instruct-q8_0.gguf';
const modelDirectory = Platform.OS === 'ios' ? `${FileSystem.documentDirectory}local-models/`: `${FileSystem.cacheDirectory}local-models/`;
const fullModelPath = `${modelDirectory}${modelName}`;

const modelUrl = 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf';

async function modelExists(): Promise<Boolean> {
    return (await FileSystem.getInfoAsync(fullModelPath)).exists;
}

export async function initLlamaContext(progressCallback: (progress: number) => void): Promise<LlamaContext> {

    const modelIsDownloaded = await modelExists();

    if (!modelIsDownloaded) {
        console.log(`Model is not downloaded, downloading into ${fullModelPath}...`);
        await FileSystem.makeDirectoryAsync(modelDirectory, { intermediates: true }).catch(() => {});
        const result = await FileSystem.createDownloadResumable(
            modelUrl,
            fullModelPath,
            {},
            (progress) => {
                progressCallback(progress.totalBytesWritten / progress.totalBytesExpectedToWrite);
            }
          ).downloadAsync();
        console.log('Downloaded result:', result);
    }

    if (!(await modelExists())) {
        console.error('Model is not downloaded');
    }

    return await initLlama({
        model: fullModelPath,
        use_mlock: true,
        n_ctx: 2048,
        n_gpu_layers: Platform.OS === 'ios' ? 99 : 0
    });
}