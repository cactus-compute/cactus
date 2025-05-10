import RNFS from 'react-native-fs'

const defaultModelFolderName: string = "models"
const defaultModelUrl: string = "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"

const supportedProviders: string[] = ['huggingface.co']
const supportedFormats: string[] = ['gguf']

export class ModelDownloader {
    modelUrl: string
    modelFolderName: string
    modelName: string = ''
    fullModelPath: string = ''
    fullModelFolderPath: string = ''

    constructor(modelUrl?: string, modelFolderName?: string) {
        this.modelUrl = modelUrl || defaultModelUrl
        this.modelFolderName = modelFolderName || defaultModelFolderName
        this.validateURLFormat();
        this.parseModelName();
    }

    validateURLFormat() {
        let isValid: boolean = true
        if (!this.modelUrl.startsWith('http://') && !this.modelUrl.startsWith('https://')) {
            console.log('Invalid model URL', this.modelUrl)
            isValid = false
        }
        if (!supportedProviders.includes(this.modelUrl.split('//').at(1)?.split('/').at(0) || '')) {
            console.log('Invalid model provider', this.modelUrl.split('//').at(0) || '')
            isValid = false
        }
        if (!supportedFormats.includes(this.modelUrl.split('.').at(-1) || '')) {
            console.log('Invalid model format', this.modelUrl.split('.').at(-1) || '')
            isValid = false
        }
        if (!isValid) {
            throw new Error("Invalid model URL")
        }
    }

    parseModelName() {
        const path = this.modelUrl.split('/').filter(Boolean)
        this.modelName = path.pop() || ''
        this.fullModelFolderPath = `${RNFS.DocumentDirectoryPath}/${this.modelFolderName}`
        this.fullModelPath = `${this.fullModelFolderPath}/${this.modelName}`
    }

    async createModelFolder() {
        if (!(await RNFS.exists(this.fullModelFolderPath))) {
            await RNFS.mkdir(this.fullModelFolderPath)
            console.log('Model folder created', this.fullModelFolderPath)
        }
    }

    async downloadModelIfNotExists(
        onProgress?: (progress: number) => void,
        onSuccess?: (modelPath: string) => void
    ): Promise<string> {
        if (!(await RNFS.exists(this.fullModelPath))) {
            await this.createModelFolder()
            await RNFS.downloadFile({
                fromUrl: this.modelUrl,
                toFile: this.fullModelPath,
                begin: (_: RNFS.DownloadBeginCallbackResult) => {}, // if you don't provide begin, progress won't work!!
                progress: onProgress ? (response: RNFS.DownloadProgressCallbackResult) => {
                    if (response.contentLength > 0) {
                        let percentage = Math.floor(response.bytesWritten/response.contentLength*10000)/100;
                        onProgress(percentage)
                    }
                } : () => {},
                progressInterval: 100,
            }).promise.then(() => {
                onSuccess?.(this.fullModelPath)
            }).catch((error: any) => {
                throw new Error(`Error downloading model: ${error}`)
            })
        }
        return this.fullModelPath;
    }
}