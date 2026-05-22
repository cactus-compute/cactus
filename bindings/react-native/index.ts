import {NativeModules} from 'react-native';

export interface CactusNativeModule {
  init(modelPath: string, corpusDir: string | null, cacheIndex: boolean): Promise<string>;
  destroy(handle: string): Promise<void>;
  reset(handle: string): Promise<void>;
  stop(handle: string): Promise<void>;
  prefill(
    handle: string,
    messagesJson: string,
    optionsJson: string | null,
    toolsJson: string | null,
    pcmDataBase64: string | null,
  ): Promise<string>;
  complete(
    handle: string,
    messagesJson: string,
    optionsJson: string | null,
    toolsJson: string | null,
    pcmDataBase64: string | null,
  ): Promise<string>;
  tokenize(handle: string, text: string): Promise<number[]>;
  scoreWindow(
    handle: string,
    tokens: number[],
    start: number,
    end: number,
    context: number,
  ): Promise<string>;
  transcribe(
    handle: string,
    audioPath: string | null,
    prompt: string | null,
    optionsJson: string | null,
    pcmDataBase64: string | null,
  ): Promise<string>;
  embed(handle: string, text: string, normalize: boolean): Promise<number[]>;
  imageEmbed(handle: string, imagePath: string): Promise<number[]>;
  audioEmbed(handle: string, audioPath: string): Promise<number[]>;
  vad(
    handle: string,
    audioPath: string | null,
    optionsJson: string | null,
    pcmDataBase64: string | null,
  ): Promise<string>;
  diarize(
    handle: string,
    audioPath: string | null,
    optionsJson: string | null,
    pcmDataBase64: string | null,
  ): Promise<string>;
  embedSpeaker(
    handle: string,
    audioPath: string | null,
    optionsJson: string | null,
    pcmDataBase64: string | null,
    maskWeights: number[] | null,
  ): Promise<string>;
  ragQuery(handle: string, query: string, topK: number): Promise<string>;
  indexInit(indexDir: string, embeddingDim: number): Promise<string>;
  indexAdd(
    handle: string,
    ids: number[],
    documents: string[],
    embeddings: number[][],
    metadatas: string[] | null,
  ): Promise<void>;
  indexDelete(handle: string, ids: number[]): Promise<void>;
  indexGet(handle: string, ids: number[]): Promise<string>;
  indexQuery(handle: string, embedding: number[], optionsJson: string | null): Promise<string>;
  indexCompact(handle: string): Promise<void>;
  indexDestroy(handle: string): Promise<void>;
  logSetLevel(level: number): Promise<void>;
  setTelemetryEnvironment(
    framework: string | null,
    cacheLocation: string | null,
    version: string | null,
  ): Promise<void>;
  setAppId(appId: string): Promise<void>;
  telemetryFlush(): Promise<void>;
  telemetryShutdown(): Promise<void>;
  getLastError(): Promise<string | null>;
}

const LINKING_ERROR =
  'Cactus native module not linked. ' +
  'Add the bindings/react-native native bridge files and register NativeModules.Cactus.';

const nativeModule = NativeModules.Cactus as CactusNativeModule | undefined;

if (!nativeModule) {
  throw new Error(LINKING_ERROR);
}

export const Cactus = nativeModule;

export default Cactus;
