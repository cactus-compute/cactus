import { NativeModules, EmitterSubscription, NativeEventEmitter, Platform } from 'react-native';

const LINKING_ERROR =
  `The package 'cactus-react-native' doesn't seem to be linked. Make sure: \n\n` +
  Platform.select({ ios: "- You have run 'pod install'\n", default: '' }) +
  '- You rebuilt the app after installing the package\n' +
  '- You are not using Expo Go\n';

// Define the interface for the AudioInputModule (iOS)
interface AudioInputModuleIOS {
  requestPermissions(): Promise<boolean>;
  startRecording(): Promise<string>;
  stopRecording(): Promise<{ filePath: string; fileSize: number }>;
  initSTT(modelPath: string): Promise<void>;
  processAudioFile(filePath: string): Promise<string>; // Returns transcription
  releaseSTT(): Promise<void>;
}

// Define the interface for the CactusModule (Android - assumed to be augmented for STT)
interface CactusModuleAndroid {
  // Assuming it might have its own audio methods or uses a separate module
  // For now, let's include STT methods here as per the task
  initSTT(modelPath: string): Promise<void>;
  processAudioFile(filePath: string): Promise<string>; // Returns transcription
  releaseSTT(): Promise<void>;

  // Placeholder for Android-specific audio methods if they are part of CactusModule
  // If Android uses a separate RNAudioInputModule, that would be defined separately
  requestPermissions?(): Promise<boolean>;
  startRecording?(): Promise<string>;
  stopRecording?(): Promise<{ filePath: string; fileSize: number }>;
}

const AudioInputModuleIOS: AudioInputModuleIOS = NativeModules.AudioInputModule
  ? NativeModules.AudioInputModule
  : new Proxy(
      {},
      {
        get() {
          throw new Error(Platform.OS === 'ios' ? LINKING_ERROR : 'AudioInputModule is not available on Android. Use CactusModule directly for STT or a specific Android audio module.');
        },
      }
    );

const CactusModuleAndroid: CactusModuleAndroid = NativeModules.CactusModule
  ? NativeModules.CactusModule
  : new Proxy(
      {},
      {
        get() {
          throw new Error(Platform.OS === 'android' ? LINKING_ERROR : 'CactusModule is not available on iOS for these specific audio methods. Use AudioInputModule.');
        },
      }
    );


/**
 * Handler for events emitted by the VoiceToText service, such as transcriptions or errors.
 * @param event - The event object, specific to the event type.
 */
export type VoiceToTextEventHandler = (event: any) => void;

/**
 * Provides an interface for voice-to-text (STT) functionality,
 * including audio recording, permission handling, and STT engine interaction.
 */
export class VoiceToText {
  private nativeAudioModule: AudioInputModuleIOS | CactusModuleAndroid; // Combined type
  private eventEmitter: NativeEventEmitter;
  private onAudioDataSubscription?: EmitterSubscription;
  private onErrorSubscription?: EmitterSubscription;

  private modelPath: string | null = null;
  private isRecording: boolean = false;
  private currentTranscription: string | null = null;

  /**
   * Initializes the VoiceToText service, setting up native module access
   * and event listeners for audio data and errors.
   */
  constructor() {
    if (Platform.OS === 'ios') {
      this.nativeAudioModule = AudioInputModuleIOS;
      this.eventEmitter = new NativeEventEmitter(NativeModules.AudioInputModule);
    } else if (Platform.OS === 'android') {
      // Assuming CactusModule will also handle events if it does audio.
      // If Android has a separate AudioInputModule, this would need adjustment.
      this.nativeAudioModule = CactusModuleAndroid;
      this.eventEmitter = new NativeEventEmitter(NativeModules.CactusModule);
    } else {
      throw new Error('Unsupported platform');
    }
    this.setupListeners();
  }

  private setupListeners() {
    this.onAudioDataSubscription = this.eventEmitter.addListener('onAudioData', (data: { filePath: string }) => {
      console.log('onAudioData:', data);
      // Automatically process audio if a model is initialized
      if (this.modelPath && data.filePath) {
        this.processAudio(data.filePath).catch(error => {
          console.error('Error auto-processing audio:', error);
          // Optionally emit an error event to the JS consumer
          this.eventEmitter.emit('onError', { message: 'Error auto-processing audio', details: error });
        });
      }
    });

    this.onErrorSubscription = this.eventEmitter.addListener('onError', (error: any) => {
      console.error('Native module error:', error);
      // Forward or handle error
    });
  }

  /**
   * Requests microphone permissions from the user.
   * On iOS, this calls the `requestPermissions` method of `AudioInputModule`.
   * On Android, this method should ideally be called after an explicit Android
   * permission request flow (e.g., using `PermissionsAndroid` from `react-native`).
   * The `VoiceToText.ts` example `App.tsx` demonstrates this Android flow.
   * @returns A promise that resolves to `true` if permission is granted, `false` otherwise.
   */
  async requestPermissions(): Promise<boolean> {
    if (Platform.OS === 'ios' && this.nativeAudioModule.requestPermissions) {
        return (this.nativeAudioModule as AudioInputModuleIOS).requestPermissions();
    } else if (Platform.OS === 'android' && (this.nativeAudioModule as CactusModuleAndroid).requestPermissions) {
        return (this.nativeAudioModule as CactusModuleAndroid).requestPermissions!();
    }
    console.warn('requestPermissions not implemented for this platform in VoiceToText module or underlying native module.');
    return false;
  }

  /**
   * Starts audio recording.
   * Requires STT to be initialized via `initSTT` first.
   * @returns A promise that resolves with a message upon successful start, or rejects on error.
   * @throws An error if STT is not initialized or if recording fails to start.
   */
  async start(): Promise<string> {
    if (this.isRecording) {
      console.warn('Recording is already in progress.');
      return "Already recording";
    }
    if (!this.modelPath) {
        throw new Error('STT model not initialized. Call initSTT(modelPath) first.');
    }
    if (Platform.OS === 'ios' && this.nativeAudioModule.startRecording) {
        const result = await (this.nativeAudioModule as AudioInputModuleIOS).startRecording();
        this.isRecording = true;
        return result;
    } else if (Platform.OS === 'android' && (this.nativeAudioModule as CactusModuleAndroid).startRecording) {
        const result = await (this.nativeAudioModule as CactusModuleAndroid).startRecording!();
        this.isRecording = true;
        return result;
    }
    throw new Error('startRecording not implemented for this platform.');
  }

  /**
   * Stops the current audio recording.
   * @returns A promise that resolves with an object containing the `filePath` and `fileSize`
   *          of the recorded audio, or `null` if no recording was in progress. Rejects on error.
   * @throws An error if stopping the recording fails.
   */
  async stop(): Promise<{ filePath: string; fileSize: number } | null> {
    if (!this.isRecording) {
      console.warn('No recording in progress to stop.');
      return null;
    }
    if (Platform.OS === 'ios' && this.nativeAudioModule.stopRecording) {
        const result = await (this.nativeAudioModule as AudioInputModuleIOS).stopRecording();
        this.isRecording = false;
        return result;
    } else if (Platform.OS === 'android' && (this.nativeAudioModule as CactusModuleAndroid).stopRecording) {
        const result = await (this.nativeAudioModule as CactusModuleAndroid).stopRecording!();
        this.isRecording = false;
        return result;
    }
    throw new Error('stopRecording not implemented for this platform.');
  }

  /**
   * Initializes the Speech-to-Text (STT) engine with the specified model.
   * The model file must exist at the given `modelPath` on the device.
   * @param modelPath - The local file system path to the STT model file.
   * @returns A promise that resolves when the STT engine is initialized, or rejects on error.
   */
  async initSTT(modelPath: string): Promise<void> {
    this.modelPath = modelPath;
    return this.nativeAudioModule.initSTT(modelPath);
  }

  /**
   * Processes a given audio file for speech-to-text transcription.
   * Requires STT to be initialized via `initSTT` first.
   * The result of the transcription is typically emitted via an 'onTranscription' event
   * by the underlying native module, or directly returned if the native method supports it.
   * This method itself updates `currentTranscription` and emits an `onTranscription` event via the JS event emitter.
   * @param audioPath - The local file system path to the audio file to be processed.
   * @returns A promise that resolves with the transcription string.
   * @throws An error if STT is not initialized or if processing fails.
   */
  async processAudio(audioPath: string): Promise<string> {
    if (!this.modelPath) {
        throw new Error('STT model not initialized. Call initSTT(modelPath) first.');
    }
    const transcription = await this.nativeAudioModule.processAudioFile(audioPath);
    this.currentTranscription = transcription;
    // Optionally emit an event with the new transcription
    this.eventEmitter.emit('onTranscription', { transcription });
    return transcription;
  }

  /**
   * Retrieves the most recent transcription result.
   * @returns The current transcription string, or `null` if no transcription is available.
   */
  getTranscription(): string | null {
    return this.currentTranscription;
  }

  /**
   * Releases resources used by the STT engine.
   * Should be called when STT functionality is no longer needed to free up memory.
   * @returns A promise that resolves when STT resources are released.
   */
  async releaseSTT(): Promise<void> {
    if (this.modelPath) {
      await this.nativeAudioModule.releaseSTT();
      this.modelPath = null;
      this.currentTranscription = null;
    }
  }

  /**
   * Releases all resources used by the VoiceToText instance, including
   * native event listeners and the STT engine.
   * Call this when the component using VoiceToText is unmounted.
   */
  release(): void {
    this.onAudioDataSubscription?.remove();
    this.onErrorSubscription?.remove();
    this.releaseSTT().catch(error => console.error("Error releasing STT resources:", error));
    console.log('VoiceToText module released.');
  }

  /**
   * Sets a user-specific vocabulary (initial prompt) to guide the STT engine.
   * This can improve transcription accuracy for specific terms or contexts.
   * @param vocabulary A string containing words or phrases.
   * @returns A promise that resolves when the vocabulary is set, or rejects on error.
   * @remarks The Android native implementation for STT is currently facing integration issues.
   */
  async setUserVocabulary(vocabulary: string): Promise<void> {
    if (!this.modelPath) {
      throw new Error('STT model not initialized. Call initSTT(modelPath) first.');
    }
    if (Platform.OS === 'ios') {
      // Assuming AudioInputModuleIOS will have setUserVocabulary exposed from native
      const module = this.nativeAudioModule as AudioInputModuleIOS & { setUserVocabulary?: (vocab: string) => Promise<void> };
      if (module.setUserVocabulary) {
        return module.setUserVocabulary(vocabulary);
      } else {
        throw new Error('setUserVocabulary not implemented in AudioInputModule for iOS');
      }
    } else if (Platform.OS === 'android') {
      // Assuming CactusModuleAndroid will have setUserVocabulary exposed from native
      const module = this.nativeAudioModule as CactusModuleAndroid & { setUserVocabulary?: (vocab: string) => Promise<void> };
      if (module.setUserVocabulary) {
        return module.setUserVocabulary(vocabulary);
      } else {
        throw new Error('setUserVocabulary not implemented in CactusModule for Android');
      }
    }
    throw new Error('setUserVocabulary not implemented for this platform.');
  }
}

// Example of how to handle events if CactusModule is also an emitter for Android
// This assumes NativeModules.CactusModule exists and can emit events.
// If Android uses a different module for audio events, that module's NativeEventEmitter would be used.
// For now, the constructor handles platform-specific emitter setup.
