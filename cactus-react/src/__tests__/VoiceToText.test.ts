import { NativeModules, Platform, NativeEventEmitter } from 'react-native';
import { VoiceToText } from '../VoiceToText'; // Adjust path as necessary

// Mock NativeModules
jest.mock('react-native', () => {
  const RN = jest.requireActual('react-native');

  // Mock specific native modules that VoiceToText interacts with
  RN.NativeModules.AudioInputModule = {
    requestPermissions: jest.fn(),
    startRecording: jest.fn(),
    stopRecording: jest.fn(),
    initSTT: jest.fn(),
    processAudioFile: jest.fn(),
    releaseSTT: jest.fn(),
    // Mock any other methods VoiceToText might call on AudioInputModule
  };

  RN.NativeModules.CactusModule = {
    // Assuming CactusModule on Android has a similar STT API or audio methods
    requestPermissions: jest.fn(),
    startRecording: jest.fn(),
    stopRecording: jest.fn(),
    initSTT: jest.fn(),
    processAudioFile: jest.fn(),
    releaseSTT: jest.fn(),
    // Mock any other methods VoiceToText might call on CactusModule
  };

  // Mock NativeEventEmitter
  RN.NativeEventEmitter = jest.fn(() => ({
    addListener: jest.fn((eventName, callback) => ({
        remove: jest.fn(), // Return a mock subscription object
    })),
    removeListener: jest.fn(), // Deprecated, but good to have a mock
    removeAllListeners: jest.fn(), // For cleanup
    // Mock other NativeEventEmitter methods if used by VoiceToText
  }));


  return RN;
});


describe('VoiceToText', () => {
  let voiceToText: VoiceToText;
  let mockNativeEventEmitterInstance: NativeEventEmitter;

  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();

    // Re-assign mock instances if NativeEventEmitter constructor is crucial for the test
    // This ensures that the addListener is called on the instance used by VoiceToText
    mockNativeEventEmitterInstance = new NativeEventEmitter(Platform.OS === 'ios' ? NativeModules.AudioInputModule : NativeModules.CactusModule);
    (NativeEventEmitter as jest.Mock).mockImplementation(() => mockNativeEventEmitterInstance);


    voiceToText = new VoiceToText();
  });

  afterEach(() => {
    voiceToText.release(); // Clean up listeners
  });

  it('constructor initializes listeners', () => {
    expect(NativeEventEmitter).toHaveBeenCalledTimes(1);
    expect(mockNativeEventEmitterInstance.addListener).toHaveBeenCalledWith('onAudioData', expect.any(Function));
    expect(mockNativeEventEmitterInstance.addListener).toHaveBeenCalledWith('onError', expect.any(Function));
  });

  describe('requestPermissions', () => {
    it('calls AudioInputModule.requestPermissions on iOS', async () => {
      Platform.OS = 'ios';
      (NativeModules.AudioInputModule.requestPermissions as jest.Mock).mockResolvedValue(true);
      const result = await voiceToText.requestPermissions();
      expect(NativeModules.AudioInputModule.requestPermissions).toHaveBeenCalled();
      expect(result).toBe(true);
    });

    it('calls CactusModule.requestPermissions on Android', async () => {
      Platform.OS = 'android';
      // Re-create voiceToText for Android platform context in constructor
      voiceToText = new VoiceToText();
      (NativeModules.CactusModule.requestPermissions as jest.Mock).mockResolvedValue(true);
      const result = await voiceToText.requestPermissions();
      expect(NativeModules.CactusModule.requestPermissions).toHaveBeenCalled();
      expect(result).toBe(true);
    });
  });

  describe('initSTT', () => {
    it('calls native initSTT and sets modelPath', async () => {
      Platform.OS = 'ios'; // Or 'android', behavior should be similar for this method
      voiceToText = new VoiceToText();
      const modelPath = 'path/to/model';
      (NativeModules.AudioInputModule.initSTT as jest.Mock).mockResolvedValue(undefined);
      await voiceToText.initSTT(modelPath);
      expect(NativeModules.AudioInputModule.initSTT).toHaveBeenCalledWith(modelPath);
      // @ts-ignore // Access private member for test verification
      expect(voiceToText.modelPath).toBe(modelPath);
    });
  });

  describe('start', () => {
    it('throws error if STT model not initialized', async () => {
      await expect(voiceToText.start()).rejects.toThrow('STT model not initialized.');
    });

    it('calls native startRecording if model is initialized', async () => {
      Platform.OS = 'ios';
      voiceToText = new VoiceToText();
      await voiceToText.initSTT('path/to/model');
      (NativeModules.AudioInputModule.startRecording as jest.Mock).mockResolvedValue('Recording started');
      await voiceToText.start();
      expect(NativeModules.AudioInputModule.startRecording).toHaveBeenCalled();
      // @ts-ignore
      expect(voiceToText.isRecording).toBe(true);
    });
  });

  describe('stop', () => {
    it('calls native stopRecording', async () => {
      Platform.OS = 'ios';
      voiceToText = new VoiceToText();
      await voiceToText.initSTT('path/to/model');
      await voiceToText.start(); // Start recording first
      (NativeModules.AudioInputModule.stopRecording as jest.Mock).mockResolvedValue({ filePath: 'audio.m4a', fileSize: 1234 });
      await voiceToText.stop();
      expect(NativeModules.AudioInputModule.stopRecording).toHaveBeenCalled();
       // @ts-ignore
      expect(voiceToText.isRecording).toBe(false);
    });
  });

  describe('processAudio', () => {
    it('throws error if STT model not initialized', async () => {
      await expect(voiceToText.processAudio('path/to/audio.m4a')).rejects.toThrow('STT model not initialized.');
    });

    it('calls native processAudioFile and sets transcription', async () => {
      Platform.OS = 'ios';
      voiceToText = new VoiceToText();
      const audioPath = 'path/to/audio.m4a';
      const mockTranscription = 'Hello world';
      await voiceToText.initSTT('path/to/model');
      (NativeModules.AudioInputModule.processAudioFile as jest.Mock).mockResolvedValue(mockTranscription);

      const transcription = await voiceToText.processAudio(audioPath);

      expect(NativeModules.AudioInputModule.processAudioFile).toHaveBeenCalledWith(audioPath);
      expect(transcription).toBe(mockTranscription);
      // @ts-ignore
      expect(voiceToText.currentTranscription).toBe(mockTranscription);
      // Verify event emission if processAudio is expected to emit 'onTranscription'
      // This depends on your VoiceToText implementation details for event handling
    });
  });


  describe('releaseSTT', () => {
    it('calls native releaseSTT and resets modelPath', async () => {
      Platform.OS = 'ios';
      voiceToText = new VoiceToText();
      await voiceToText.initSTT('path/to/model');
      (NativeModules.AudioInputModule.releaseSTT as jest.Mock).mockResolvedValue(undefined);
      await voiceToText.releaseSTT();
      expect(NativeModules.AudioInputModule.releaseSTT).toHaveBeenCalled();
      // @ts-ignore
      expect(voiceToText.modelPath).toBeNull();
    });
  });

  describe('setUserVocabulary (Placeholder)', () => {
    it('logs a message and resolves', async () => {
      const consoleSpy = jest.spyOn(console, 'log');
      await voiceToText.setUserVocabulary(['test', 'vocab']);
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('setUserVocabulary called with 2 items.'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('This feature is a placeholder'));
      consoleSpy.mockRestore();
    });
  });

  describe('Event Handling', () => {
    it('handles onAudioData event and calls processAudio', () => {
        Platform.OS = 'ios';
        voiceToText = new VoiceToText(); // Re-init to ensure fresh emitter mock

        // Initialize STT so processAudio can be called
        const modelPath = 'test/model.bin';
        (NativeModules.AudioInputModule.initSTT as jest.Mock).mockResolvedValue(undefined);
        voiceToText.initSTT(modelPath);

        const mockProcessAudio = jest.spyOn(voiceToText, 'processAudio').mockResolvedValue('mock transcription');

        // Simulate event emission
        const mockAudioDataEvent = { filePath: 'test/audio.wav' };
        // Find the onAudioData callback registered by VoiceToText
        const onAudioDataCallback = (mockNativeEventEmitterInstance.addListener as jest.Mock).mock.calls.find(
            call => call[0] === 'onAudioData'
        )[1];
        onAudioDataCallback(mockAudioDataEvent);

        expect(mockProcessAudio).toHaveBeenCalledWith(mockAudioDataEvent.filePath);
        mockProcessAudio.mockRestore();
    });

    it('handles onError event', () => {
        Platform.OS = 'ios';
        voiceToText = new VoiceToText(); // Re-init
        const consoleErrorSpy = jest.spyOn(console, 'error');

        const mockErrorEvent = { message: 'Test error from native' };
        const onErrorCallback = (mockNativeEventEmitterInstance.addListener as jest.Mock).mock.calls.find(
            call => call[0] === 'onError'
        )[1];
        onErrorCallback(mockErrorEvent);

        expect(consoleErrorSpy).toHaveBeenCalledWith('Native module error:', mockErrorEvent);
        consoleErrorSpy.mockRestore();
    });
  });

  // Test release method more thoroughly if it does more than remove listeners
  it('release method cleans up subscriptions', () => {
    // Spy on the remove methods of the mock subscriptions
    // This requires a more elaborate setup for the mock subscriptions if not already done
    // For this example, we assume the addListener mock returns an object with a 'remove' jest.fn()
    // This is already handled by the current NativeEventEmitter mock.

    // Call release
    voiceToText.release();

    // Check if the remove methods on subscriptions were called
    // This part is tricky because the subscriptions are stored internally.
    // A more robust way would be to check the effect of release, e.g., no more event processing.
    // Or, ensure the mock addListener returns spyable remove functions.
    // The current mock does provide a jest.fn() for remove, but we don't have direct access to it here
    // without modifying VoiceToText to expose subscriptions or making the mock more complex.
    // However, we can verify that the NativeEventEmitter's internal removeAllListeners (if used) or
    // individual removeListener calls were made if VoiceToText used them.
    // For now, this test mainly ensures `release` runs without error and calls `releaseSTT`.
    expect(NativeModules.AudioInputModule.releaseSTT).toHaveBeenCalledTimes(1); // or CactusModule for Android
  });

});
