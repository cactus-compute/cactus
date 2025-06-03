import React, { useState, useEffect, useCallback } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  PermissionsAndroid,
  Platform,
  TextInput,
  EmitterSubscription,
  NativeEventEmitter, // Import NativeEventEmitter
  NativeModules, // Import NativeModules
} from 'react-native';
import { VoiceToText } from 'cactus-react'; // Assuming cactus-react is linked

// Get the AudioInputModule for direct event listening if VoiceToText doesn't re-emit all
// This is a common pattern if the class itself doesn't manage all event types.
const { AudioInputModule, CactusModule } = NativeModules;

const App = () => {
  const [voiceToText, setVoiceToText] = useState<VoiceToText | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [transcribedText, setTranscribedText] = useState('');
  const [sttError, setSttError] = useState('');
  const [sttModelPath, setSttModelPath] = useState<string | null>(null); // Example model path

  // Listener subscriptions
  let onAudioDataSubscription: EmitterSubscription | null = null;
  let onErrorSubscription: EmitterSubscription | null = null;
  let onTranscriptionSubscription: EmitterSubscription | null = null;

  useEffect(() => {
    // Initialize VoiceToText instance
    const vtt = new VoiceToText();
    setVoiceToText(vtt);

    // IMPORTANT: Determine the correct event emitter
    // VoiceToText class in cactus-react was written to use AudioInputModule for iOS events
    // and CactusModule for Android events.
    // If VoiceToText itself emits these events via its own Emitter, use that.
    // Otherwise, listen to the native modules directly as shown here.
    let eventEmitterSource;
    if (Platform.OS === 'ios') {
      eventEmitterSource = AudioInputModule;
    } else if (Platform.OS === 'android') {
      eventEmitterSource = CactusModule; // Assuming CactusModule handles STT events on Android
    }

    if (eventEmitterSource) {
        const nativeEventEmitter = new NativeEventEmitter(eventEmitterSource);

        // Listen for transcription results (if VoiceToText emits this)
        // This is an example if your VoiceToText class emits 'onTranscription'
        onTranscriptionSubscription = nativeEventEmitter.addListener('onTranscription', (event) => {
          console.log('onTranscription event:', event);
          if (event.transcription) {
            setTranscribedText(event.transcription);
            setSttError('');
          }
        });

        // Fallback: Listen for onAudioData if onTranscription is not directly available
        // or if you want to handle the raw file path from recording.
        onAudioDataSubscription = nativeEventEmitter.addListener('onAudioData', async (event) => {
            console.log('onAudioData event:', event);
            if (event.filePath && vtt && sttModelPath) { // Ensure model is initialized
                try {
                    // This assumes processAudio in VoiceToText will trigger an onTranscription event
                    // or directly return transcription. If it returns directly:
                    // const transcription = await vtt.processAudio(event.filePath);
                    // setTranscribedText(transcription);

                    // If processAudio internally emits 'onTranscription', this listener above will catch it.
                    // Otherwise, you might need to set state from a direct return value.
                    await vtt.processAudio(event.filePath);
                } catch (e: any) {
                    console.error('Error processing audio data:', e);
                    setSttError(e.message || 'Error processing audio data');
                }
            }
        });

        onErrorSubscription = nativeEventEmitter.addListener('onError', (error) => {
            console.error('STT Native Module Error:', error);
            setSttError(error.message || JSON.stringify(error));
            setIsRecording(false);
        });
    }


    // TODO: Set your actual STT model path here
    // This could come from a config file, user input, or be bundled.
    // For demonstration, we'll use a placeholder.
    // Ensure this model is available on the device at the specified path.
    const modelPath = Platform.OS === 'ios' ? 'path/to/your/stt_model.bin' : '/sdcard/stt_model.bin';
    setSttModelPath(modelPath);

    // Initialize STT engine when the component mounts and modelPath is known
    // This is simplified; in a real app, you might init based on user action or app state.
    // Also, ensure the model file exists at `modelPath` on the device.
    if (vtt && modelPath) {
        console.log(`Initializing STT with model: ${modelPath}`);
        vtt.initSTT(modelPath)
            .then(() => console.log('STT Engine Initialized'))
            .catch(e => {
                console.error('Failed to initialize STT:', e);
                setSttError(e.message || 'Failed to init STT');
            });
    }

    // Optional: Call setUserVocabulary if needed
    // vtt.setUserVocabulary(["custom word", "Cactus AI"])
    //   .then(() => console.log("User vocabulary set (placeholder)"))
    //   .catch(e => console.error("Error setting user vocabulary:", e));


    return () => {
      // Clean up listeners and VoiceToText instance
      onAudioDataSubscription?.remove();
      onErrorSubscription?.remove();
      onTranscriptionSubscription?.remove();
      voiceToText?.release();
    };
  }, []); // Empty dependency array ensures this runs once on mount

  const requestPermissionsAndroid = async () => {
    try {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
        {
          title: 'Microphone Permission',
          message: 'This app needs access to your microphone for voice transcription.',
          buttonNeutral: 'Ask Me Later',
          buttonNegative: 'Cancel',
          buttonPositive: 'OK',
        },
      );
      return granted === PermissionsAndroid.RESULTS.GRANTED;
    } catch (err) {
      console.warn(err);
      return false;
    }
  };

  const handleToggleRecord = async () => {
    if (!voiceToText || !sttModelPath) {
      setSttError('VoiceToText service or model path not available.');
      return;
    }

    // Initialize STT if not already (idempotent check or rely on constructor init)
    // For simplicity, assuming initSTT was called on mount. If not, call here:
    // try {
    //   await voiceToText.initSTT(sttModelPath);
    // } catch (e: any) {
    //   setSttError(`Failed to init STT: ${e.message}`);
    //   return;
    // }


    let hasPermission = false;
    if (Platform.OS === 'android') {
      hasPermission = await requestPermissionsAndroid();
    } else if (Platform.OS === 'ios') {
      // For iOS, requestPermissions is part of the VoiceToText class
      hasPermission = await voiceToText.requestPermissions();
    }

    if (!hasPermission) {
      setSttError('Microphone permission denied.');
      return;
    }

    if (isRecording) {
      try {
        console.log('Stopping recording...');
        // stop() should trigger onAudioData if successful, which then calls processAudio
        await voiceToText.stop();
        setIsRecording(false);
        console.log('Recording stopped.');
      } catch (e: any) {
        console.error('Failed to stop recording:', e);
        setSttError(e.message || 'Failed to stop recording');
        setIsRecording(false);
      }
    } else {
      try {
        setTranscribedText(''); // Clear previous transcription
        setSttError('');
        console.log('Starting recording...');
        await voiceToText.start();
        setIsRecording(true);
        console.log('Recording started.');
      } catch (e: any) {
        console.error('Failed to start recording:', e);
        setSttError(e.message || 'Failed to start recording');
      }
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>React Native Voice Chat</Text>
      <TouchableOpacity
        style={[styles.button, isRecording ? styles.buttonRecording : styles.buttonNotRecording]}
        onPress={handleToggleRecord}
        disabled={!sttModelPath} // Disable if model path not set (STT not ready)
        >
        <Text style={styles.buttonText}>
          {isRecording ? 'Stop Recording' : 'Start Recording'}
        </Text>
      </TouchableOpacity>
      <TextInput
        style={styles.textInput}
        value={transcribedText}
        onChangeText={setTranscribedText}
        placeholder="Transcribed text will appear here..."
        multiline
      />
      {sttError ? <Text style={styles.errorText}>{sttError}</Text> : null}
      {!sttModelPath && <Text style={styles.errorText}>STT Model path not set. Voice input disabled.</Text>}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f0f0f0',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  button: {
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 25,
    marginBottom: 20,
    elevation: 3,
    shadowOpacity: 0.3,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 2 },
  },
  buttonRecording: {
    backgroundColor: '#e74c3c', // Red when recording
  },
  buttonNotRecording: {
    backgroundColor: '#3498db', // Blue when not recording
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '500',
  },
  textInput: {
    width: '100%',
    height: 100,
    borderColor: '#bdc3c7',
    borderWidth: 1,
    borderRadius: 5,
    padding: 10,
    backgroundColor: 'white',
    textAlignVertical: 'top',
    marginBottom: 10,
  },
  errorText: {
    color: '#c0392b',
    marginTop: 10,
    textAlign: 'center',
  },
});

export default App;
