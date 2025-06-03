// CactusSTTService.swift
import Foundation
import AVFoundation // For AVAudioSession if needed directly, though mostly via iOSAudioInputManager

// Assuming FFI functions are declared in a bridging header or available through a module
// These are placeholders for the actual C FFI function names that would be in `cactus.h` or a similar C header
// e.g., C Functions:
// int cactus_stt_init_ffi(const char* model_path);
// const char* cactus_stt_process_file_ffi(const char* file_path);
// void cactus_stt_release_ffi();
// void cactus_stt_free_string_ffi(const char* str);

/// Provides an interface for Speech-to-Text (STT) functionality.
///
/// This class manages audio input via `iOSAudioInputManager`, interacts with a C++ STT engine
/// (via FFI calls, currently placeholders), and provides transcription results.
public class CactusSTTService: NSObject, AudioInputManagerDelegate {

    private let audioInputManager: iOSAudioInputManager
    private var transcriptionHandler: ((String?, Error?) -> Void)?
    private var modelPath: String?
    // private var isSTTInitialized: Bool = false // Replaced by checking sttContext
    private var isCapturingVoice: Bool = false

    // Pointer to the native STT context
    private var sttContext: UnsafeMutablePointer<cactus_stt_context_t>? = nil

    /// Returns true if the STT service has been initialized with a model.
    public var isInitialized: Bool {
        return sttContext != nil
    }

    /// Initializes a new `CactusSTTService`.
    ///
    /// It sets up an internal `iOSAudioInputManager` and assigns itself as the delegate.
    public override init() {
        self.audioInputManager = iOSAudioInputManager()
        super.init()
        self.audioInputManager.delegate = self
    }

    /// Initializes the STT engine with the specified model file.
    ///
    /// This method should be called before any other STT operations.
    /// The actual STT engine initialization (e.g., loading the model via C FFI) is currently a placeholder.
    /// - Parameters:
    ///   - modelPath: The local file system path to the STT model.
    ///   - completion: A closure called upon completion, passing an optional `Error` if initialization failed.
    public func initSTT(modelPath: String, completion: @escaping (Error?) -> Void) {
        self.modelPath = modelPath
        print("[CactusSTTService] Initializing STT with model: \(modelPath)")

        if self.sttContext != nil {
            cactus_stt_free(self.sttContext)
            self.sttContext = nil
        }

        // Assuming language "en" for now, or make it a parameter
        if let modelPathCString = modelPath.cString(using: .utf8),
           let languageCString = "en".cString(using: .utf8) {
            self.sttContext = cactus_stt_init(modelPathCString, languageCString)
            if self.sttContext != nil {
                print("[CactusSTTService] STT initialized successfully.")
                completion(nil)
            } else {
                let error = STTError.initializationFailed("cactus_stt_init returned null.")
                print("[CactusSTTService] STT initialization error: \(error.localizedDescription)")
                completion(error)
            }
        } else {
            let error = STTError.initializationFailed("Failed to convert modelPath or language to C string.")
            print("[CactusSTTService] STT initialization error: \(error.localizedDescription)")
            completion(error)
        }
    }

    /// Starts capturing audio from the microphone and processes it for speech-to-text.
    ///
    /// Before starting, it checks if STT is initialized and if another capture is already in progress.
    /// It requests microphone permissions via `iOSAudioInputManager`.
    /// Transcription results or errors are delivered through the `transcriptionHandler`.
    /// - Parameter transcriptionHandler: A closure called with the transcription string or an error.
    public func startVoiceCapture(transcriptionHandler: @escaping (String?, Error?) -> Void) {
        guard isInitialized else { // Use the new isInitialized computed property
            transcriptionHandler(nil, STTError.notInitialized)
            return
        }
        guard !isCapturingVoice else {
            transcriptionHandler(nil, STTError.alreadyCapturing)
            return
        }

        self.transcriptionHandler = transcriptionHandler
        self.isCapturingVoice = true

        audioInputManager.requestPermissions { [weak self] granted in
            guard let self = self else { return }
            if granted {
                do {
                    try self.audioInputManager.startRecording()
                    print("[CactusSTTService] Voice capture started.")
                } catch {
                    print("[CactusSTTService] Error starting audio recording: \(error.localizedDescription)")
                    self.isCapturingVoice = false
                    self.transcriptionHandler?(nil, STTError.audioRecordingFailed(error))
                }
            } else {
                print("[CactusSTTService] Microphone permission denied.")
                self.isCapturingVoice = false
                self.transcriptionHandler?(nil, STTError.permissionDenied)
            }
        }
    }

    public func stopVoiceCapture() {
        guard isCapturingVoice else {
            // Not an error, but good to know.
            print("[CactusSTTService] Not currently capturing voice.")
            return
        }
        print("[CactusSTTService] Stopping voice capture.")
        audioInputManager.stopRecording()
        // Processing will happen in the AudioInputManagerDelegate callback `audioInputManager(_:didCaptureAudioFile:withData:)`
        // Set isCapturingVoice to false there, after processing.
    }

    /// Processes a given audio file for speech-to-text transcription.
    ///
    /// Requires STT to be initialized via `initSTT`.
    /// The actual STT processing (via C FFI) is currently a placeholder.
    /// - Parameters:
    ///   - filePath: The local file system path to the audio file.
    ///   - completion: A closure called with the transcription string or an error.
    public func processAudioFile(filePath: String, completion: @escaping (String?, Error?) -> Void) {
        guard let context = self.sttContext else {
            completion(nil, STTError.notInitialized)
            return
        }
        print("[CactusSTTService] Processing audio file: \(filePath)")

        // Note: The FFI's cactus_stt_process_audio expects raw float samples.
        // This function currently does not read the file and convert it to samples.
        // For now, we will just call getTranscription, assuming audio might have been
        // processed by other means or this is a placeholder for a more complex flow.
        // To truly process a file, one would need to:
        // 1. Read audio file (e.g., using AVFoundation)
        // 2. Convert to required format (PCM 32-bit float, 16kHz, mono)
        // 3. Call cactus_stt_process_audio with the sample buffer
        // 4. Then call cactus_stt_get_transcription

        if let cTranscription = cactus_stt_get_transcription(context) {
            let transcription = String(cString: cTranscription)
            cactus_free_string_c(UnsafeMutablePointer(mutating: cTranscription)) // Free the C string
            print("[CactusSTTService] Transcription retrieved: \(transcription)")
            completion(transcription, nil)
        } else {
            let error = STTError.processingFailed("cactus_stt_get_transcription returned null.")
            print("[CactusSTTService] STT processing error: \(error.localizedDescription)")
            completion(nil, error)
        }
    }

    /// Releases resources used by the STT engine.
    ///
    /// This method should be called when STT functionality is no longer needed.
    /// The actual STT engine resource release (via C FFI) is currently a placeholder.
    /// - Parameter completion: A closure called upon completion, passing an optional `Error` if release failed.
    public func releaseSTT(completion: @escaping (Error?) -> Void) {
        print("[CactusSTTService] Releasing STT resources.")
        if let context = self.sttContext {
            cactus_stt_free(context)
            self.sttContext = nil
            print("[CactusSTTService] STT resources released.")
            completion(nil)
        } else {
            print("[CactusSTTService] STT already released or not initialized.")
            completion(nil) // Or an error if preferred for trying to release a null context
        }
        self.modelPath = nil
    }

    // MARK: - AudioInputManagerDelegate (Public for protocol conformance, internal use)

    /// Handles the `didCaptureAudioFile` callback from `iOSAudioInputManager`.
    /// If currently capturing voice, it proceeds to process the captured audio file for transcription.
    /// - Parameters:
    ///   - manager: The `iOSAudioInputManager` instance.
    ///   - url: The `URL` of the captured audio file.
    ///   - data: Optional `Data` of the audio file.
    public func audioInputManager(_ manager: iOSAudioInputManager, didCaptureAudioFile url: URL, withData data: Data?) {
        print("[CactusSTTService] Audio captured: \(url.path), data size: \(data?.count ?? 0) bytes.")
        guard isCapturingVoice else {
            // This might happen if stopRecording was called, then recording finishes much later.
            // Or if it's a file processed not via start/stop voice capture.
            print("[CactusSTTService] Audio data received but not in voice capturing mode. Ignoring.")
            return
        }

        processAudioFile(filePath: url.path) { [weak self] transcription, error in
            guard let self = self else { return }
            if let error = error {
                print("[CactusSTTService] Error during transcription via delegate: \(error.localizedDescription)")
                self.transcriptionHandler?(nil, error)
            } else if let transcription = transcription {
                print("[CactusSTTService] Transcription received via delegate: \(transcription)")
                self.transcriptionHandler?(transcription, nil)
            } else {
                // Should not happen if error is nil and transcription is nil.
                self.transcriptionHandler?(nil, STTError.unknown("Transcription result was nil with no error."))
            }
            self.isCapturingVoice = false // Reset capturing state
        }
    }

    /// Handles the `didFailWithError` callback from `iOSAudioInputManager`.
    /// If currently capturing voice, it forwards the error to the `transcriptionHandler`.
    /// - Parameters:
    ///   - manager: The `iOSAudioInputManager` instance.
    ///   - error: The `Error` that occurred during audio input.
    public func audioInputManager(_ manager: iOSAudioInputManager, didFailWithError error: Error) {
        print("[CactusSTTService] Audio input manager failed with error: \(error.localizedDescription)")
        if isCapturingVoice {
            self.transcriptionHandler?(nil, STTError.audioRecordingFailed(error))
            self.isCapturingVoice = false // Reset capturing state
        }
        // Otherwise, the error is not related to an active voice capture session initiated by this service.
    }

    // MARK: - User-Specific Adaptation (Placeholder)

    /**
     Sets a user-specific vocabulary string to guide the STT engine.

     This can improve accuracy for uncommon words, names, or specific contexts by providing an "initial prompt" to the underlying Whisper model.

     - Parameter vocabulary: The string containing words or phrases for context.
     - Parameter completion: A callback indicating success or failure.
                             `error` will be non-nil if an issue occurred.
    */
    public func setUserVocabulary(vocabulary: String, completion: @escaping (Error?) -> Void) {
        guard let context = self.sttContext else {
            print("[CactusSTTService] STT context not initialized. Cannot set vocabulary.")
            completion(STTError.notInitialized)
            return
        }

        if let cVocabulary = vocabulary.cString(using: .utf8) {
            cactus_stt_set_user_vocabulary(context, cVocabulary)
            print("[CactusSTTService] User vocabulary set to: \(vocabulary)")
            completion(nil)
        } else {
            print("[CactusSTTService] Error converting vocabulary string to C string.")
            completion(STTError.processingFailed("Failed to convert vocabulary to C string"))
        }
    }
}

/// Enumerates possible errors specific to `CactusSTTService` operations.
public enum STTError: Error, LocalizedError {
    /// STT service was used before `initSTT` was successfully called.
    case notInitialized
    /// `startVoiceCapture` was called while a capture was already in progress.
    case alreadyCapturing
    /// Microphone permission was denied by the user.
    case permissionDenied
    /// The STT engine failed to initialize (e.g., model loading error).
    case initializationFailed(String)
    /// STT processing of an audio file or stream failed.
    case processingFailed(String)
    /// An error occurred during audio recording via `iOSAudioInputManager`.
    case audioRecordingFailed(Error)
    /// An unknown or unspecified STT error occurred.
    case unknown(String)
    /// A specific feature (like setUserVocabulary) is not yet fully implemented.
    case featureNotImplemented(String)

    /// Provides a localized description for each `STTError` case.
    public var errorDescription: String? {
        switch self {
        case .notInitialized: return "STT Service is not initialized. Call initSTT() first."
        case .alreadyCapturing: return "Voice capture is already in progress."
        case .permissionDenied: return "Microphone permission was denied."
        case .initializationFailed(let msg): return "STT engine initialization failed: \(msg)"
        case .processingFailed(let msg): return "STT processing failed: \(msg)"
        case .audioRecordingFailed(let err): return "Audio recording failed: \(err.localizedDescription)"
        case .unknown(let msg): return "An unknown STT error occurred: \(msg)"
        case .featureNotImplemented(let featureName): return "\(featureName) is not yet implemented at the core C++ level."
        }
    }
}
