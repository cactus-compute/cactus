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
    private var isSTTInitialized: Bool = false
    private var isCapturingVoice: Bool = false

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

        // Placeholder for C FFI call
        // let result = cactus_stt_init_ffi((modelPath as NSString).utf8String)
        // if result == 0 { // Assuming 0 is success
        //     self.isSTTInitialized = true
        //     print("[CactusSTTService] STT initialized successfully.")
        //     completion(nil)
        // } else {
        //     let error = STTError.initializationFailed("Failed to initialize STT engine with code \(result).")
        //     print("[CactusSTTService] STT initialization error: \(error.localizedDescription)")
        //     completion(error)
        // }

        // Simulate successful initialization for now
        self.isSTTInitialized = true
        print("[CactusSTTService] STT initialized successfully (Placeholder).")
        completion(nil)
    }

    /// Starts capturing audio from the microphone and processes it for speech-to-text.
    ///
    /// Before starting, it checks if STT is initialized and if another capture is already in progress.
    /// It requests microphone permissions via `iOSAudioInputManager`.
    /// Transcription results or errors are delivered through the `transcriptionHandler`.
    /// - Parameter transcriptionHandler: A closure called with the transcription string or an error.
    public func startVoiceCapture(transcriptionHandler: @escaping (String?, Error?) -> Void) {
        guard isSTTInitialized else {
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
        guard isSTTInitialized else {
            completion(nil, STTError.notInitialized)
            return
        }
        print("[CactusSTTService] Processing audio file: \(filePath)")

        var pathForFFI = filePath
        if filePath.hasPrefix("file://"), let url = URL(string: filePath) {
            pathForFFI = url.path
        }

        // Placeholder for C FFI call
        // let cTranscription = cactus_stt_process_file_ffi((pathForFFI as NSString).utf8String)
        // if let cTrans = cTranscription {
        //     let transcription = String(cString: cTrans)
        //     cactus_stt_free_string_ffi(cTrans) // Important: manage memory from C
        //     print("[CactusSTTService] Transcription successful: \(transcription)")
        //     completion(transcription, nil)
        // } else {
        //     let error = STTError.processingFailed("STT processing returned null.")
        //     print("[CactusSTTService] STT processing error: \(error.localizedDescription)")
        //     completion(nil, error)
        // }

        // Simulate successful processing for now
        let placeholderTranscription = "Placeholder transcription for file: \(pathForFFI)"
        print("[CactusSTTService] Transcription successful (Placeholder): \(placeholderTranscription)")
        completion(placeholderTranscription, nil)
    }

    /// Releases resources used by the STT engine.
    ///
    /// This method should be called when STT functionality is no longer needed.
    /// The actual STT engine resource release (via C FFI) is currently a placeholder.
    /// - Parameter completion: A closure called upon completion, passing an optional `Error` if release failed.
    public func releaseSTT(completion: @escaping (Error?) -> Void) {
        print("[CactusSTTService] Releasing STT resources.")
        // Placeholder for C FFI call
        // cactus_stt_release_ffi()
        self.isSTTInitialized = false
        self.modelPath = nil
        print("[CactusSTTService] STT resources released (Placeholder).")
        completion(nil)
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

    /// Sets user-specific vocabulary to potentially bias the STT engine.
    ///
    /// **Note:** This is currently a placeholder implementation. The actual biasing
    /// of the STT engine via C FFI calls is not yet implemented.
    ///
    /// - Parameters:
    ///   - vocabulary: An array of words or phrases to suggest to the STT engine.
    ///   - completion: A closure called upon completion. Currently always returns `nil` for the error.
    public func setUserVocabulary(vocabulary: [String], completion: @escaping (Error?) -> Void) {
        // TODO: Implement once core C++ FFI functionality is available.
        // This would involve:
        // 1. Converting the [String] to a JSON string.
        // 2. Calling an FFI function like `cactus_stt_set_vocabulary_ffi(self.stt_ctx_pointer, vocabJsonCString)`.
        // For now, just log that it's a placeholder and complete.
        print("[CactusSTTService] setUserVocabulary called with \(vocabulary.count) items. This feature is a placeholder and not yet implemented at the core C++ level.")

        // Optionally, return a specific "not implemented" error if desired:
        // completion(STTError.featureNotImplemented("setUserVocabulary"))
        // For now, completing with nil to indicate "success" for the placeholder.
        completion(nil)
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
