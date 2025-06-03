// iOSAudioInputManager.swift
import Foundation
import AVFoundation

/// Delegate protocol for `iOSAudioInputManager` to communicate recording events.
public protocol AudioInputManagerDelegate: AnyObject {
    /// Called when audio recording finishes successfully and the audio file is available.
    /// - Parameters:
    ///   - manager: The `iOSAudioInputManager` instance that handled the recording.
    ///   - url: The `URL` of the recorded audio file.
    ///   - data: Optional `Data` object containing the audio file's contents. This is provided for flexibility.
    func audioInputManager(_ manager: iOSAudioInputManager, didCaptureAudioFile url: URL, withData data: Data?)

    /// Called when an error occurs during the audio recording process.
    /// - Parameters:
    ///   - manager: The `iOSAudioInputManager` instance.
    ///   - error: The `Error` that occurred.
    func audioInputManager(_ manager: iOSAudioInputManager, didFailWithError error: Error)
}

/// Manages audio input using `AVFoundation`, including requesting permissions,
/// starting/stopping recording, and handling delegate callbacks for recording events.
public class iOSAudioInputManager: NSObject, AVAudioRecorderDelegate {

    private var audioRecorder: AVAudioRecorder?
    private var currentRecordingURL: URL?

    /// The delegate to receive audio recording event callbacks.
    public weak var delegate: AudioInputManagerDelegate?

    /// Initializes a new `iOSAudioInputManager`.
    public override init() {
        super.init()
    }

    /// Requests permission from the user to access the microphone.
    ///
    /// The completion handler is called asynchronously on the main thread.
    /// - Parameter completion: A closure that takes a `Bool` indicating whether permission was granted.
    public func requestPermissions(completion: @escaping (Bool) -> Void) {
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            DispatchQueue.main.async {
                completion(granted)
            }
        }
    }

    /// Starts audio recording and saves it to a file.
    ///
    /// This method configures the `AVAudioSession`, sets up the `AVAudioRecorder`,
    /// and begins recording. Errors during setup or recording start will be thrown.
    /// Recording completion or errors are communicated via the `AudioInputManagerDelegate`.
    /// - Parameter fileName: The desired file name for the recording (default is "audioRecording.m4a").
    ///   The file will be saved in the app's documents directory.
    /// - Throws: `AudioInputError` if session setup, file creation, or recorder setup fails.
    public func startRecording(fileName: String = "audioRecording.m4a") throws {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .default, options: [.duckOthers, .defaultToSpeaker])
            try audioSession.setActive(true)
        } catch {
            throw AudioInputError.sessionSetupFailed(error)
        }

        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        currentRecordingURL = documentsPath.appendingPathComponent(fileName)

        guard let currentRecordingURL = currentRecordingURL else {
            throw AudioInputError.fileCreationFailed("Could not create URL for recording.")
        }

        // Ensure directory exists
        let directory = currentRecordingURL.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
        }


        let settings = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 16000, // Common sample rate for STT
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.medium.rawValue // Medium for smaller file size, High for better quality
        ]

        do {
            audioRecorder = try AVAudioRecorder(url: currentRecordingURL, settings: settings)
            audioRecorder?.delegate = self
            audioRecorder?.isMeteringEnabled = true // Optional: for audio level metering
            if !(audioRecorder?.record() ?? false) {
                 throw AudioInputError.recordingFailed("Failed to start AVRecorder recording.")
            }
        } catch {
            self.currentRecordingURL = nil
            throw AudioInputError.recorderSetupFailed(error)
        }
    }

    public func stopRecording() {
        guard let recorder = audioRecorder, recorder.isRecording else {
            // Not an error if already stopped, but can notify if needed
            // delegate?.audioInputManager(self, didFailWithError: AudioInputError.notRecording)
            return
        }
        recorder.stop()
        // `audioRecorderDidFinishRecording` will be called by the delegate method
    }

    // AVAudioRecorderDelegate methods
    public func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setActive(false)
        } catch {
            // Log this error but don't necessarily make it fatal for the recording process itself
            print("[iOSAudioInputManager] Error deactivating audio session: \(error.localizedDescription)")
        }

        guard let url = currentRecordingURL else {
            delegate?.audioInputManager(self, didFailWithError: AudioInputError.fileNotFound)
            cleanup()
            return
        }

        if flag {
            // Optionally read data here if needed by delegate, or let delegate read from URL
            var audioData: Data? = nil
            do {
                audioData = try Data(contentsOf: url)
                if audioData?.isEmpty ?? true {
                     delegate?.audioInputManager(self, didFailWithError: AudioInputError.fileEmpty)
                     cleanup()
                     return
                }
            } catch {
                 delegate?.audioInputManager(self, didFailWithError: AudioInputError.fileReadFailed(error))
                 cleanup()
                 return
            }
            delegate?.audioInputManager(self, didCaptureAudioFile: url, withData: audioData)
        } else {
            delegate?.audioInputManager(self, didFailWithError: AudioInputError.recordingFailed("Recording finished unsuccessfully."))
        }
        cleanup()
    }

    // MARK: - AVAudioRecorderDelegate (Public for protocol conformance, internal use)

    /// Delegate method called by `AVAudioRecorder` when recording finishes.
    /// This method deactivates the audio session and informs the `AudioInputManagerDelegate`.
    /// - Parameters:
    ///   - recorder: The `AVAudioRecorder` instance.
    ///   - flag: `true` if recording finished successfully, `false` otherwise.
    public func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setActive(false)
        } catch {
            // Log this error but don't necessarily make it fatal for the recording process itself
            print("[iOSAudioInputManager] Error deactivating audio session: \(error.localizedDescription)")
        }

        guard let url = currentRecordingURL else {
            delegate?.audioInputManager(self, didFailWithError: AudioInputError.fileNotFound)
            cleanup()
            return
        }

        if flag {
            var audioData: Data? = nil
            do {
                audioData = try Data(contentsOf: url)
                if audioData?.isEmpty ?? true {
                     delegate?.audioInputManager(self, didFailWithError: AudioInputError.fileEmpty)
                     cleanup()
                     return
                }
            } catch {
                 delegate?.audioInputManager(self, didFailWithError: AudioInputError.fileReadFailed(error))
                 cleanup()
                 return
            }
            delegate?.audioInputManager(self, didCaptureAudioFile: url, withData: audioData)
        } else {
            delegate?.audioInputManager(self, didFailWithError: AudioInputError.recordingFailed("Recording finished unsuccessfully."))
        }
        cleanup()
    }

    /// Delegate method called by `AVAudioRecorder` if an encoding error occurs.
    /// - Parameters:
    ///   - recorder: The `AVAudioRecorder` instance.
    ///   - error: The `Error` that occurred during encoding.
    public func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        if let error = error {
            delegate?.audioInputManager(self, didFailWithError: AudioInputError.encodingFailed(error))
        } else {
            delegate?.audioInputManager(self, didFailWithError: AudioInputError.unknown("Encoding failed with unknown error."))
        }
        cleanup()
    }

    private func cleanup() {
        audioRecorder = nil
        currentRecordingURL = nil
    }
}

/// Enumerates possible errors that can occur during audio input operations.
public enum AudioInputError: Error, LocalizedError {
    /// Error during `AVAudioSession` setup (e.g., setting category or activating).
    case sessionSetupFailed(Error)
    /// Failed to create the URL or path for the audio recording file.
    case fileCreationFailed(String)
    /// The recorded audio file was not found at the expected path.
    case fileNotFound
    /// The recorded audio file was found but is empty.
    case fileEmpty
    /// Failed to read data from the recorded audio file.
    case fileReadFailed(Error)
    /// Error during `AVAudioRecorder` setup.
    case recorderSetupFailed(Error)
    /// General recording failure (e.g., `AVAudioRecorder.record()` returned false or finished unsuccessfully).
    case recordingFailed(String)
    /// An attempt was made to stop recording, but no recording was in progress.
    case notRecording
    /// An error occurred during audio encoding.
    case encodingFailed(Error)
    /// An unknown or unspecified error occurred.
    case unknown(String)

    /// Provides a localized description for each `AudioInputError` case.
    public var errorDescription: String? {
        switch self {
        case .sessionSetupFailed(let err): return "Audio session setup failed: \(err.localizedDescription)"
        case .fileCreationFailed(let msg): return "File creation failed: \(msg)"
        case .fileNotFound: return "Recorded audio file not found."
        case .fileEmpty: return "Recorded audio file is empty."
        case .fileReadFailed(let err): return "Failed to read audio file: \(err.localizedDescription)"
        case .recorderSetupFailed(let err): return "Audio recorder setup failed: \(err.localizedDescription)"
        case .recordingFailed(let msg): return "Audio recording failed: \(msg)"
        case .notRecording: return "No recording is currently in progress."
        case .encodingFailed(let err): return "Audio encoding failed: \(err.localizedDescription)"
        case .unknown(let msg): return "An unknown audio input error occurred: \(msg)"
        }
    }
}
