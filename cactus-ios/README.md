# Cactus Native iOS SDK

This SDK provides native iOS capabilities for running AI models on-device, including Speech-to-Text (STT) functionality.

## Features

- Speech-to-Text (STT) using `CactusSTTService`.
- (Future: Other on-device AI functionalities from the Cactus C++ core).

## Installation

1.  **Add XCFramework**: Ensure `cactus.xcframework` is linked to your Xcode project. This XCFramework contains the core C++ logic and necessary FFI functions.
2.  **Add Swift Wrappers**: Include the Swift wrapper files (e.g., `CactusSTTService.swift`, `iOSAudioInputManager.swift`, and the bridging header `CactusSDK-Bridging-Header.h`) in your project from the `Cactus/` directory.
3.  **Bridging Header**: Configure your Xcode project's build settings to use the `CactusSDK-Bridging-Header.h`. Go to "Build Settings" -> "Swift Compiler - General" -> "Objective-C Bridging Header" and set the path to `YourProjectName/PathTo/CactusSDK-Bridging-Header.h`. This header should import the C FFI headers from `cactus.xcframework` (e.g., `#import <cactus/cactus_ffi.h>`).

## Voice-to-Text (STT)

The `CactusSTTService` class provides an interface for speech-to-text functionality using the device microphone.

### Basic STT Usage

```swift
import UIKit
import Cactus // Your module name for the SDK wrappers

class MyViewController: UIViewController {

    var cactusSTTService: CactusSTTService?
    var isRecording = false

    // Example UI elements (you would connect these to your Storyboard or create them programmatically)
    var recordButton: UIButton!
    var transcriptionTextView: UITextView!
    var statusLabel: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()
        setupMyUI() // Your UI setup

        initializeSTT()
    }

    func initializeSTT() {
        self.cactusSTTService = CactusSTTService()

        // TODO: Replace with the actual path to your STT model file.
        // This model should be bundled with your app or downloaded to a known location.
        // For example, if bundled:
        guard let modelPath = Bundle.main.path(forResource: "your_stt_model_filename", ofType: "bin") else {
            print("STT Model not found in bundle.")
            statusLabel.text = "Error: STT Model not found."
            recordButton.isEnabled = false
            return
        }

        statusLabel.text = "Initializing STT..."
        cactusSTTService?.initSTT(modelPath: modelPath) { [weak self] error in
            DispatchQueue.main.async {
                guard let self = self else { return }
                if let error = error {
                    self.statusLabel.text = "STT Init Error: \(error.localizedDescription)"
                    self.recordButton.isEnabled = false
                    print("STT Initialization Error: \(error.localizedDescription)")
                } else {
                    self.statusLabel.text = "STT Initialized. Ready."
                    self.recordButton.isEnabled = true
                    print("STT Initialized Successfully")

                    // Optional: Set user-specific vocabulary (currently a placeholder)
                    self.cactusSTTService?.setUserVocabulary(vocabulary: ["custom word", "Cactus AI"]) { vocabError in
                        if let vocabError = vocabError {
                            print("Error setting vocab (placeholder): \(vocabError.localizedDescription)")
                        } else {
                            print("User vocabulary set (placeholder).")
                        }
                    }
                }
            }
        }
    }

    @objc func recordButtonPressed() {
        guard let sttService = cactusSTTService else {
            statusLabel.text = "STT Service not available."
            return
        }

        if isRecording {
            sttService.stopVoiceCapture()
            recordButton.setTitle("Start Recording", for: .normal)
            statusLabel.text = "Stopping... Processing..."
            // isRecording will be set to false by the STTService's completion handler
        } else {
            // Request microphone permission first (CactusSTTService handles this internally via iOSAudioInputManager)
            // or you can use AVAudioSession.sharedInstance().requestRecordPermission directly here if preferred.

            statusLabel.text = "Recording..."
            recordButton.setTitle("Stop Recording", for: .normal)
            transcriptionTextView.text = "" // Clear previous transcription
            isRecording = true

            sttService.startVoiceCapture { [weak self] transcription, error in
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    self.isRecording = false // Reset recording state
                    self.recordButton.setTitle("Start Recording", for: .normal)

                    if let error = error {
                        self.statusLabel.text = "STT Error: \(error.localizedDescription)"
                        self.transcriptionTextView.text = "Error: \(error.localizedDescription)"
                        print("Transcription Error: \(error.localizedDescription)")
                    } else if let transcription = transcription {
                        self.statusLabel.text = "Transcription received."
                        self.transcriptionTextView.text = transcription
                        print("Transcription: \(transcription)")
                    } else {
                        self.statusLabel.text = "Transcription complete (no text/error)."
                    }
                }
            }
        }
    }

    // Example for processing a pre-existing audio file
    func transcribeAudioFile(filePath: String) {
        guard let sttService = cactusSTTService else { return }

        // Ensure STT is initialized (e.g., check a flag set in initSTT completion)
        // if !sttService.isInitialized { await initSTTEngine(); }

        sttService.processAudioFile(filePath: filePath) { [weak self] (transcription, error) in
            DispatchQueue.main.async {
                if let error = error {
                    self?.transcriptionTextView.text = "File STT Error: \(error.localizedDescription)"
                } else if let transcription = transcription {
                    self?.transcriptionTextView.text = "File Transcription: \(transcription)"
                }
            }
        }
    }

    deinit {
        cactusSTTService?.releaseSTT { error in
            if let error = error { print("Error releasing STT: \(error.localizedDescription)") }
            else { print("STT resources released.") }
        }
    }

    func setupMyUI() { /* ... Your UI setup code ... */ }
}
```

### Required Permissions

**`Info.plist`:**
You must add the `NSMicrophoneUsageDescription` key to your app's `Info.plist` file. Provide a string that explains to the user why your app needs access to the microphone.

Example:
```xml
<key>NSMicrophoneUsageDescription</key>
<string>This app uses the microphone to capture your voice for transcription with the Cactus STT service.</string>
```

The `iOSAudioInputManager` (used by `CactusSTTService`) will request microphone permission when `startVoiceCapture` is called for the first time. Ensure your app handles cases where permission might be denied.

## (Future) Other SDK Features

Details on other AI functionalities will be added here as they become available in the native iOS SDK.
