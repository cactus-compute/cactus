import AVFoundation
import Accelerate // For potential format conversions if needed, though AVAudioEngine can often provide desired format

// Protocol for the delegate to receive audio data and errors
protocol AudioInputManagerDelegate: AnyObject {
    func didReceiveAudio(buffer: AVAudioPCMBuffer) // Or provide float array directly
    func didEncounterError(error: Error)
}

class AudioInputManager {

    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode? // Make it optional to handle potential init failures
    private let audioSession = AVAudioSession.sharedInstance() // Get shared session instance

    weak var delegate: AudioInputManagerDelegate?

    private(set) var isRecording = false
    private let recordingQueue = DispatchQueue(label: "com.cactus.audiomanager.recordingQueue", qos: .userInitiated)

    // Desired format for STT
    private let desiredSampleRate: Double = 16000.0
    private let desiredChannels: AVAudioChannelCount = 1 // Mono

    init() {
        // Initialize audioEngine here. If it fails, methods like startRecording will handle it.
        self.audioEngine = AVAudioEngine()
        if let engine = self.audioEngine {
            self.inputNode = engine.inputNode
        } else {
            // This case should ideally not happen unless AVAudioEngine init itself fails, which is rare.
            // Error will be reported during startRecording if engine is nil.
            print("AudioInputManager: AVAudioEngine initialization failed.")
        }
    }

    func checkMicrophonePermission(completion: @escaping (Bool) -> Void) {
        switch audioSession.recordPermission {
        case .granted:
            completion(true)
        case .denied:
            completion(false)
        case .undetermined:
            audioSession.requestRecordPermission { granted in
                // Ensure completion is called on the main thread if UI updates depend on it,
                // but for this manager, direct completion is fine.
                completion(granted)
            }
        @unknown default:
            completion(false)
        }
    }

    func startRecording() {
        recordingQueue.async { [weak self] in // Perform setup and recording on a dedicated queue
            guard let self = self else { return }

            guard !self.isRecording else {
                print("AudioInputManager: Already recording.")
                // Optionally, call delegate with an error:
                // let alreadyRecordingError = NSError(domain: "AudioInputManagerError", code: 100, userInfo: [NSLocalizedDescriptionKey: "Already recording."])
                // self.delegate?.didEncounterError(error: alreadyRecordingError)
                return
            }

            self.checkMicrophonePermission { [weak self] granted in
                guard let self = self else { return }
                // Ensure subsequent operations are on our recordingQueue
                self.recordingQueue.async {
                    if !granted {
                        let permissionError = NSError(domain: "AudioInputManagerError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Microphone permission not granted."])
                        self.delegate?.didEncounterError(error: permissionError)
                        // self.isRecording remains false
                        return
                    }

                    // Configure audio session
                    do {
                        // Set category as early as possible.
                        // Options: .duckOthers is common for recording, .allowBluetoothA2DP might be relevant for some use cases.
                        try self.audioSession.setCategory(.record, mode: .measurement, options: [.duckOthers])
                        try self.audioSession.setPreferredSampleRate(self.desiredSampleRate)
                        try self.audioSession.setActive(true, options: .notifyOthersOnDeactivation)
                    } catch {
                        self.delegate?.didEncounterError(error: error)
                        // self.isRecording remains false
                        return
                    }

                    guard let audioEngine = self.audioEngine, let inputNode = self.inputNode else {
                        let engineError = NSError(domain: "AudioInputManagerError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Audio engine or input node not initialized."])
                        self.delegate?.didEncounterError(error: engineError)
                        self.attemptDeactivateSessionOnError()
                        return
                    }

                    let inputFormat = inputNode.inputFormat(forBus: 0)

                    guard let desiredFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                                            sampleRate: self.desiredSampleRate,
                                                            channels: self.desiredChannels,
                                                            interleaved: false) else {
                        let formatError = NSError(domain: "AudioInputManagerError", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create desired AVAudioFormat."])
                        self.delegate?.didEncounterError(error: formatError)
                        self.attemptDeactivateSessionOnError()
                        return
                    }

                    inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] (buffer, time) in
                        guard let self = self, self.isRecording else { return }

                        if inputFormat.sampleRate != self.desiredSampleRate || inputFormat.channelCount != self.desiredChannels || inputFormat.commonFormat != .pcmFormatFloat32 {
                             // Perform conversion
                            if let converter = AVAudioConverter(from: inputFormat, to: desiredFormat) {
                                // Calculate capacity for the output buffer.
                                // This needs to be proportional to the input buffer's frame length, adjusted for sample rate differences.
                                let outputFrameCapacity = AVAudioFrameCount(desiredFormat.sampleRate * Double(buffer.frameLength) / inputFormat.sampleRate)
                                guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: desiredFormat, frameCapacity: outputFrameCapacity) else {
                                    // Log error, potentially call delegate. This can be noisy.
                                    print("AudioInputManager: Failed to create conversion buffer.")
                                    return
                                }

                                var error: NSError?
                                // Input block for the converter
                                let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
                                    outStatus.pointee = .haveData
                                    return buffer
                                }

                                // Perform the conversion
                                converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)

                                if let error = error {
                                    // Log error, potentially call delegate. This can be noisy.
                                     print("AudioInputManager: Error during audio conversion: \(error.localizedDescription)")
                                    return
                                }
                                // Successfully converted, pass the converted buffer to the delegate
                                self.delegate?.didReceiveAudio(buffer: convertedBuffer)
                            } else {
                                // Log error, potentially call delegate.
                                print("AudioInputManager: Failed to create AVAudioConverter.")
                            }
                        } else {
                            // Format is already correct, pass the original buffer
                            self.delegate?.didReceiveAudio(buffer: buffer)
                        }
                    }

                    do {
                        audioEngine.prepare() // Prepare the engine
                        try audioEngine.start()
                        self.isRecording = true
                        print("AudioInputManager: Recording started.")
                    } catch {
                        self.delegate?.didEncounterError(error: error)
                        // self.isRecording remains false
                        inputNode.removeTap(onBus: 0) // Clean up tap
                        self.attemptDeactivateSessionOnError()
                    }
                }
            }
        }
    }

    func stopRecording() {
        recordingQueue.async { [weak self] in
            guard let self = self else { return }

            guard self.isRecording else {
                print("AudioInputManager: Not recording or already stopped.")
                return
            }

            self.isRecording = false // Set this early to stop data processing in tap

            if let audioEngine = self.audioEngine, audioEngine.isRunning {
                 audioEngine.stop()
            }
            if let inputNode = self.inputNode {
                 inputNode.removeTap(onBus: 0)
            }

            // Deactivate audio session
            self.attemptDeactivateSessionOnError() // Also used for regular stop
            print("AudioInputManager: Recording stopped.")
        }
    }

    private func attemptDeactivateSessionOnError() {
        // This function is called on the recordingQueue
        do {
            try self.audioSession.setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            // self.delegate?.didEncounterError(error: error) // Usually not critical to report session deactivation failure
            print("AudioInputManager: Failed to deactivate audio session: \(error.localizedDescription)")
        }
    }

    func prepareForDeinit() {
        // Ensure this is called from a context that doesn't conflict with recordingQueue,
        // or make stopRecording synchronous if needed (with care for deadlocks).
        // This is a blocking call to ensure cleanup before deinit.
        recordingQueue.sync {
            if self.isRecording {
                self.isRecording = false // Prevent further tap processing

                if let audioEngine = self.audioEngine, audioEngine.isRunning {
                    audioEngine.stop()
                }
                if let inputNode = self.inputNode {
                    inputNode.removeTap(onBus: 0)
                }
                self.attemptDeactivateSessionOnError() // Deactivate session
            }
            // Release the engine itself
            self.audioEngine = nil
            self.inputNode = nil
            print("AudioInputManager: Cleaned up for deinit.")
        }
    }

    deinit {
        print("AudioInputManager deinit")
        // Fallback cleanup, though explicit prepareForDeinit is preferred.
        // This deinit might be on any thread. If it's on recordingQueue, sync call to it would deadlock.
        // If on main thread, sync call to background queue is okay.
        // For safety, ensure isRecording is false, but rely on prepareForDeinit for actual resource release.
        if isRecording {
            print("AudioInputManager deinit: Still marked as recording. Explicit prepareForDeinit() was likely not called.")
            // Avoid direct resource manipulation here if it's not thread-safe without the queue.
            // The queue itself might be gone if this is a late deinit.
        }
    }
}
