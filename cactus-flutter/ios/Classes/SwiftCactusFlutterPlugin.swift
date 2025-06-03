import Flutter
import UIKit
import AVFoundation // For AVAudioSession and permission checks

// Helper extension to convert UnsafeBufferPointer to Array
extension UnsafeBufferPointer {
    func toArray() -> [Element] {
        return Array(self)
    }
}

public class SwiftCactusFlutterPlugin: NSObject, FlutterPlugin, AudioInputManagerDelegate, FlutterStreamHandler {

    private static let AUDIO_METHOD_CHANNEL_NAME = "com.cactus.sdk/audio_input_methods"
    private static let AUDIO_EVENT_CHANNEL_NAME = "com.cactus.sdk/audio_input_events"

    private var audioMethodChannel: FlutterMethodChannel?
    private var audioEventChannel: FlutterEventChannel?
    private var audioInputManager: AudioInputManager?
    private var audioEventSink: FlutterEventSink?

    // To hold the FlutterResult for async permission request, if a more complex flow was needed.
    // For this implementation, checkMicrophonePermission's callback handles the result directly.
    // private var pendingPermissionResult: FlutterResult?

    public static func register(with registrar: FlutterPluginRegistrar) {
        let instance = SwiftCactusFlutterPlugin()
        let messenger = registrar.messenger()

        instance.audioMethodChannel = FlutterMethodChannel(name: AUDIO_METHOD_CHANNEL_NAME, binaryMessenger: messenger)
        instance.audioMethodChannel?.setMethodCallHandler(instance.handleAudioMethodCall)

        instance.audioEventChannel = FlutterEventChannel(name: AUDIO_EVENT_CHANNEL_NAME, binaryMessenger: messenger)
        instance.audioEventChannel?.setStreamHandler(instance) // The plugin itself will handle stream callbacks

        // `registrar.addApplicationDelegate(instance)` could be used if specific app lifecycle events needed handling here.
        // For this audio plugin, direct app delegate methods are not strictly necessary unless handling interruptions, etc.

        // Initialize AudioInputManager lazily or here if preferred.
        // instance.audioInputManager = AudioInputManager()
        // instance.audioInputManager?.delegate = instance // Set delegate if initialized here
    }

    // Detach and cleanup
    public func detachFromEngine(for registrar: FlutterPluginRegistrar) {
        print("SwiftCactusFlutterPlugin: detachFromEngine")
        audioMethodChannel?.setMethodCallHandler(nil)
        audioEventChannel?.setStreamHandler(nil)

        audioInputManager?.prepareForDeinit() // Ensure AudioInputManager cleans up
        audioInputManager = nil

        audioEventSink = nil
    }

    // Lazy initializer for AudioInputManager
    private func ensureAudioInputManagerInitialized() {
        if audioInputManager == nil {
            audioInputManager = AudioInputManager()
            audioInputManager?.delegate = self
        }
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        // This is the main handler if the plugin itself is the MethodCallHandler for a general channel.
        // For this task, we are using a specific channel handler.
        // This function can be removed if not conforming to a global FlutterPlugin MethodCallHandler.
        result(FlutterMethodNotImplemented)
    }

    public func handleAudioMethodCall(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        ensureAudioInputManagerInitialized() // Ensure manager is ready

        switch call.method {
        case "hasPermission":
            audioInputManager?.checkMicrophonePermission { granted in
                result(granted)
            }
        case "requestPermission":
            // AudioInputManager's checkMicrophonePermission already handles request if undetermined.
            audioInputManager?.checkMicrophonePermission { granted in
                // 'granted' here will be the result after the user responds to a system prompt if one was shown,
                // or the current status if already determined.
                result(granted)
            }
        case "startRecording":
            // Delegate should already be set by ensureAudioInputManagerInitialized
            audioInputManager?.startRecording()
            // Note: startRecording in AudioInputManager is async.
            // We return success immediately, actual status/errors come via EventChannel.
            result(true)
        case "stopRecording":
            audioInputManager?.stopRecording()
            result(true)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // MARK: - AudioInputManagerDelegate

    public func didReceiveAudio(buffer: AVAudioPCMBuffer) {
        guard let eventSink = self.audioEventSink else { return }

        // Extract Float data and convert to [Double] for Flutter
        // This assumes the buffer from AudioInputManager is already in the desired
        // format (Float32, 16kHz, mono) due to conversion within AudioInputManager.
        guard let channelData = buffer.floatChannelData else {
            print("AudioInputManagerDelegate: Failed to get floatChannelData.")
            return
        }

        // Assuming mono audio as per desired format in AudioInputManager
        // If stereo, this would need adjustment or send interleaved.
        let frameLength = Int(buffer.frameLength)
        let dataPointer = channelData[0] // Pointer to the Float data for the first (and only) channel

        var doubleArray: [Double] = []
        doubleArray.reserveCapacity(frameLength)

        for i in 0..<frameLength {
            doubleArray.append(Double(dataPointer[i]))
        }

        eventSink(doubleArray)
    }

    public func didEncounterError(error: Error) {
        guard let eventSink = self.audioEventSink else { return }
        eventSink(FlutterError(code: "AUDIO_INPUT_ERROR",
                               message: error.localizedDescription,
                               details: nil)) // Can add more details if needed
    }

    // MARK: - FlutterStreamHandler (for Audio Event Channel)

    public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        print("SwiftCactusFlutterPlugin: EventChannel onListen")
        self.audioEventSink = events
        // It might be desirable to check for permissions here or when startRecording is called.
        // If AudioInputManager is set up to auto-start on listen, that logic would go here.
        // For this plugin, start/stop is explicit via MethodChannel.
        return nil
    }

    public func onCancel(withArguments arguments: Any?) -> FlutterError? {
        print("SwiftCactusFlutterPlugin: EventChannel onCancel")
        self.audioEventSink = nil
        // If recording should stop when Dart cancels the stream:
        // audioInputManager?.stopRecording()
        // This depends on the desired behavior of the plugin.
        return nil
    }
}
