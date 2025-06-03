package com.cactus.cactus_flutter;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.util.Log;

import com.cactus.cactus_sdk.audio.AudioInputModule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.embedding.engine.plugins.activity.ActivityAware;
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding;
import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.plugin.common.EventChannel;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.PluginRegistry;

public class CactusFlutterPlugin implements FlutterPlugin, ActivityAware, MethodChannel.MethodCallHandler, PluginRegistry.RequestPermissionsResultListener {

    private static final String TAG = "CactusFlutterPlugin";
    private static final String AUDIO_METHOD_CHANNEL_NAME = "com.cactus.sdk/audio_input_methods";
    private static final String AUDIO_EVENT_CHANNEL_NAME = "com.cactus.sdk/audio_input_events";
    private static final int RECORD_AUDIO_PERMISSION_REQUEST_CODE = 1001;

    private MethodChannel audioMethodChannel;
    private EventChannel audioEventChannel;
    private AudioInputModule audioInputModule;
    private EventChannel.EventSink audioEventSink;

    private Context applicationContext;
    private Activity activity;

    // To store pending result for permission request
    private MethodChannel.Result pendingPermissionResult;

    @Override
    public void onAttachedToEngine(@NonNull FlutterPluginBinding binding) {
        Log.d(TAG, "onAttachedToEngine");
        applicationContext = binding.getApplicationContext();
        BinaryMessenger messenger = binding.getBinaryMessenger();

        audioMethodChannel = new MethodChannel(messenger, AUDIO_METHOD_CHANNEL_NAME);
        audioMethodChannel.setMethodCallHandler(this);

        audioEventChannel = new EventChannel(messenger, AUDIO_EVENT_CHANNEL_NAME);
        audioEventChannel.setStreamHandler(audioStreamHandler);

        // Initialize AudioInputModule listener here as it's tied to the event sink
        // which is set up when Dart listens to the EventChannel.
    }

    @Override
    public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
        Log.d(TAG, "onDetachedFromEngine");
        if (audioInputModule != null) {
            audioInputModule.stopRecording(); // Ensure recording is stopped
            audioInputModule = null;
        }
        audioMethodChannel.setMethodCallHandler(null);
        audioEventChannel.setStreamHandler(null);
        applicationContext = null;
    }

    // ActivityAware methods
    @Override
    public void onAttachedToActivity(@NonNull ActivityPluginBinding binding) {
        Log.d(TAG, "onAttachedToActivity");
        activity = binding.getActivity();
        binding.addRequestPermissionsResultListener(this);
    }

    @Override
    public void onDetachedFromActivity() {
        Log.d(TAG, "onDetachedFromActivity");
        activity = null;
        // If audioInputModule holds activity context, it might need cleanup or a new activity ref.
    }

    @Override
    public void onReattachedToActivityForConfigChanges(@NonNull ActivityPluginBinding binding) {
        Log.d(TAG, "onReattachedToActivityForConfigChanges");
        onAttachedToActivity(binding);
    }

    @Override
    public void onDetachedFromActivityForConfigChanges() {
        Log.d(TAG, "onDetachedFromActivityForConfigChanges");
        onDetachedFromActivity();
    }

    // MethodCallHandler
    @Override
    public void onMethodCall(@NonNull MethodCall call, @NonNull MethodChannel.Result result) {
        Log.d(TAG, "onMethodCall: " + call.method);
        switch (call.method) {
            case "hasPermission":
                result.success(AudioInputModule.hasMicrophonePermission(applicationContext));
                break;
            case "requestPermission":
                handleRequestPermission(result);
                break;
            case "startRecording":
                handleStartRecording(result);
                break;
            case "stopRecording":
                handleStopRecording(result);
                break;
            default:
                result.notImplemented();
        }
    }

    private void handleRequestPermission(@NonNull MethodChannel.Result result) {
        if (AudioInputModule.hasMicrophonePermission(applicationContext)) {
            result.success(true);
            return;
        }
        if (activity != null) {
            if (pendingPermissionResult != null) {
                // Another request is pending, complete the old one with a temporary failure.
                pendingPermissionResult.error("CONCURRENT_REQUEST", "A new permission request was made before the previous one completed.", null);
            }
            pendingPermissionResult = result;
            ActivityCompat.requestPermissions(activity,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    RECORD_AUDIO_PERMISSION_REQUEST_CODE);
            // The result will be sent in onRequestPermissionsResult
        } else {
            result.error("NO_ACTIVITY", "Activity not available to request permissions.", null);
        }
    }

    private void handleStartRecording(@NonNull MethodChannel.Result result) {
        if (audioInputModule == null) {
            audioInputModule = new AudioInputModule(audioDataListener);
        }
        // Check permission again before starting, in case it was revoked or never granted.
        if (!AudioInputModule.hasMicrophonePermission(applicationContext)) {
            result.error("NO_PERMISSION", "RECORD_AUDIO permission not granted.", null);
            return;
        }
        try {
            audioInputModule.startRecording(applicationContext); // Context for permission check and AudioRecord
            result.success(true);
        } catch (Exception e) {
            Log.e(TAG, "Error starting recording: " + e.getMessage(), e);
            result.error("START_FAILED", "Failed to start recording: " + e.getMessage(), null);
        }
    }

    private void handleStopRecording(@NonNull MethodChannel.Result result) {
        if (audioInputModule != null) {
            try {
                audioInputModule.stopRecording();
                result.success(true);
            } catch (Exception e) {
                Log.e(TAG, "Error stopping recording: " + e.getMessage(), e);
                result.error("STOP_FAILED", "Failed to stop recording: " + e.getMessage(), null);
            }
        } else {
            result.success(true); // Nothing to stop
        }
    }

    // AudioDataListener implementation
    private final AudioInputModule.AudioDataListener audioDataListener = new AudioInputModule.AudioDataListener() {
        @Override
        public void onAudioDataReceived(float[] audioData) {
            if (audioEventSink != null) {
                // Convert float[] to List<Double> for Flutter, as platform channels handle List<Double> well.
                // Direct Float32List might be more efficient if Dart side is set up for it.
                List<Double> doubleList = new ArrayList<>(audioData.length);
                for (float f : audioData) {
                    doubleList.add((double) f);
                }
                // Ensure this is run on the main thread if UI updates are directly tied to it,
                // but EventSink.success itself is thread-safe.
                // For audio data, it's usually fine to send from background thread.
                audioEventSink.success(doubleList);
            }
        }

        @Override
        public void onError(String errorMessage) {
            if (audioEventSink != null) {
                audioEventSink.error("AUDIO_ERROR", errorMessage, null);
            }
        }
    };

    // EventChannel.StreamHandler implementation
    private final EventChannel.StreamHandler audioStreamHandler = new EventChannel.StreamHandler() {
        @Override
        public void onListen(Object arguments, EventChannel.EventSink events) {
            Log.d(TAG, "EventChannel: onListen");
            audioEventSink = events;
            // If auto-start recording on listen is desired, it could be triggered here.
            // However, explicit startRecording call from Dart is generally better.
        }

        @Override
        public void onCancel(Object arguments) {
            Log.d(TAG, "EventChannel: onCancel");
            audioEventSink = null;
            // Consider stopping recording if Dart cancels the stream and recording is active.
            // This depends on desired plugin behavior.
            // if (audioInputModule != null && audioInputModule.isRecording()) {
            //     audioInputModule.stopRecording();
            // }
        }
    };

    // PluginRegistry.RequestPermissionsResultListener
    @Override
    public boolean onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == RECORD_AUDIO_PERMISSION_REQUEST_CODE) {
            if (pendingPermissionResult != null) {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    pendingPermissionResult.success(true);
                } else {
                    pendingPermissionResult.success(false); // Or error out: pendingPermissionResult.error("PERMISSION_DENIED", "Microphone permission denied.", null);
                }
                pendingPermissionResult = null; // Clear the pending result
            }
            return true; // Handled
        }
        return false; // Not handled
    }
}
