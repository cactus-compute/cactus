package com.cactusreact.audio;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.core.DeviceEventManagerModule;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

// Assuming AudioInputModule is accessible.
// This might be from a shared library, or its source adapted into this project.
// For this task, we use the defined interface and methods of AudioInputModule.
import com.cactus.cactus_sdk.audio.AudioInputModule;

// It's good practice to also import android.content.Context for clarity
import android.content.Context;
import android.util.Log;

public class RNAudioInputModule extends ReactContextBaseJavaModule implements AudioInputModule.AudioDataListener {

    private static final String MODULE_NAME = "RNAudioInput";
    private static final String TAG = "RNAudioInputModule"; // For logging

    private AudioInputModule audioInputModule;
    private final ReactApplicationContext reactContext;

    public RNAudioInputModule(ReactApplicationContext reactContext) {
        super(reactContext);
        this.reactContext = reactContext;
    }

    @NonNull
    @Override
    public String getName() {
        return MODULE_NAME;
    }

    private void sendEvent(String eventName, @Nullable Object params) {
        try {
            if (this.reactContext.hasActiveCatalystInstance()) {
                this.reactContext
                    .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
                    .emit(eventName, params);
            } else {
                Log.w(TAG, "Tried to send event " + eventName + " without an active Catalyst instance.");
            }
        } catch (Exception e) {
            Log.e(TAG, "Error sending event " + eventName + ": " + e.getMessage());
        }
    }

    @ReactMethod
    public void hasPermission(Promise promise) {
        try {
            // Use application context for permission check as it doesn't require an Activity
            // and AudioInputModule.hasMicrophonePermission is static.
            boolean hasPerm = AudioInputModule.hasMicrophonePermission(this.reactContext.getApplicationContext());
            promise.resolve(hasPerm);
        } catch (Exception e) {
            Log.e(TAG, "Error in hasPermission: " + e.getMessage(), e);
            promise.reject("PERMISSION_ERROR", "Error checking microphone permission: " + e.getMessage(), e);
        }
    }

    @ReactMethod
    public void requestPermission(Promise promise) {
        // For React Native, PermissionsAndroid.request() from the JS side is the standard way.
        // This native method will simply check and report current status.
        // If not granted, JS side should then use PermissionsAndroid.request().
        try {
            boolean hasPerm = AudioInputModule.hasMicrophonePermission(this.reactContext.getApplicationContext());
            promise.resolve(hasPerm);
            // If !hasPerm, the JS side should initiate PermissionsAndroid.request().
            // We don't trigger the OS permission dialog directly from here to align with RN best practices.
        } catch (Exception e) {
            Log.e(TAG, "Error in requestPermission (check status): " + e.getMessage(), e);
            promise.reject("PERMISSION_CHECK_ERROR", "Error checking microphone permission status: " + e.getMessage(), e);
        }
    }

    @ReactMethod
    public void startRecording(Promise promise) {
        try {
            Context currentContext = this.reactContext.getApplicationContext();
             // Check permission again before starting
            if (!AudioInputModule.hasMicrophonePermission(currentContext)) {
                promise.reject("NO_PERMISSION", "Microphone permission not granted.");
                return;
            }

            if (audioInputModule == null) {
                // 'this' is the AudioDataListener for the AudioInputModule instance
                audioInputModule = new AudioInputModule(this);
            }

            // AudioInputModule's startRecording needs a Context. ApplicationContext is fine here.
            audioInputModule.startRecording(currentContext);
            promise.resolve(true);
            Log.d(TAG, "Audio recording started via RNAudioInputModule.");
        } catch (Exception e) {
            Log.e(TAG, "Error in startRecording: " + e.getMessage(), e);
            promise.reject("START_ERROR", "Failed to start recording: " + e.getMessage(), e);
        }
    }

    @ReactMethod
    public void stopRecording(Promise promise) {
        try {
            if (audioInputModule != null) {
                audioInputModule.stopRecording();
                Log.d(TAG, "Audio recording stopped via RNAudioInputModule.");
            }
            promise.resolve(true);
        } catch (Exception e) {
            Log.e(TAG, "Error in stopRecording: " + e.getMessage(), e);
            promise.reject("STOP_ERROR", "Failed to stop recording: " + e.getMessage(), e);
        }
    }

    // Implementation of AudioInputModule.AudioDataListener
    @Override
    public void onAudioDataReceived(float[] audioData) {
        WritableArray dataArray = Arguments.createArray();
        for (float val : audioData) {
            dataArray.pushDouble((double)val); // Convert float to double for React Native bridge
        }
        sendEvent("onAudioData", dataArray);
    }

    @Override
    public void onError(String errorMessage) {
        WritableMap errorMap = Arguments.createMap();
        errorMap.putString("message", errorMessage);
        sendEvent("onAudioError", errorMap);
    }

    // Required for React Native: addListener and removeListeners to satisfy NativeEventEmitter
    // These methods are often called by JS side when NativeEventEmitter.addListener is used.
    @ReactMethod
    public void addListener(String eventName) {
        // Keep: Required for RN built-in Event Emitter Calls.
        // This method can be empty if events are directly emitted using
        // DeviceEventManagerModule.RCTDeviceEventEmitter, as done with sendEvent().
        // No specific logic needed here for this implementation.
        Log.d(TAG, "addListener called for event: " + eventName);
    }

    @ReactMethod
    public void removeListeners(Integer count) {
        // Keep: Required for RN built-in Event Emitter Calls.
        // Similar to addListener, this can be empty for this direct emission pattern.
        // No specific logic needed here.
        Log.d(TAG, "removeListeners called with count: " + count);
    }
}
