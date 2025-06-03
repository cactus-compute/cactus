package com.cactus.cactus_sdk.audio;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log; // Use Android logging
import androidx.core.content.ContextCompat; // Use AndroidX

import java.util.concurrent.atomic.AtomicBoolean;

public class AudioInputModule {

    private static final String TAG = "AudioInputModule";
    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static final int BYTES_PER_SAMPLE = 2; // For ENCODING_PCM_16BIT

    private AudioRecord audioRecord;
    private Thread recordingThread;
    private final AtomicBoolean isRecording = new AtomicBoolean(false);
    private int bufferSizeInBytes;

    private final AudioDataListener dataListener;

    public interface AudioDataListener {
        void onAudioDataReceived(float[] audioData);
        void onError(String errorMessage);
    }

    public AudioInputModule(AudioDataListener listener) {
        if (listener == null) {
            throw new IllegalArgumentException("AudioDataListener cannot be null");
        }
        this.dataListener = listener;
    }

    public static boolean hasMicrophonePermission(Context context) {
        if (context == null) {
            Log.e(TAG, "Context is null in hasMicrophonePermission");
            return false;
        }
        return ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
    }

    public void startRecording(Context context) {
        if (context == null) {
            dataListener.onError("Context cannot be null for startRecording.");
            return;
        }
        if (isRecording.get()) {
            // dataListener.onError("Already recording."); // Can be noisy if called multiple times
            Log.w(TAG, "StartRecording called when already recording.");
            return;
        }

        if (!hasMicrophonePermission(context)) {
            dataListener.onError("RECORD_AUDIO permission not granted.");
            return;
        }

        bufferSizeInBytes = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        if (bufferSizeInBytes == AudioRecord.ERROR || bufferSizeInBytes == AudioRecord.ERROR_BAD_VALUE) {
            Log.w(TAG, "Failed to get min buffer size (" + bufferSizeInBytes + "), using default (SAMPLE_RATE * 2 * BYTES_PER_SAMPLE, e.g. 2 seconds for stereo, 1 for mono).");
            // For mono, 1 second buffer. If stereo was used, it would be 2 seconds.
            bufferSizeInBytes = SAMPLE_RATE * 1 * BYTES_PER_SAMPLE * 2; // Default to 2s buffer as a fallback
        }

        try {
            // Ensure thread-safe creation of AudioRecord
            synchronized (this) {
                if (audioRecord != null) {
                    Log.w(TAG, "AudioRecord instance already exists during startRecording. Releasing old one.");
                    releaseAudioRecord(); // Release previous instance if any
                }
                audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC,
                                              SAMPLE_RATE,
                                              CHANNEL_CONFIG,
                                              AUDIO_FORMAT,
                                              bufferSizeInBytes);
            }

            if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                dataListener.onError("AudioRecord initialization failed. State: " + audioRecord.getState());
                releaseAudioRecord();
                return;
            }

            audioRecord.startRecording();
            isRecording.set(true); // Set recording status *after* successful start
            Log.d(TAG, "Audio recording started. Buffer size: " + bufferSizeInBytes + " bytes.");

            recordingThread = new Thread(this::recordLoop, "AudioRecordingThread");
            recordingThread.start();

        } catch (SecurityException e) {
            Log.e(TAG, "SecurityException during AudioRecord init: " + e.getMessage());
            dataListener.onError("SecurityException during AudioRecord init: " + e.getMessage());
            isRecording.set(false);
            releaseAudioRecord();
        } catch (IllegalStateException e) {
            Log.e(TAG, "IllegalStateException during AudioRecord start: " + e.getMessage());
            dataListener.onError("IllegalStateException during AudioRecord start: " + e.getMessage());
            isRecording.set(false);
            releaseAudioRecord();
        } catch (Exception e) { // Catch any other unexpected exceptions
            Log.e(TAG, "Exception during AudioRecord start: " + e.getMessage(), e);
            dataListener.onError("Exception during AudioRecord start: " + e.getMessage());
            isRecording.set(false);
            releaseAudioRecord();
        }
    }

    private void recordLoop() {
        Log.d(TAG, "Record loop started.");
        // Calculate buffer size for shorts based on byte buffer size
        short[] shortBuffer = new short[bufferSizeInBytes / BYTES_PER_SAMPLE];
        float[] floatBuffer = new float[shortBuffer.length]; // Re-use this buffer for conversion

        while (isRecording.get()) {
            AudioRecord currentAudioRecord;
            synchronized (this) {
                 currentAudioRecord = audioRecord;
            }

            if (currentAudioRecord == null) {
                Log.e(TAG, "AudioRecord is null in recordLoop, stopping.");
                dataListener.onError("AudioRecord became null unexpectedly in recordLoop.");
                isRecording.set(false); // Ensure loop terminates
                break;
            }

            int shortsRead = currentAudioRecord.read(shortBuffer, 0, shortBuffer.length);

            if (shortsRead > 0) {
                // Convert short[] to float[] and normalize
                for (int i = 0; i < shortsRead; i++) {
                    floatBuffer[i] = shortBuffer[i] / 32768.0f;
                }

                // Create a new float array with the exact number of samples read
                // to avoid sending a partially filled buffer or larger buffer than necessary.
                float[] exactData = new float[shortsRead];
                System.arraycopy(floatBuffer, 0, exactData, 0, shortsRead);
                dataListener.onAudioDataReceived(exactData);

            } else if (shortsRead < 0) {
                Log.e(TAG, "AudioRecord read error: " + shortsRead);
                dataListener.onError("AudioRecord read error code: " + shortsRead);
                isRecording.set(false); // Stop recording on read error
            }
            // If shortsRead == 0, it means no data was read in this iteration, but it's not an error.
            // The loop will continue as long as isRecording is true.
        }
        Log.d(TAG, "Record loop finished.");
        // Cleanup is done here to ensure resources are released when the loop exits,
        // regardless of how it exited (normal stop or error).
        releaseAudioRecord();
    }

    public void stopRecording() {
        Log.d(TAG, "stopRecording called.");
        if (!isRecording.getAndSet(false)) { // Atomically set to false and get previous value
            Log.w(TAG, "Not recording or stopRecording was already called.");
            // If recordingThread is somehow alive without isRecording being true,
            // or if audioRecord is not null, this ensures cleanup.
            if (recordingThread != null && recordingThread.isAlive()) {
                 Log.w(TAG, "isRecording was false, but thread is alive. Attempting to join.");
            } else if (audioRecord != null) {
                 Log.w(TAG, "isRecording was false, but audioRecord not null. Attempting release.");
            } else {
                return; // Nothing to do.
            }
        }

        // If the thread was started, attempt to join it.
        if (recordingThread != null) {
            try {
                recordingThread.join(500); // Wait for the thread to finish
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                Log.e(TAG, "Recording thread interruption while stopping: " + e.getMessage());
                dataListener.onError("Recording thread interruption: " + e.getMessage());
            }
            recordingThread = null; // Nullify the thread reference
        }

        // The recordLoop itself calls releaseAudioRecord upon exiting.
        // However, if the loop never started (e.g., error in startRecording after thread creation but before start),
        // or if there's a desire for an explicit cleanup path here, ensure release.
        // This also handles cases where stopRecording might be called very quickly after start.
        synchronized (this) {
            if (audioRecord != null) {
                Log.d(TAG, "Performing final cleanup in stopRecording for audioRecord.");
                releaseAudioRecord(); // releaseAudioRecord is synchronized and null-checks
            }
        }
        Log.d(TAG, "Audio recording stopped completely.");
    }

    // Synchronized to prevent race conditions from concurrent calls (e.g. from recordLoop and stopRecording)
    private synchronized void releaseAudioRecord() {
        if (audioRecord != null) {
            Log.d(TAG, "Releasing AudioRecord. Current recording state: " + audioRecord.getRecordingState());
            if (audioRecord.getState() == AudioRecord.STATE_INITIALIZED) { // Only stop if initialized
                try {
                    if (audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                        audioRecord.stop();
                        Log.d(TAG, "AudioRecord.stop() called.");
                    }
                } catch (IllegalStateException e) {
                     Log.e(TAG, "AudioRecord.stop() failed during release: " + e.getMessage());
                     // This can happen if stop is called when not recording, which is fine.
                }
            }
            audioRecord.release(); // Release the native resources
            audioRecord = null;    // Nullify the reference
            Log.d(TAG, "AudioRecord released and nulled.");
        } else {
            Log.d(TAG, "releaseAudioRecord called but audioRecord was already null.");
        }
    }

    public boolean isRecording() {
        // Check both our atomic flag and the AudioRecord's state.
        // The audioRecord can be non-null but not recording if setup failed or stop was called.
        AudioRecord currentAudioRecord = null;
        synchronized (this) {
            currentAudioRecord = audioRecord;
        }
        boolean arStateRecording = false;
        if (currentAudioRecord != null) {
            try {
                arStateRecording = currentAudioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING;
            } catch (IllegalStateException e) {
                Log.e(TAG, "IllegalStateException when checking AudioRecord state in isRecording: " + e.getMessage());
                // This might happen if audioRecord was released by another thread concurrently.
                // Consider it not recording in this case.
            }
        }
        return isRecording.get() && arStateRecording;
    }
}
