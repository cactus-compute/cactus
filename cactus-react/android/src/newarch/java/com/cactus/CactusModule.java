package com.cactus;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.module.annotations.ReactModule;

import java.util.HashMap;
import java.util.Random;
import java.io.File;
import java.io.FileInputStream;
import java.io.PushbackInputStream;

@ReactModule(name = Cactus.NAME)
public class CactusModule extends NativeCactusSpec {
  public static final String NAME = Cactus.NAME; // Assuming Cactus.NAME is defined elsewhere or replace it.
                                            // For JNI, it's typical to load the library in a static block.
  static {
    System.loadLibrary("cactus"); // Or your specific JNI library name
  }

  private Cactus cactus = null; // This line might be problematic if Cactus.java is not found.
                                // For now, I'll assume it's for non-STT methods.
                                // Or STT methods will be static or JNI direct.

  public CactusModule(ReactApplicationContext reactContext) {
    super(reactContext);
    // If Cactus class is essential and not found, this will be an issue.
    // For STT methods, we will use direct JNI calls in this file.
    try {
      // This is a bit of a hack. If Cactus.java is missing, this will fail.
      // We are trying to keep the original structure for non-STT methods.
      cactus = new Cactus(reactContext);
    } catch (NoClassDefFoundError e) {
        System.err.println("Warning: com.cactus.Cactus class not found. Non-STT methods may fail.");
        cactus = null; // Ensure cactus is null if class not found
    }
  }

  @Override
  @NonNull
  public String getName() {
    // Return actual name, Cactus.NAME might rely on the missing class.
    return "Cactus";
  }

  // Native (JNI) methods for STT
  private native void nativeInitSTT(String modelPath, Promise promise);
  private native void nativeProcessAudioFile(String filePath, Promise promise);
  private native void nativeReleaseSTT(Promise promise);
  // TODO: Add nativeGetTranscription if needed

  @ReactMethod
  public void toggleNativeLog(boolean enabled, Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.toggleNativeLog(enabled, promise);
  }

  @ReactMethod
  public void setContextLimit(double limit, Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.setContextLimit(limit, promise);
  }

  @ReactMethod
  public void modelInfo(final String model, final ReadableArray skip, final Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.modelInfo(model, skip, promise);
  }

  @ReactMethod
  public void initContext(double id, final ReadableMap params, final Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.initContext(id, params, promise);
  }

  @ReactMethod
  public void getFormattedChat(double id, String messages, String chatTemplate, ReadableMap params, Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.getFormattedChat(id, messages, chatTemplate, params, promise);
  }

  @ReactMethod
  public void loadSession(double id, String path, Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.loadSession(id, path, promise);
  }

  @ReactMethod
  public void saveSession(double id, String path, double size, Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.saveSession(id, path, size, promise);
  }

  @ReactMethod
  public void completion(double id, final ReadableMap params, final Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.completion(id, params, promise);
  }

  @ReactMethod
  public void stopCompletion(double id, final Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.stopCompletion(id, promise);
  }

  @ReactMethod
  public void tokenize(double id, final String text, final Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.tokenize(id, text, promise);
  }

  @ReactMethod
  public void detokenize(double id, final ReadableArray tokens, final Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.detokenize(id, tokens, promise);
  }

  @ReactMethod
  public void embedding(double id, final String text, final ReadableMap params, final Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.embedding(id, text, params, promise);
  }

  @ReactMethod
  public void bench(double id, final double pp, final double tg, final double pl, final double nr, final Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.bench(id, pp, tg, pl, nr, promise);
  }

  @ReactMethod
  public void applyLoraAdapters(double id, final ReadableArray loraAdapters, final Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.applyLoraAdapters(id, loraAdapters, promise);
  }

  @ReactMethod
  public void removeLoraAdapters(double id, final Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.removeLoraAdapters(id, promise);
  }

  @ReactMethod
  public void getLoadedLoraAdapters(double id, final Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.getLoadedLoraAdapters(id, promise);
  }

  @ReactMethod
  public void releaseContext(double id, Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.releaseContext(id, promise);
  }

  @ReactMethod
  public void releaseAllContexts(Promise promise) {
    if (cactus == null) { promise.reject("E_NO_CACTUS_INSTANCE", "Cactus helper class not initialized."); return; }
    cactus.releaseAllContexts(promise);
  }

  // STT Methods
  @ReactMethod
  public void initSTT(String modelPath, Promise promise) {
    nativeInitSTT(modelPath, promise);
  }

  @ReactMethod
  public void processAudioFile(String filePath, Promise promise) {
    nativeProcessAudioFile(filePath, promise);
  }

  @ReactMethod
  public void releaseSTT(Promise promise) {
    nativeReleaseSTT(promise);
  }
  // End STT Methods
}
