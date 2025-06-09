package com.cactus;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
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
public class CactusModule extends ReactContextBaseJavaModule {
  public static final String NAME = Cactus.NAME;

  private Cactus cactus = null;

  public CactusModule(ReactApplicationContext reactContext) {
    super(reactContext);
    cactus = new Cactus(reactContext);
  }

  @Override
  @NonNull
  public String getName() {
    return NAME;
  }

  @ReactMethod
  public void toggleNativeLog(boolean enabled, Promise promise) {
    cactus.toggleNativeLog(enabled, promise);
  }

  @ReactMethod
  public void setContextLimit(double limit, Promise promise) {
    cactus.setContextLimit(limit, promise);
  }

  @ReactMethod
  public void modelInfo(final String model, final ReadableArray skip, final Promise promise) {
    cactus.modelInfo(model, skip, promise);
  }

  @ReactMethod
  public void initContext(double id, final ReadableMap params, final Promise promise) {
    cactus.initContext(id, params, promise);
  }

  @ReactMethod
  public void getFormattedChat(double id, String messages, String chatTemplate, ReadableMap params, Promise promise) {
    cactus.getFormattedChat(id, messages, chatTemplate, params, promise);
  }

  @ReactMethod
  public void loadSession(double id, String path, Promise promise) {
    cactus.loadSession(id, path, promise);
  }

  @ReactMethod
  public void saveSession(double id, String path, int size, Promise promise) {
    cactus.saveSession(id, path, size, promise);
  }

  @ReactMethod
  public void completion(double id, final ReadableMap params, final Promise promise) {
    cactus.completion(id, params, promise);
  }

  @ReactMethod
  public void stopCompletion(double id, final Promise promise) {
    cactus.stopCompletion(id, promise);
  }

  @ReactMethod
  public void tokenize(double id, final String text, final Promise promise) {
    cactus.tokenize(id, text, promise);
  }

  @ReactMethod
  public void tokenize(double id, final String text, final ReadableArray mediaPaths, final Promise promise) {
    cactus.tokenize(id, text, mediaPaths, promise);
  }

  @ReactMethod
  public void detokenize(double id, final ReadableArray tokens, final Promise promise) {
    cactus.detokenize(id, tokens, promise);
  }

  @ReactMethod
  public void embedding(double id, final String text, final ReadableMap params, final Promise promise) {
    cactus.embedding(id, text, params, promise);
  }

  @ReactMethod
  public void bench(double id, final double pp, final double tg, final double pl, final double nr, final Promise promise) {
    cactus.bench(id, pp, tg, pl, nr, promise);
  }

  @ReactMethod
  public void applyLoraAdapters(double id, final ReadableArray loraAdapters, final Promise promise) {
    cactus.applyLoraAdapters(id, loraAdapters, promise);
  }

  @ReactMethod
  public void removeLoraAdapters(double id, final Promise promise) {
    cactus.removeLoraAdapters(id, promise);
  }

  @ReactMethod
  public void getLoadedLoraAdapters(double id, final Promise promise) {
    cactus.getLoadedLoraAdapters(id, promise);
  }

  @ReactMethod
  public void initMultimodal(double id, final String mmprojPath, final boolean useGpu, final Promise promise) {
    cactus.initMultimodal(id, mmprojPath, useGpu, promise);
  }

  @ReactMethod
  public void isMultimodalEnabled(double id, final Promise promise) {
    cactus.isMultimodalEnabled(id, promise);
  }

  @ReactMethod
  public void isMultimodalSupportVision(double id, final Promise promise) {
    cactus.isMultimodalSupportVision(id, promise);
  }

  @ReactMethod
  public void isMultimodalSupportAudio(double id, final Promise promise) {
    cactus.isMultimodalSupportAudio(id, promise);
  }

  @ReactMethod
  public void releaseMultimodal(double id, final Promise promise) {
    cactus.releaseMultimodal(id, promise);
  }

  @ReactMethod
  public void multimodalCompletion(double id, final String prompt, final ReadableArray mediaPaths, final ReadableMap params, final Promise promise) {
    cactus.multimodalCompletion(id, prompt, mediaPaths, params, promise);
  }

  @ReactMethod
  public void initVocoder(double id, final String vocoderModelPath, final Promise promise) {
    cactus.initVocoder(id, vocoderModelPath, promise);
  }

  @ReactMethod
  public void isVocoderEnabled(double id, final Promise promise) {
    cactus.isVocoderEnabled(id, promise);
  }

  @ReactMethod
  public void getTTSType(double id, final Promise promise) {
    cactus.getTTSType(id, promise);
  }

  @ReactMethod
  public void getFormattedAudioCompletion(double id, final String speakerJsonStr, final String textToSpeak, final Promise promise) {
    cactus.getFormattedAudioCompletion(id, speakerJsonStr, textToSpeak, promise);
  }

  @ReactMethod
  public void getAudioCompletionGuideTokens(double id, final String textToSpeak, final Promise promise) {
    cactus.getAudioCompletionGuideTokens(id, textToSpeak, promise);
  }

  @ReactMethod
  public void decodeAudioTokens(double id, final ReadableArray tokens, final Promise promise) {
    cactus.decodeAudioTokens(id, tokens, promise);
  }

  @ReactMethod
  public void releaseVocoder(double id, final Promise promise) {
    cactus.releaseVocoder(id, promise);
  }

  @ReactMethod
  public void releaseContext(double id, Promise promise) {
    cactus.releaseContext(id, promise);
  }

  @ReactMethod
  public void releaseAllContexts(Promise promise) {
    cactus.releaseAllContexts(promise);
  }
}
