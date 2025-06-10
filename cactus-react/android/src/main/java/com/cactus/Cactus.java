package com.cactus;

import androidx.annotation.NonNull;
import android.util.Log;
import android.os.Build;
import android.os.Handler;
import android.os.AsyncTask;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.LifecycleEventListener;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.Arguments;

import java.util.HashMap;
import java.util.Random;
import java.io.File;
import java.io.FileInputStream;
import java.io.PushbackInputStream;

public class Cactus implements LifecycleEventListener {
  public static final String NAME = "Cactus";

  private ReactApplicationContext reactContext;

  public Cactus(ReactApplicationContext reactContext) {
    reactContext.addLifecycleEventListener(this);
    this.reactContext = reactContext;
  }

  private HashMap<AsyncTask, String> tasks = new HashMap<>();

  private HashMap<Integer, LlamaContext> contexts = new HashMap<>();

  public void toggleNativeLog(boolean enabled, Promise promise) {
    new AsyncTask<Void, Void, Boolean>() {
      private Exception exception;

      @Override
      protected Boolean doInBackground(Void... voids) {
        try {
          LlamaContext.toggleNativeLog(reactContext, enabled);
          return true;
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Boolean result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
  }

  private int llamaContextLimit = -1;

  public void setContextLimit(double limit, Promise promise) {
    llamaContextLimit = (int) limit;
    promise.resolve(null);
  }

  public void modelInfo(final String model, final ReadableArray skip, final Promise promise) {
    new AsyncTask<Void, Void, WritableMap>() {
      private Exception exception;

      @Override
      protected WritableMap doInBackground(Void... voids) {
        try {
          String[] skipArray = new String[skip.size()];
          for (int i = 0; i < skip.size(); i++) {
            skipArray[i] = skip.getString(i);
          }
          return LlamaContext.modelInfo(model, skipArray);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(WritableMap result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
  }

  public void initContext(double id, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, WritableMap>() {
      private Exception exception;

      @Override
      protected WritableMap doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context != null) {
            throw new Exception("Context already exists");
          }
          if (llamaContextLimit > -1 && contexts.size() >= llamaContextLimit) {
            throw new Exception("Context limit reached");
          }
          LlamaContext llamaContext = new LlamaContext(contextId, reactContext, params);
          if (llamaContext.getContext() == 0) {
            throw new Exception("Failed to initialize context");
          }
          contexts.put(contextId, llamaContext);
          WritableMap result = Arguments.createMap();
          result.putBoolean("gpu", false);
          result.putString("reasonNoGPU", "Currently not supported");
          result.putMap("model", llamaContext.getModelDetails());
          result.putString("androidLib", llamaContext.getLoadedLibrary());
          return result;
        } catch (Exception e) {
          exception = e;
          return null;
        }
      }

      @Override
      protected void onPostExecute(WritableMap result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "initContext");
  }

  public void getFormattedChat(double id, final String messages, final String chatTemplate, final ReadableMap params, Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Object>() {
      private Exception exception;

      @Override
      protected Object doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          if (params.hasKey("jinja") && params.getBoolean("jinja")) {
            ReadableMap result = context.getFormattedChatWithJinja(messages, chatTemplate, params);
            if (result.hasKey("_error")) {
              throw new Exception(result.getString("_error"));
            }
            return result;
          }
          return context.getFormattedChat(messages, chatTemplate);
        } catch (Exception e) {
          exception = e;
          return null;
        }
      }

      @Override
      protected void onPostExecute(Object result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "getFormattedChat-" + contextId);
  }

  public void loadSession(double id, final String path, Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, WritableMap>() {
      private Exception exception;

      @Override
      protected WritableMap doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          WritableMap result = context.loadSession(path);
          return result;
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(WritableMap result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "loadSession-" + contextId);
  }

  public void saveSession(double id, final String path, double size, Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Integer>() {
      private Exception exception;

      @Override
      protected Integer doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          Integer count = context.saveSession(path, (int) size);
          return count;
        } catch (Exception e) {
          exception = e;
        }
        return -1;
      }

      @Override
      protected void onPostExecute(Integer result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "saveSession-" + contextId);
  }

  public void completion(double id, final ReadableMap params, final Promise promise) {
    Log.d(NAME, "🟢 BRIDGE: completion() method called with contextId=" + (int)id);
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, WritableMap>() {
      private Exception exception;

      @Override
      protected WritableMap doInBackground(Void... voids) {
        Log.d(NAME, "⚡ BRIDGE: AsyncTask doInBackground starting...");
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            Log.e(NAME, "❌ BRIDGE: Context not found for id=" + contextId);
            throw new Exception("Context not found");
          }
          Log.d(NAME, "✅ BRIDGE: Context found, checking if predicting...");
          if (context.isPredicting()) {
            Log.e(NAME, "❌ BRIDGE: Context is busy (predicting)");
            throw new Exception("Context is busy");
          }
          Log.d(NAME, "🚀 BRIDGE: About to call context.completion()...");
          WritableMap result = context.completion(params);
          Log.d(NAME, "✅ BRIDGE: context.completion() returned successfully");
          return result;
        } catch (Exception e) {
          Log.e(NAME, "❌ BRIDGE: Exception in doInBackground: " + e.getMessage());
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(WritableMap result) {
        Log.d(NAME, "📤 BRIDGE: onPostExecute called");
        if (exception != null) {
          Log.e(NAME, "❌ BRIDGE: Rejecting promise with exception: " + exception.getMessage());
          promise.reject(exception);
          return;
        }
        Log.d(NAME, "✅ BRIDGE: Resolving promise with result");
        promise.resolve(result);
        tasks.remove(this);
        Log.d(NAME, "🏁 BRIDGE: completion() finished successfully");
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    Log.d(NAME, "📋 BRIDGE: AsyncTask queued for execution");
    tasks.put(task, "completion-" + contextId);
  }

  public void stopCompletion(double id, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          context.stopCompletion();
          AsyncTask completionTask = null;
          for (AsyncTask task : tasks.keySet()) {
            if (tasks.get(task).equals("completion-" + contextId)) {
              task.get();
              break;
            }
          }
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Void result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "stopCompletion-" + contextId);
  }

  public void tokenize(double id, final String text, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, WritableMap>() {
      private Exception exception;

      @Override
      protected WritableMap doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.tokenize(text);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(WritableMap result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "tokenize-" + contextId);
  }

  public void tokenize(double id, final String text, final ReadableArray mediaPaths, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, WritableMap>() {
      private Exception exception;

      @Override
      protected WritableMap doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.tokenize(text, mediaPaths);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(WritableMap result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "tokenize-" + contextId);
  }

  public void detokenize(double id, final ReadableArray tokens, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, String>() {
      private Exception exception;

      @Override
      protected String doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.detokenize(tokens);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(String result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "detokenize-" + contextId);
  }

  public void embedding(double id, final String text, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, WritableMap>() {
      private Exception exception;

      @Override
      protected WritableMap doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.getEmbedding(text, params);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(WritableMap result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "embedding-" + contextId);
  }

  public void bench(double id, final double pp, final double tg, final double pl, final double nr, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, String>() {
      private Exception exception;

      @Override
      protected String doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.bench((int) pp, (int) tg, (int) pl, (int) nr);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(String result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "bench-" + contextId);
  }

  public void applyLoraAdapters(double id, final ReadableArray loraAdapters, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          if (context.isPredicting()) {
            throw new Exception("Context is busy");
          }
          context.applyLoraAdapters(loraAdapters);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Void result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "applyLoraAdapters-" + contextId);
  }

  public void removeLoraAdapters(double id, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          if (context.isPredicting()) {
            throw new Exception("Context is busy");
          }
          context.removeLoraAdapters();
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Void result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(null);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "removeLoraAdapters-" + contextId);
  }

  public void getLoadedLoraAdapters(double id, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, ReadableArray>() {
      private Exception exception;

      @Override
      protected ReadableArray doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.getLoadedLoraAdapters();
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(ReadableArray result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "getLoadedLoraAdapters-" + contextId);
  }

  public void releaseContext(double id, Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context " + id + " not found");
          }
          context.interruptLoad();
          context.stopCompletion();
          AsyncTask completionTask = null;
          for (AsyncTask task : tasks.keySet()) {
            if (tasks.get(task).equals("completion-" + contextId)) {
              task.get();
              break;
            }
          }
          context.release();
          contexts.remove(contextId);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Void result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(null);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "releaseContext-" + contextId);
  }

  public void releaseAllContexts(Promise promise) {
    AsyncTask task = new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          onHostDestroy();
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Void result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(null);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "releaseAllContexts");
  }

  @Override
  public void onHostResume() {
  }

  @Override
  public void onHostPause() {
  }

  @Override
  public void onHostDestroy() {
    for (LlamaContext context : contexts.values()) {
      context.stopCompletion();
    }
    for (AsyncTask task : tasks.keySet()) {
      try {
        task.get();
      } catch (Exception e) {
        Log.e(NAME, "Failed to wait for task", e);
      }
    }
    tasks.clear();
    for (LlamaContext context : contexts.values()) {
      context.release();
    }
    contexts.clear();
  }

  public void initMultimodal(double id, final String mmprojPath, final boolean useGpu, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Boolean>() {
      private Exception exception;

      @Override
      protected Boolean doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.initMultimodal(mmprojPath, useGpu);
        } catch (Exception e) {
          exception = e;
        }
        return false;
      }

      @Override
      protected void onPostExecute(Boolean result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "initMultimodal-" + contextId);
  }

  public void isMultimodalEnabled(double id, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Boolean>() {
      private Exception exception;

      @Override
      protected Boolean doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.isMultimodalEnabled();
        } catch (Exception e) {
          exception = e;
        }
        return false;
      }

      @Override
      protected void onPostExecute(Boolean result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "isMultimodalEnabled-" + contextId);
  }

  public void isMultimodalSupportVision(double id, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Boolean>() {
      private Exception exception;

      @Override
      protected Boolean doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.isMultimodalSupportVision();
        } catch (Exception e) {
          exception = e;
        }
        return false;
      }

      @Override
      protected void onPostExecute(Boolean result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "isMultimodalSupportVision-" + contextId);
  }

  public void isMultimodalSupportAudio(double id, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Boolean>() {
      private Exception exception;

      @Override
      protected Boolean doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.isMultimodalSupportAudio();
        } catch (Exception e) {
          exception = e;
        }
        return false;
      }

      @Override
      protected void onPostExecute(Boolean result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "isMultimodalSupportAudio-" + contextId);
  }

  public void releaseMultimodal(double id, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          context.releaseMultimodal();
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Void result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(null);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "releaseMultimodal-" + contextId);
  }

  public void multimodalCompletion(double id, final String prompt, final ReadableArray mediaPaths, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, WritableMap>() {
      private Exception exception;

      @Override
      protected WritableMap doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          if (context.isPredicting()) {
            throw new Exception("Context is busy");
          }
          return context.multimodalCompletion(prompt, mediaPaths, params);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(WritableMap result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "multimodalCompletion-" + contextId);
  }

  public void initVocoder(double id, final String vocoderModelPath, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Boolean>() {
      private Exception exception;

      @Override
      protected Boolean doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.initVocoder(vocoderModelPath);
        } catch (Exception e) {
          exception = e;
        }
        return false;
      }

      @Override
      protected void onPostExecute(Boolean result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "initVocoder-" + contextId);
  }

  public void isVocoderEnabled(double id, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Boolean>() {
      private Exception exception;

      @Override
      protected Boolean doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.isVocoderEnabled();
        } catch (Exception e) {
          exception = e;
        }
        return false;
      }

      @Override
      protected void onPostExecute(Boolean result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "isVocoderEnabled-" + contextId);
  }

  public void getTTSType(double id, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Integer>() {
      private Exception exception;

      @Override
      protected Integer doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.getTTSType();
        } catch (Exception e) {
          exception = e;
        }
        return -1;
      }

      @Override
      protected void onPostExecute(Integer result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "getTTSType-" + contextId);
  }

  public void getFormattedAudioCompletion(double id, final String speakerJsonStr, final String textToSpeak, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, String>() {
      private Exception exception;

      @Override
      protected String doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.getFormattedAudioCompletion(speakerJsonStr, textToSpeak);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(String result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "getFormattedAudioCompletion-" + contextId);
  }

  public void getAudioCompletionGuideTokens(double id, final String textToSpeak, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, WritableArray>() {
      private Exception exception;

      @Override
      protected WritableArray doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.getAudioCompletionGuideTokens(textToSpeak);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(WritableArray result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "getAudioCompletionGuideTokens-" + contextId);
  }

  public void decodeAudioTokens(double id, final ReadableArray tokens, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, WritableArray>() {
      private Exception exception;

      @Override
      protected WritableArray doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          return context.decodeAudioTokens(tokens);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(WritableArray result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "decodeAudioTokens-" + contextId);
  }

  public void releaseVocoder(double id, final Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context not found");
          }
          context.releaseVocoder();
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Void result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(null);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "releaseVocoder-" + contextId);
  }
}
