# Cactus Android ProGuard Rules
# These rules ensure JNI classes and methods are preserved during minification

# Keep all classes in the cactus package
-keep class com.cactus.** { *; }

# Keep CactusHttp class and all its methods (called via JNI)
-keepclassmembers class com.cactus.CactusHttp {
    public static void post(long, java.lang.String, java.lang.String, java.lang.String, java.lang.String, int);
    public static void get(long, java.lang.String, java.lang.String, int);
    private static native void nativeOnResponse(long, int, java.lang.String, java.lang.String);
    public static void init(android.content.Context);
    public static boolean isNetworkAvailable();
    public static void shutdown();
}

# Keep Cactus main class native methods
-keepclassmembers class com.cactus.Cactus {
    private static native long nativeInit(java.lang.String, java.lang.String);
    private static native java.lang.String nativeGetLastError();
    private static native boolean nativeIsNetworkAvailable();
    private static native java.lang.String nativeTestHttp(java.lang.String);
    private native void nativeDestroy(long);
    private native void nativeReset(long);
    private native void nativeStop(long);
    private native java.lang.String nativeComplete(long, java.lang.String, java.lang.String, java.lang.String, com.cactus.TokenCallback);
    private native java.lang.String nativeTranscribe(long, java.lang.String, java.lang.String, java.lang.String, byte[]);
    private native float[] nativeEmbed(long, java.lang.String, boolean);
    private native java.lang.String nativeRagQuery(long, java.lang.String, int);
    private native int[] nativeTokenize(long, java.lang.String);
    private native java.lang.String nativeScoreWindow(long, int[], int, int, int);
    private native float[] nativeImageEmbed(long, java.lang.String);
    private native float[] nativeAudioEmbed(long, java.lang.String);
    private native long nativeStreamTranscribeInit(long);
}

# Keep CactusIndex native methods
-keepclassmembers class com.cactus.CactusIndex {
    private static native long nativeIndexInit(java.lang.String, int);
    private native int nativeIndexAdd(long, int[], java.lang.String[], java.lang.String[], float[][], int);
    private native int nativeIndexDelete(long, int[]);
    private native java.lang.String nativeIndexQuery(long, float[], int, java.lang.String);
    private native int nativeIndexCompact(long);
    private native void nativeIndexDestroy(long);
}

# Keep callback interfaces
-keep interface com.cactus.TokenCallback { *; }

# Keep model/result classes (for JSON serialization)
-keep class com.cactus.Message { *; }
-keep class com.cactus.CompletionOptions { *; }
-keep class com.cactus.CompletionResult { *; }
-keep class com.cactus.TranscriptionResult { *; }
-keep class com.cactus.CactusException { *; }
-keep class com.cactus.StreamTranscriber { *; }

# Keep native library loading
-keepclasseswithmembernames class * {
    native <methods>;
}

# Don't warn about missing classes that might not be present on all Android versions
-dontwarn android.net.NetworkInfo
-dontwarn android.net.ConnectivityManager
