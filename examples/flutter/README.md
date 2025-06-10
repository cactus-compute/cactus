# Cactus Flutter Chat Example

This application demonstrates the capabilities of the `cactus` Flutter plugin for on-device AI model inference. It showcases how to:

*   Initialize the Cactus AI engine with a model (downloaded from a URL or loaded from a local path).
*   Perform chat completions using the loaded model.
*   Stream responses token by token.
*   Manage the lifecycle of the `CactusContext`.

## Prerequisites

*   Flutter SDK installed.
*   An appropriate GGUF model URL (or a model file placed in a path accessible by the app if you modify the code to load locally).
*   Ensure you have set up your environment for Flutter development (e.g., Xcode for iOS, Android Studio/SDK for Android).

## Getting Started

This example application is designed to be run directly.

1.  **Obtain the Example Code:**
    *   If you have cloned the main `cactus` repository, navigate to this directory: `examples/flutter-chat`.
    *   If you have downloaded this example folder individually (e.g., from a ZIP or by forking and cloning a specific part), ensure you have this `flutter-chat` folder.

2.  **Navigate to the Example Directory in Your Terminal:**
    ```bash
    cd path/to/your/copy/of/flutter-chat 
    ```

3.  **Get Dependencies:**
    ```bash
    flutter pub get
    ```
    This will fetch all necessary packages, including the `cactus` plugin from pub.dev.

4.  **Configure Model URL:**
    Open `lib/main.dart` in this example project. Find the `initializeModel()` function and update the `modelUrl` in `CactusInitParams` to point to a valid GGUF model URL that you wish to test.

    ```dart
    // Inside lib/main.dart, in initializeModel():
    final initParams = CactusInitParams(
      modelUrl: 'YOUR_ACTUAL_MODEL_URL_HERE', // <-- REPLACE THIS
      // ... other parameters
    );
    ```

5.  **Run the Application:**
    ```bash
    flutter run
    ```
    Select your target device (iOS simulator/device or Android emulator/device).

## Features Demonstrated

*   **Model Initialization**: Shows how to use `CactusContext.init()` with `CactusInitParams`, including progress callbacks for model downloads.
*   **Chat UI**: A basic interface to send messages and display responses.
*   **Streaming**: Tokens from the AI model are displayed as they arrive using the `onNewToken` callback.
*   **Resource Management**: Demonstrates calling `cactusContext.free()` when the context is no longer needed.

## About the `cactus` Plugin

For more detailed information about the `cactus` Flutter plugin itself, its API, and advanced features, please refer to the [main plugin README](../../cactus-flutter/README.md).
