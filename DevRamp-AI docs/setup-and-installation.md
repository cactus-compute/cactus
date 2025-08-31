# Setup & Installation

# Cactus: Cross-Platform LLM/VLM/TTS Framework - Setup & Installation Guide

This guide provides a comprehensive walkthrough for setting up and deploying the `cactus` framework, enabling you to run LLMs, VLMs, and TTS models locally on smartphones.

## 📋 Prerequisites and Requirements

Before proceeding, ensure you have the following:

* **CMake:**  A build system for C++.  Download and install from [https://cmake.org/](https://cmake.org/).
* **A C++ Compiler:**  A suitable C++ compiler (e.g., g++ or clang++) compatible with your system.
* **Git:** For cloning the repository.  Download and install from [https://git-scm.com/](https://git-scm.com/).
* **Swift (for iOS/macOS):** If targeting iOS or macOS, you'll need Xcode and the Swift command-line tools.
* **Kotlin (for Android):** If targeting Android, you'll need the Kotlin compiler and Android Studio.
* **Flutter (Optional):** For Flutter integration.  Follow the instructions on [https://flutter.dev/](https://flutter.dev/).
* **React Native (Optional):** For React Native integration.  Follow the instructions on [https://reactnative.dev/](https://reactnative.dev/).
* **GGUF Models:** You'll need compatible GGUF models for your chosen tasks (LLM, VLM, TTS).  These can often be found on Hugging Face.


## 🔧 Step-by-Step Installation Process

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/<your_repo_url>/cactus.git
   cd cactus
   ```

2. **Build the C++ Core:** Navigate to the `cpp` directory.  The exact build process might vary depending on your system and compiler. A basic example using CMake is shown below:

   ```bash
   cd cpp
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

   This will compile the core C++ components, including `cactus_completion.cpp`, `cactus_embedding.cpp`, etc.  You might need to adjust CMake settings based on your system and dependencies.  Refer to the `cpp/README.md` for more detailed instructions and potential platform-specific configurations.

3. **Platform-Specific Integration:**

   * **iOS/macOS (Swift):**  The `Package.swift` file provides instructions for integrating the C++ core into a Swift project using Swift Package Manager.  You'll need to create a Swift project and add `cactus` as a dependency.

   * **Android (Kotlin):**  You'll need to create a Kotlin Multiplatform project and integrate the C++ core using the appropriate Kotlin/Native mechanisms.  Detailed instructions for this are not provided in the current information.

   * **Flutter:** Integrate the C++ core using Flutter's platform channels.  This will require creating platform-specific code (e.g., Java/Kotlin for Android, Objective-C/Swift for iOS) to interact with the C++ library.

   * **React Native:** Similar to Flutter, you'll need to use platform channels to bridge the gap between React Native and the C++ core.

4. **Install Dependencies:** The exact dependencies will depend on the platform and chosen model.  The codebase uses `nlohmann::ordered_json` (likely for JSON handling) and libraries for GGUF model loading.  You might need to install these separately using your package manager (e.g., `apt`, `brew`, `vcpkg`).


## ⚡ Quick Start Guide

Once the C++ core is built and integrated into your chosen platform, you can start using the `cactus` API.  The following example demonstrates a basic LLM completion using the C++ API (this requires adapting to your chosen platform's binding):

```cpp
#include "cactus/cactus_completion.h" // Example - adapt to your actual header

int main() {
  cactus::CompletionContext context; // Initialize context (implementation details omitted)
  std::string prompt = "What is the capital of France?";
  std::string completion = context.complete(prompt); // Perform completion
  std::cout << completion << std::endl;
  return 0;
}
```

Remember to replace this with the appropriate code for your chosen platform (Swift, Kotlin, Flutter, React Native).  Refer to the relevant examples in the `cpp/example` directory for more detailed usage examples.


## 🚀 How to Run and Deploy

The execution and deployment process depends heavily on the chosen platform.

* **C++ (for testing):** You can run the examples in the `cpp/example` directory after building the C++ core.

* **iOS/Android:** Build and deploy your application using Xcode (iOS) or Android Studio (Android).

* **Flutter/React Native:** Build and deploy your application using the Flutter or React Native command-line tools.


## 🛠️ Development Environment Setup

Setting up your development environment involves installing the prerequisites mentioned above and configuring your IDE (Xcode, Android Studio, VS Code, etc.) for C++, Swift, Kotlin, and the chosen cross-platform framework (Flutter or React Native).  The `cpp/README.md` might offer additional environment-specific instructions.


## 🔍 Troubleshooting Common Issues

* **Compilation Errors:** Carefully review compiler error messages.  Ensure all dependencies are correctly installed and linked.  Check the CMake configuration.

* **Runtime Errors:**  Use logging statements (`LOG_INFO`, `LOG_ERROR`, etc.) within the C++ code to track down runtime issues.

* **Model Loading Errors:**  Verify that the GGUF model is correctly placed and accessible.  Check the model's format and compatibility with the `cactus` framework.

* **Platform-Specific Issues:** Consult the documentation for your chosen platform (iOS, Android, Flutter, React Native) for troubleshooting guidance.


This guide provides a foundation for setting up and using the `cactus` framework.  Remember to consult the repository's `README.md` and platform-specific documentation for more detailed information and advanced usage scenarios.  The lack of detailed API documentation in the provided information limits the precision of this guide; further documentation within the repository would greatly improve its usability.


## About This Documentation

- **Type:** Setup
- **Generated:** 30/08/2025
- **Source:** DevRamp AI Analysis

---

*This documentation was automatically generated by [DevRamp](https://devramp.com) - AI-powered developer onboarding and context generation platform.*
