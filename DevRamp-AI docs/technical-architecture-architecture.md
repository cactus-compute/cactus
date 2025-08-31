# Technical Architecture

**Type:** Architecture  
**Created:** 30/08/2025  
**Last Updated:** Just now

---

# Cactus Architecture Documentation

This document details the architecture of the `cactus` cross-platform framework for deploying LLMs, VLMs, and TTS models on smartphones.

## 1. Overall System Design and Structure

`cactus` employs a layered architecture with a C++ core providing the fundamental model loading, inference, and processing capabilities. This core is then wrapped with platform-specific bindings for integration with Flutter, React Native, and Kotlin Multiplatform, enabling deployment across iOS, Android, and potentially other platforms.

The system can be visualized as follows:

```
+-----------------+     +-----------------+     +-----------------+     +-----------------+
| Mobile App      | <--> | Platform Bindings | <--> | C++ Core        | <--> | AI Models       |
| (Flutter, RN,   |     | (Swift, Kotlin)  |     | (LLM, VLM, TTS) |     | (GGUF format)   |
| Kotlin Multiplat)|     |                   |     |                   |     |                   |
+-----------------+     +-----------------+     +-----------------+     +-----------------+
```

## 2. Component Interaction and Data Flow

1. **Mobile App:** The application initiates requests (e.g., text completion, embedding generation, TTS synthesis) through the platform-specific bindings.

2. **Platform Bindings:** These act as intermediaries, translating requests from the mobile app's language (Swift, Kotlin, Dart) into a format consumable by the C++ core.  They also handle the marshaling of data between the C++ core and the mobile app.

3. **C++ Core:** This layer performs the core AI processing.  It loads the GGUF model, preprocesses input, performs inference, and postprocesses the output.  Key components within the C++ core include:
    * `cactus_context.cpp`: Likely manages the overall context and state for model interactions.
    * `cactus_completion.cpp`: Handles LLM completion tasks.
    * `cactus_embedding.cpp`: Generates embeddings from text.
    * `cactus_chat.cpp`: Manages chat interactions, potentially using Jinja2 templating.
    * `cactus_tts.cpp`: Handles text-to-speech synthesis.
    * `cactus_loader.cpp`: Responsible for loading GGUF models.
    * `cactus_tokenization.cpp`: Handles tokenization of input and output text.
    * `cactus_utils.cpp`: Contains utility functions used across the C++ core.

4. **AI Models:**  The models (LLMs, VLMs, TTS) are loaded in GGUF format. The C++ core interacts directly with these models to perform inference.

The data flow is generally as follows:  The mobile app sends a request to the bindings, which forwards it to the C++ core. The core processes the request using the loaded model, and the result is sent back through the bindings to the mobile app.

## 3. Code Organization and Module Structure

The code is primarily organized into a `cpp` directory containing the C++ core, and separate directories (implied by the build files) for platform-specific bindings.  The use of namespaces (`namespace cactus`) within the C++ code helps to organize the code and prevent naming conflicts.

## 4. Design Patterns and Architectural Decisions

* **Layered Architecture:**  This promotes modularity, maintainability, and reusability.  Changes in one layer have minimal impact on other layers.
* **Abstraction:** The platform bindings abstract away the complexities of interacting with different mobile platforms.
* **Dependency Inversion:** The C++ core is designed to be independent of specific mobile platforms, relying on the bindings for platform-specific interactions.

## 5. Dependencies and Integrations

* **GGUF:** The framework relies heavily on the GGUF model format for model loading and storage.
* **Jinja2 (likely):**  Used for templating in chat interactions (indicated by the AI analysis).
* **External C++ Libraries:**  The C++ core likely uses external libraries for model loading (GGML), JSON parsing (e.g., `nlohmann::ordered_json`), and potentially other functionalities.
* **Platform-Specific Dependencies:**  The bindings depend on the respective mobile development frameworks (Flutter, React Native, Kotlin Multiplatform).


## 6. Code Examples

**Illustrative (Hypothetical) Code Snippet from `cactus_completion.cpp`:**

```cpp
namespace cactus {

  std::string complete(const std::string& prompt, const Model& model) {
    // ... preprocessing of prompt ...
    auto result = model.generate(prompt); // Hypothetical model generation function
    // ... postprocessing of result ...
    return result;
  }

}
```

**Note:** This is a hypothetical example.  The actual implementation details are not available without access to the source code.


## 7. Areas for Improvement

* **Code Complexity:**  The high complexity scores in some C++ files suggest opportunities for refactoring to improve readability and maintainability.
* **Documentation:**  More detailed documentation of the C++ API and the cross-platform bindings would be beneficial.
* **Testing:**  Adding comprehensive unit and integration tests is crucial to ensure code robustness and reliability.
* **Error Handling:**  Explicit error handling mechanisms should be implemented throughout the C++ core and bindings.


This documentation provides a high-level overview of the `cactus` architecture.  A more detailed analysis would require access to the complete source code.


---

*Generated by DevRamp - AI-powered developer onboarding platform*
