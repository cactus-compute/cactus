# Cactus Repository Analysis

## 1. Overview
**Cactus** is a high-performance, energy-efficient AI inference engine and runtime designed primarily for **mobile and ARM devices**. It aims to provide a unified computation graph (similar to NumPy) and low-level kernels (similar to CUDA) but optimized for ARM NEON and NPU architectures.

- **Repository**: [cactus-compute/cactus](https://github.com/cactus-compute/cactus)
- **Version**: 1.5.0
- **Primary Goal**: Run SOTA models (Gemma, Whisper, LFM, Qwen) on mobile devices (iOS, Android, macOS, Raspberry Pi) with high performance and mixed precision (INT4/INT8/FP16).

## 2. Architecture

The project follows a layered architecture:

1.  **Cactus FFI**: OpenAI-compatible C API for integration with tools, RAG, and cloud handoff.
2.  **Cactus Engine**: High-level transformer engine supporting various precisions and NPU acceleration.
3.  **Cactus Models**: Implementation of specific model architectures (e.g., Gemma, Whisper) using the Cactus Graph.
4.  **Cactus Graph**: A zero-copy computation graph.
5.  **Cactus Kernels**: Low-level ARM-specific SIMD operations.

## 3. Technology Stack

-   **Core Language**: C++20 (Heavy use of template metaprogramming and modern features).
-   **Build System**: CMake.
-   **Scripting**: Bash (`setup`, `build.sh`).
-   **Bindings**:
    -   **Python**: Located in `python/` directory (standard pip package).
    -   **Others**: SDKs for React Native, Swift, Kotlin, Flutter, Rust are linked but hosted in separate repos.
-   **Dependencies**:
    -   `libcactus_pro.a` (Optional closed-source static lib for NPU acceleration).
    -   `CURL` (for model downloads/networking).
    -   `CoreML` / `Foundation` (on Apple platforms).

## 4. Platform Compatibility & Build Analysis

### **ARM Focus**
The `CMakeLists.txt` is heavily optimized for ARM architectures.
-   It mandates `-march=armv8.2-a+fp16+simd+dotprod+i8mm` flags.
-   It defines `__ARM_NEON` and other ARM-specific macros.
-   **x86/Windows Support**: There is **no explicit support for x86_64 or Windows (MSVC)** in the build configuration observed. Building on a standard Windows Intel/AMD machine will likely fail without significant modification (or emulation).

### **Operating Systems**
-   **First Class**: macOS (Apple Silicon), iOS, Android.
-   **Supported**: Linux (ARM - e.g., Raspberry Pi).
-   **Unsupported**: Windows (x86), Linux (x86).

## 5. Development Setup

The provided `setup` script is a Bash script designed for macOS/Linux environments:
1.  Configures Git hooks.
2.  Sets up a Python virtual environment (`venv`).
3.  Installs Python dependencies.
4.  Installs the Cactus CLI (`pip install -e python`).

## 6. CLI Tools

The `python` directory provides a CLI interface with commands:
-   `cactus run [model]`: Open playground.
-   `cactus download [model]`: Download weights.
-   `cactus build`: Build for ARM.
-   `cactus convert`: Convert models.

## 7. Conclusion

Cactus is a specialized, high-performance engine for **Edge AI**. It is not a general-purpose inference engine for x86 servers but specifically targets the mobile/embedded ecosystem.

**For a Windows User**:
-   Direct compilation is unlikely to work due to ARM-specific flags.
-   Usage would likely require an ARM-based Windows machine or strictly using the Remote/Cloud capabilities if available (though the core value prop is local inference).
-   To run/test, you would ideally use a MacBook (M-series), a Raspberry Pi, or an Android device.
