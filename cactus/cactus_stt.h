#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Forward declarations from whisper.h
struct whisper_context;
struct whisper_full_params;

namespace cactus {

class STT {
public:
    STT();
    ~STT();

    // Initialize the STT engine with a model path
    // model_path: Path to the ggml Whisper model file.
    // language: Language code (e.g., "en").
    // use_gpu: Whether to attempt GPU usage (if compiled with GPU support).
    bool initialize(const std::string& model_path, const std::string& language = "en", bool use_gpu = true);

    // Process audio samples for transcription.
    // samples: A vector of float audio samples (PCM 32-bit, 16kHz, mono).
    // For simplicity in this initial version, we assume the input audio is already in the correct format.
    bool processAudio(const std::vector<float>& samples);

    // Get the full transcribed text.
    std::string getTranscription();

    // (Optional Advanced) Get individual text segments with timestamps.
    // struct Segment { std::string text; int64_t t0; int64_t t1; };
    // std::vector<Segment> getSegments();

    bool isInitialized() const;

private:
    void cleanup();

    whisper_context* ctx_ = nullptr;
    std::string language_ = "en";
    // Potentially add members for whisper_full_params if customization is needed beyond defaults.
    // Or, create whisper_full_params on the stack in processAudio.
};

} // namespace cactus
