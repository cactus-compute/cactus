#include "cactus_stt.h"
#include "whisper.h" // Assuming whisper.h is in the include path

#include <cstdio> // For fprintf, stderr
#include <vector>

namespace cactus {

STT::STT() : ctx_(nullptr), language_("en") {
    // Constructor: Initialize members
}

STT::~STT() {
    cleanup();
}

bool STT::initialize(const std::string& model_path, const std::string& language, bool use_gpu) {
    if (ctx_) {
        fprintf(stderr, "STT: Already initialized. Call cleanup() first.\n");
        return false;
    }

    language_ = language;

    // Whisper context parameters
    whisper_context_params cparams = whisper_context_params_default();
    cparams.use_gpu = use_gpu;

    ctx_ = whisper_init_from_file_with_params(model_path.c_str(), cparams);

    if (ctx_ == nullptr) {
        fprintf(stderr, "STT: Failed to initialize whisper context from model '%s'\n", model_path.c_str());
        return false;
    }

    return true;
}

bool STT::processAudio(const std::vector<float>& samples) {
    if (!ctx_) {
        fprintf(stderr, "STT: Not initialized. Call initialize() first.\n");
        return false;
    }

    if (samples.empty()) {
        fprintf(stderr, "STT: Audio samples vector is empty.\n");
        return false;
    }

    // For simplicity, we use default whisper_full_params.
    // These can be customized further if needed.
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    // Set language if it was provided and is different from default.
    // whisper.h typically defaults to "en" if params.language is nullptr.
    // However, explicitly setting it ensures the desired language is used.
    wparams.language = language_.c_str();

    // Disable printing progress to stderr from whisper.cpp
    // wparams.print_progress = false;
    // wparams.print_special = false;
    // wparams.print_realtime = false;
    // wparams.print_timestamps = false;


    if (whisper_full(ctx_, wparams, samples.data(), samples.size()) != 0) {
        fprintf(stderr, "STT: Failed to process audio\n");
        return false;
    }

    return true;
}

std::string STT::getTranscription() {
    if (!ctx_) {
        fprintf(stderr, "STT: Not initialized. Cannot get transcription.\n");
        return "";
    }

    std::string full_text;
    const int n_segments = whisper_full_n_segments(ctx_);
    for (int i = 0; i < n_segments; ++i) {
        const char* segment_text = whisper_full_get_segment_text(ctx_, i);
        if (segment_text) {
            full_text += segment_text;
        }
    }
    return full_text;
}

// (Optional Advanced) Get individual text segments with timestamps.
// std::vector<STT::Segment> STT::getSegments() {
//     std::vector<Segment> segments;
//     if (!ctx_) {
//         fprintf(stderr, "STT: Not initialized. Cannot get segments.\n");
//         return segments;
//     }
//     const int n_segments = whisper_full_n_segments(ctx_);
//     for (int i = 0; i < n_segments; ++i) {
//         const char* text = whisper_full_get_segment_text(ctx_, i);
//         int64_t t0 = whisper_full_get_segment_t0(ctx_, i);
//         int64_t t1 = whisper_full_get_segment_t1(ctx_, i);
//         if (text) {
//             segments.push_back({text, t0, t1});
//         }
//     }
//     return segments;
// }

bool STT::isInitialized() const {
    return ctx_ != nullptr;
}

void STT::cleanup() {
    if (ctx_) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
}

} // namespace cactus
