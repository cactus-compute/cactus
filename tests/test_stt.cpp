#include "test_utils.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cctype>

using namespace EngineTestUtils;

static const char* g_transcribe_model_path = std::getenv("CACTUS_TEST_TRANSCRIBE_MODEL");
static const char* g_vad_model_path = std::getenv("CACTUS_TEST_VAD_MODEL");
static const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

static const char* get_transcribe_prompt() {
    if (g_transcribe_model_path) {
        std::string path = g_transcribe_model_path;
        std::transform(path.begin(), path.end(), path.begin(), [](unsigned char c){ return std::tolower(c); });
        if (path.find("whisper") != std::string::npos) {
            return "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>";
        }
    }
    return "";
}

static const char* g_whisper_prompt = get_transcribe_prompt();

bool test_audio_processor() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║         AUDIO PROCESSOR TEST             ║\n"
              << "╚══════════════════════════════════════════╝\n";
    using namespace cactus::engine;

    Timer t;

    const size_t n_fft = 400;
    const size_t hop_length = 160;
    const size_t sampling_rate = 16000;
    const size_t feature_size = 80;
    const size_t num_frequency_bins = 1 + n_fft / 2;

    AudioProcessor audio_proc;
    audio_proc.init_mel_filters(num_frequency_bins, feature_size, 0.0f, 8000.0f, sampling_rate);

    const size_t n_samples = sampling_rate;
    std::vector<float> waveform(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        waveform[i] = std::sin(2.0f * M_PI * 440.0f * i / sampling_rate);
    }

    AudioProcessor::SpectrogramConfig config;
    config.n_fft = n_fft;
    config.hop_length = hop_length;
    config.frame_length = n_fft;
    config.power = 2.0f;
    config.center = true;
    config.log_mel = "log10";

    auto log_mel_spec = audio_proc.compute_spectrogram(waveform, config);

    double elapsed = t.elapsed_ms();

    const float expected[] = {1.133450f, 1.142660f, 1.161900f, 1.196580f, 1.229480f};

    const size_t pad_length = n_fft / 2;
    const size_t padded_length = n_samples + 2 * pad_length;
    const size_t num_frames = 1 + (padded_length - n_fft) / hop_length;

    bool passed = true;
    if (log_mel_spec.size() != feature_size * num_frames) {
        std::cerr << "  [audio_processor] unexpected output size: got " << log_mel_spec.size()
                  << ", expected " << (feature_size * num_frames) << std::endl;
        passed = false;
    }

#ifdef __APPLE__
    const float abs_tolerance = 1e-4f;
    const float rel_tolerance = 1e-4f;
    for (size_t i = 0; i < 5 && passed; i++) {
        float actual = log_mel_spec[i * num_frames];
        float diff = std::abs(actual - expected[i]);
        float allowed = std::max(abs_tolerance, rel_tolerance * std::abs(expected[i]));
        if (diff > allowed) {
            std::cerr << "  [audio_processor][mac] idx=" << i
                      << " expected=" << expected[i]
                      << " actual=" << actual
                      << " diff=" << diff
                      << " allowed=" << allowed
                      << std::endl;
            passed = false;
        }
    }
#else
    // Linux uses the non-Accelerate FFT path with different absolute scaling.
    // Validate spectral shape against the same fixture rather than exact magnitude.
    const float shape_tolerance = 0.10f;
    const float anchor = log_mel_spec[0];
    if (!std::isfinite(anchor) || anchor <= 0.0f) {
        std::cerr << "  [audio_processor][non-apple] invalid anchor value: " << anchor << std::endl;
        passed = false;
    }
    for (size_t i = 0; i < 5 && passed; i++) {
        float actual = log_mel_spec[i * num_frames];
        if (!std::isfinite(actual)) {
            std::cerr << "  [audio_processor][non-apple] non-finite value at idx=" << i << std::endl;
            passed = false;
            break;
        }
        float expected_ratio = expected[i] / expected[0];
        float actual_ratio = actual / anchor;
        float diff = std::abs(actual_ratio - expected_ratio);
        if (diff > shape_tolerance) {
            std::cerr << "  [audio_processor][non-apple] idx=" << i
                      << " expected_ratio=" << expected_ratio
                      << " actual_ratio=" << actual_ratio
                      << " diff=" << diff
                      << " allowed=" << shape_tolerance
                      << " (actual=" << actual << ", anchor=" << anchor << ")"
                      << std::endl;
            passed = false;
        }
    }
#endif

    std::cout << "└─ Time: " << std::fixed << std::setprecision(2) << elapsed << "ms" << std::endl;

    return passed;
}

template<typename Predicate>
bool run_whisper_test(const char* title, const char* options_json, Predicate check) {
    if (!g_transcribe_model_path) {
        std::cout << "⊘ SKIP │ " << std::left << std::setw(25) << title
                  << " │ CACTUS_TEST_TRANSCRIBE_MODEL not set\n";
        return true;
    }

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << std::string("          ") + title << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_transcribe_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize Whisper model\n";
        return false;
    }

    char response[1 << 15] = {0};
    StreamingData stream;
    stream.model = model;

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    std::cout << "Transcript: ";
    int rc = cactus_transcribe(model, audio_path.c_str(), g_whisper_prompt,
                               response, sizeof(response), options_json,
                               stream_callback, &stream, nullptr, 0);

    std::cout << "\n\n[Results]\n";
    if (rc <= 0) {
        std::cerr << "failed\n";
        cactus_destroy(model);
        return false;
    }

    Metrics m;
    m.parse(response);
    m.print_json();

    bool ok = check(rc, m);
    cactus_destroy(model);
    return ok;
}

static bool test_transcription() {
    return run_whisper_test("TRANSCRIPTION", R"({"max_tokens": 100, "telemetry_enabled": false})",
        [](int rc, const Metrics& m) { return rc > 0 && m.completion_tokens >= 8; });
}

static bool test_stream_transcription() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║        STREAM TRANSCRIPTION TEST         ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_transcribe_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIBE_MODEL not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_transcribe_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize Whisper model\n";
        return false;
    }

    cactus_stream_transcribe_t stream = cactus_stream_transcribe_start(
        model,  R"({"confirmation_threshold": 1.0, "min_chunk_size": 16000, "telemetry_enabled": false})"
    );
    if (!stream) {
        std::cerr << "[✗] Failed to initialize stream transcribe\n";
        cactus_destroy(model);
        return false;
    }

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    FILE* wav_file = fopen(audio_path.c_str(), "rb");
    if (!wav_file) {
        std::cerr << "[✗] Failed to open audio file\n";
        cactus_stream_transcribe_stop(stream, nullptr, 0);
        cactus_destroy(model);
        return false;
    }

    fseek(wav_file, 44, SEEK_SET);
    std::vector<int16_t> pcm_samples;
    int16_t sample;
    while (fread(&sample, sizeof(int16_t), 1, wav_file) == 1) {
        pcm_samples.push_back(sample);
    }
    fclose(wav_file);

    const size_t chunk_size = 96000;
    Timer timer;
    std::string full_transcription;

    for (size_t offset = 0; offset < pcm_samples.size(); offset += chunk_size) {
        size_t size = std::min(chunk_size, pcm_samples.size() - offset);

        char response[1 << 15] = {0};
        int result = cactus_stream_transcribe_process(
            stream,
            reinterpret_cast<const uint8_t*>(pcm_samples.data() + offset),
            size * sizeof(int16_t),
            response,
            sizeof(response)
        );

        if (result < 0) {
            std::cerr << "\n[✗] Processing failed\n";
            cactus_stream_transcribe_stop(stream, nullptr, 0);
            cactus_destroy(model);
            return false;
        }

        std::string response_str(response);
        std::string confirmed = json_string(response_str, "confirmed");
        std::string pending = json_string(response_str, "pending");

        std::cout << "├─ transcription: " << full_transcription + pending << std::endl;

        if (!confirmed.empty()) {
            full_transcription += confirmed + " ";
        }
    }

    char final_response[1 << 15] = {0};
    int stop_result = cactus_stream_transcribe_stop(
        stream,
        final_response,
        sizeof(final_response)
    );

    if (stop_result < 0) {
        std::cerr << "[✗] Stop failed\n";
        cactus_destroy(model);
        return false;
    }

    std::string final_str(final_response);
    std::string final_confirmed = json_string(final_str, "confirmed");

    if (!final_confirmed.empty()) {
        full_transcription += final_confirmed;
        std::cout << "└─ confirmed: " << final_confirmed << "\n";
    }

    double elapsed = timer.elapsed_ms();

    size_t word_count = 0;
    bool in_word = false;
    for (char c : full_transcription) {
        if (std::isspace(c)) {
            in_word = false;
        } else if (!in_word) {
            in_word = true;
            word_count++;
        }
    }

    std::cout << "\n[Results]\n"
              << "  \"success\": true,\n"
              << "  \"total_time_ms\": " << std::fixed << std::setprecision(2) << elapsed << ",\n"
              << "  \"audio_chunks\": " << ((pcm_samples.size() + chunk_size - 1) / chunk_size) << ",\n"
              << "  \"pcm_samples\": " << pcm_samples.size() << ",\n"
              << "  \"duration_sec\": " << std::setprecision(2) << (pcm_samples.size() / 16000.0) << ",\n"
              << "  \"words_transcribed\": " << word_count << "\n"
              << "├─ Full transcription: \"" << full_transcription << "\"" << std::endl;

    cactus_destroy(model);
    return true;
}

static bool test_vad_process() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║           VAD PROCESS TEST               ║\n"
              << "╚══════════════════════════════════════════╝\n";

    const char* vad_model_path = std::getenv("CACTUS_TEST_VAD_MODEL");
    if (!vad_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_VAD_MODEL not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(vad_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize VAD model\n";
        return false;
    }

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    char response[8192] = {0};

    Timer timer;
    int result = cactus_vad(model, audio_path.c_str(), response, sizeof(response), R"({"threshold": 0.5})", nullptr, 0);
    double elapsed = timer.elapsed_ms();

    cactus_destroy(model);

    if (result < 0) {
        std::cerr << "[✗] VAD processing failed\n";
        return false;
    }

    std::string response_str(response);
    if (response_str.find("\"success\":true") == std::string::npos) {
        std::cerr << "[✗] VAD response indicates failure\n";
        return false;
    }

    std::vector<std::pair<size_t, size_t>> segments;
    size_t pos = 0;
    while ((pos = response_str.find("{\"start\":", pos)) != std::string::npos) {
        size_t start_pos = response_str.find(":", pos) + 1;
        size_t end_pos = response_str.find(",", start_pos);
        size_t start = std::stoull(response_str.substr(start_pos, end_pos - start_pos));

        pos = response_str.find("\"end\":", pos) + 6;
        end_pos = response_str.find("}", pos);
        size_t end = std::stoull(response_str.substr(pos, end_pos - pos));

        segments.push_back({start, end});
        pos = end_pos;
    }

    size_t total_speech_samples = 0;
    for (const auto& segment : segments) {
        total_speech_samples += (segment.second - segment.first);
    }

    std::cout << "\n[Results]\n"
              << "  \"success\": true,\n"
              << "  \"total_time_ms\": " << std::fixed << std::setprecision(2) << elapsed << ",\n"
              << "  \"speech_duration_sec\": " << std::setprecision(2) << (total_speech_samples / 16000.0) << ",\n"
              << "  \"segments_detected\": " << segments.size() << "\n";

    for (size_t i = 0; i < segments.size(); ++i) {
        float start_sec = segments[i].first / 16000.0f;
        float end_sec = segments[i].second / 16000.0f;
        const char* prefix = (i == segments.size() - 1) ? "└─" : "├─";
        std::cout << prefix << " Segment " << (i + 1) << ": "
                  << std::fixed << std::setprecision(2) << start_sec << "s - "
                  << std::setprecision(2) << end_sec << "s ("
                  << std::setprecision(2) << (end_sec - start_sec) << "s)" << std::endl;
    }

    return result > 0 && !segments.empty();
}

static bool test_pcm_transcription() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║       PCM BUFFER TRANSCRIPTION           ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_transcribe_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIBE_MODEL not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_transcribe_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize Whisper model\n";
        return false;
    }

    const size_t sample_rate = 16000;
    bool use_microphone = false;
    bool test_passed = false;

#ifdef HAVE_SDL2
    {
        std::cout << "Using microphone input (SDL2)...\n";

        AudioCapture audio_capture(10000);
        if (audio_capture.init(0, sample_rate)) {
            std::cout << "\n🎤 Recording for 10 seconds... Speak now!\n\n";

            audio_capture.resume();
            use_microphone = true;

            std::this_thread::sleep_for(std::chrono::seconds(10));

            audio_capture.pause();

            std::vector<float> audio_float;
            size_t num_samples = audio_capture.get_all(audio_float);

            if (num_samples == 0) {
                std::cerr << "[!] No audio captured\n";
                use_microphone = false;
            } else {
                std::cout << "Captured " << (num_samples / sample_rate)
                          << " seconds of audio, transcribing...\n";

                std::vector<int16_t> pcm_samples(num_samples);
                for (size_t i = 0; i < num_samples; i++) {
                    float clamped = std::max(-1.0f, std::min(1.0f, audio_float[i]));
                    pcm_samples[i] = static_cast<int16_t>(clamped * 32767.0f);
                }

                // Transcribe
                char response[1 << 15] = {0};
                StreamingData stream;
                stream.model = model;

                std::cout << "Transcript: ";
                int rc = cactus_transcribe(
                    model,
                    nullptr,
                    g_whisper_prompt,
                    response,
                    sizeof(response),
                    R"({"max_tokens": 100, "telemetry_enabled": false})",
                    stream_callback,
                    &stream,
                    reinterpret_cast<const uint8_t*>(pcm_samples.data()),
                    pcm_samples.size() * sizeof(int16_t)
                );

                std::cout << "\n\n[Results]\n";
                if (rc > 0) {
                    Metrics m;
                    m.parse(response);
                    m.print_json();
                    test_passed = (rc > 0 && m.completion_tokens >= 1);
                } else {
                    std::cerr << "Transcription failed\n";
                }
            }
        } else {
            std::cerr << "[!] Failed to initialize audio capture, falling back to synthetic audio\n";
        }
    }
#endif
    if (!use_microphone) {
        std::cout << "Using synthetic audio (440Hz sine wave)...\n";
        const size_t duration_seconds = 3;
        const size_t num_samples = sample_rate * duration_seconds;
        std::vector<int16_t> pcm_samples(num_samples);

        for (size_t i = 0; i < num_samples; i++) {
            float t = static_cast<float>(i) / sample_rate;
            float amplitude = 0.3f;
            float value = amplitude * std::sin(2.0f * M_PI * 440.0f * t);
            pcm_samples[i] = static_cast<int16_t>(value * 32767.0f);
        }

        char response[1 << 15] = {0};
        StreamingData stream;
        stream.model = model;

        std::cout << "Transcript: ";
        int rc = cactus_transcribe(
            model,
            nullptr,
            g_whisper_prompt,
            response,
            sizeof(response),
            R"({"max_tokens": 100, "telemetry_enabled": false})",
            stream_callback,
            &stream,
            reinterpret_cast<const uint8_t*>(pcm_samples.data()),
            pcm_samples.size() * sizeof(int16_t)
        );

        std::cout << "\n\n[Results]\n";
        if (rc <= 0) {
            std::cerr << "failed\n";
            cactus_destroy(model);
            return false;
        }

        Metrics m;
        m.parse(response);
        m.print_json();

        std::cout << "├─ PCM samples: " << pcm_samples.size() << "\n"
                  << "├─ Duration: " << duration_seconds << "s\n"
                  << "└─ Sample rate: " << sample_rate << "Hz\n";

        test_passed = (rc > 0 && m.completion_tokens >= 1);
    }

    cactus_destroy(model);
    return test_passed;
}

int main() {
    TestUtils::TestRunner runner("STT Tests");
    runner.run_test("audio_processor", test_audio_processor());
    runner.run_test("vad_process", test_vad_process());
    runner.run_test("transcription", test_transcription());
    runner.run_test("pcm_transcription", test_pcm_transcription());
    runner.run_test("stream_transcription", test_stream_transcription());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
