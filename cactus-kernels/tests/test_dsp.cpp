#include "test_utils.h"
#include <vector>
#include <cmath>
#include <cstring>

using namespace TestUtils;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool test_rfft_irfft_roundtrip() {
    const size_t n = 256;
    std::vector<float> input(n);
    for (size_t i = 0; i < n; i++) {
        input[i] = std::sin(2.0f * static_cast<float>(M_PI) * 10.0f * i / n)
                 + 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * 30.0f * i / n);
    }

    std::vector<float> freq((n / 2 + 1) * 2);
    cactus_rfft_f32_1d(input.data(), freq.data(), n, "backward");

    std::vector<float> recovered(n);
    cactus_irfft_f32_1d(freq.data(), recovered.data(), n, "backward");

    for (size_t i = 0; i < n; i++) {
        if (std::abs(recovered[i] - input[i]) > 1e-3f) {
            std::cerr << "  roundtrip mismatch at " << i << ": " << recovered[i] << " vs " << input[i] << "\n";
            return false;
        }
    }
    return true;
}

bool test_rfft_dc_signal() {
    const size_t n = 64;
    std::vector<float> input(n, 3.0f);
    std::vector<float> freq((n / 2 + 1) * 2, 0.0f);

    cactus_rfft_f32_1d(input.data(), freq.data(), n, "backward");

    if (std::abs(freq[0] - 3.0f * n) > 1.0f) {
        std::cerr << "  DC bin: " << freq[0] << " expected " << 3.0f * n << "\n";
        return false;
    }

    for (size_t i = 1; i < n / 2 + 1; i++) {
        float mag = std::hypot(freq[i * 2], freq[i * 2 + 1]);
        if (mag > 1.0f) {
            std::cerr << "  non-DC bin " << i << " has magnitude " << mag << "\n";
            return false;
        }
    }
    return true;
}

bool test_mel_filter_bank() {
    const int num_freq_bins = 257;
    const int num_mel_filters = 80;
    const float min_freq = 0.0f;
    const float max_freq = 8000.0f;
    const int sampling_rate = 16000;

    std::vector<float> filters(num_mel_filters * num_freq_bins, 0.0f);
    cactus_generate_mel_filter_bank(
        filters.data(), num_freq_bins, num_mel_filters,
        min_freq, max_freq, sampling_rate, "slaney", "slaney", false);

    for (int m = 0; m < num_mel_filters; m++) {
        bool has_nonzero = false;
        for (int f = 0; f < num_freq_bins; f++) {
            float val = filters[m * num_freq_bins + f];
            if (val < 0.0f) {
                std::cerr << "  negative filter value at mel=" << m << " freq=" << f << "\n";
                return false;
            }
            if (val > 0.0f) has_nonzero = true;
        }
        if (!has_nonzero) {
            std::cerr << "  mel filter " << m << " is all zeros\n";
            return false;
        }
    }
    return true;
}

bool test_hertz_mel_roundtrip() {
    const char* scales[] = {"htk", "kaldi", "slaney"};

    for (const char* scale : scales) {
        for (float freq : {100.0f, 440.0f, 1000.0f, 4000.0f, 8000.0f}) {
            float mel = cactus_hertz_to_mel(freq, scale);
            float back = cactus_mel_to_hertz(mel, scale);
            if (std::abs(back - freq) > 0.5f) {
                std::cerr << "  roundtrip fail for scale=" << scale << " freq=" << freq
                          << " got " << back << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_spectrogram_basic() {
    const size_t n_samples = 1600;
    const size_t n_fft = 400;
    const size_t hop = 160;
    const size_t num_freq_bins = n_fft / 2 + 1;
    const int num_mel_bins = 80;

    std::vector<float> waveform(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        waveform[i] = std::sin(2.0f * static_cast<float>(M_PI) * 440.0f * i / 16000.0f);
    }

    std::vector<float> mel_filters(num_mel_bins * num_freq_bins);
    cactus_generate_mel_filter_bank(
        mel_filters.data(), num_freq_bins, num_mel_bins,
        0.0f, 8000.0f, 16000, "slaney", "slaney", false);

    size_t pad_length = n_fft / 2;
    size_t padded_length = n_samples + 2 * pad_length;
    size_t num_frames = 1 + (padded_length - n_fft) / hop;

    std::vector<float> spectrogram(num_mel_bins * num_frames);
    size_t fft_size = n_fft;

    cactus_compute_spectrogram_f32(
        waveform.data(), n_samples,
        nullptr, 0,
        n_fft, hop, &fft_size,
        spectrogram.data(), 2.0f,
        true, "reflect", true,
        0.0f, nullptr,
        mel_filters.data(), mel_filters.size(),
        1e-10f, nullptr,
        1.0f, 1e-10f, nullptr, false);

    bool has_nonzero = false;
    for (size_t i = 0; i < spectrogram.size(); i++) {
        if (!std::isfinite(spectrogram[i])) {
            std::cerr << "  non-finite spectrogram at " << i << "\n";
            return false;
        }
        if (spectrogram[i] > 1e-8f) has_nonzero = true;
    }

    if (!has_nonzero) {
        std::cerr << "  spectrogram is all zeros\n";
        return false;
    }
    return true;
}

bool test_spectrogram_to_db() {
    std::vector<float> data = {1.0f, 10.0f, 100.0f, 1000.0f};
    cactus_spectrogram_to_db(data.data(), data.size(), 1.0f, 1e-10f, nullptr, 10.0f);

    float expected[] = {0.0f, 10.0f, 20.0f, 30.0f};
    for (size_t i = 0; i < 4; i++) {
        if (std::abs(data[i] - expected[i]) > 0.01f) {
            std::cerr << "  dB mismatch at " << i << ": " << data[i] << " vs " << expected[i] << "\n";
            return false;
        }
    }
    return true;
}

bool run_benchmarks() {
    auto bench = [](const char* label, auto fn) {
        fn();
        TestUtils::Timer t;
        for (int i = 0; i < 100; i++) fn();
        double ms = t.elapsed_ms() / 100.0;
        std::cout << "  ⚡ " << std::left << std::setw(30) << label
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    };

    {
        const size_t n = 512;
        std::vector<float> input(n, 1.0f);
        std::vector<float> output((n / 2 + 1) * 2);
        bench("rfft 512", [&]{ cactus_rfft_f32_1d(input.data(), output.data(), n, "backward"); });
    }

    {
        const size_t n = 512;
        std::vector<float> input((n / 2 + 1) * 2, 0.0f);
        input[0] = 1.0f;
        std::vector<float> output(n);
        bench("irfft 512", [&]{ cactus_irfft_f32_1d(input.data(), output.data(), n, "backward"); });
    }

    return true;
}

int main() {
    TestUtils::TestRunner runner("DSP Kernel Tests");

    runner.run_test("RFFT/IRFFT Roundtrip", test_rfft_irfft_roundtrip());
    runner.run_test("RFFT DC Signal", test_rfft_dc_signal());
    runner.run_test("Mel Filter Bank", test_mel_filter_bank());
    runner.run_test("Hertz/Mel Roundtrip", test_hertz_mel_roundtrip());
    runner.run_test("Spectrogram Basic", test_spectrogram_basic());
    runner.run_test("Spectrogram to dB", test_spectrogram_to_db());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
