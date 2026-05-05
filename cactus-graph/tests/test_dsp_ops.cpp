#include "test_utils.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace TestUtils;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool test_rfft_graph_op() {
    CactusGraph g;

    const size_t n = 64;
    size_t input = g.input({n}, Precision::FP32);

    std::vector<float> data(n);
    for (size_t i = 0; i < n; i++) {
        data[i] = std::sin(2.0f * static_cast<float>(M_PI) * 5.0f * i / n);
    }
    g.set_input(input, data.data(), Precision::FP32);

    size_t rfft_out = g.rfft(input);
    g.execute();

    float* output = static_cast<float*>(g.get_output(rfft_out));
    const auto& buf = g.get_output_buffer(rfft_out);
    if (buf.total_size != (n / 2 + 1) * 2) return false;

    bool has_nonzero = false;
    for (size_t i = 0; i < buf.total_size; i++) {
        if (!std::isfinite(output[i])) return false;
        if (std::abs(output[i]) > 1e-3f) has_nonzero = true;
    }
    return has_nonzero;
}

bool test_irfft_graph_op() {
    CactusGraph g;

    const size_t n = 64;
    size_t input = g.input({n}, Precision::FP32);

    std::vector<float> data(n);
    for (size_t i = 0; i < n; i++) {
        data[i] = std::sin(2.0f * static_cast<float>(M_PI) * 3.0f * i / n);
    }
    g.set_input(input, data.data(), Precision::FP32);

    size_t freq = g.rfft(input);
    size_t recovered = g.irfft(freq, n);
    g.execute();

    float* result = static_cast<float*>(g.get_output(recovered));
    for (size_t i = 0; i < n; i++) {
        if (std::abs(result[i] - data[i]) > 1e-3f) return false;
    }
    return true;
}

bool test_mel_filter_bank_graph_op() {
    CactusGraph g;

    const size_t num_freq = 257;
    const size_t num_mel = 80;

    size_t mel_node = g.mel_filter_bank(num_freq, num_mel, 0.0f, 8000.0f, 16000);
    g.execute();

    const auto& buf = g.get_output_buffer(mel_node);
    if (buf.shape[0] != num_freq || buf.shape[1] != num_mel) return false;

    float* filters = static_cast<float*>(g.get_output(mel_node));
    bool has_nonzero = false;
    for (size_t i = 0; i < buf.total_size; i++) {
        if (filters[i] < 0.0f) return false;
        if (filters[i] > 0.0f) has_nonzero = true;
    }
    return has_nonzero;
}

bool test_spectrogram_graph_op() {
    CactusGraph g;

    const size_t n_samples = 1600;
    const size_t frame_length = 400;
    const size_t hop = 160;
    const size_t fft_len = 400;
    const size_t num_freq = fft_len / 2 + 1;
    const size_t num_mel = 80;

    size_t mel_node = g.mel_filter_bank(num_freq, num_mel, 0.0f, 8000.0f, 16000);

    size_t wav_input = g.input({n_samples}, Precision::FP32);
    std::vector<float> waveform(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        waveform[i] = std::sin(2.0f * static_cast<float>(M_PI) * 440.0f * i / 16000.0f);
    }
    g.set_input(wav_input, waveform.data(), Precision::FP32);

    size_t spec = g.spectrogram(wav_input, mel_node, frame_length, hop, fft_len);
    g.execute();

    float* output = static_cast<float*>(g.get_output(spec));
    const auto& buf = g.get_output_buffer(spec);

    if (buf.shape[0] != num_mel) return false;

    bool has_nonzero = false;
    for (size_t i = 0; i < buf.total_size; i++) {
        if (!std::isfinite(output[i])) return false;
        if (output[i] > 1e-8f) has_nonzero = true;
    }
    return has_nonzero;
}

bool test_spectrogram_with_log() {
    CactusGraph g;

    const size_t n_samples = 1600;
    const size_t frame_length = 400;
    const size_t hop = 160;
    const size_t fft_len = 400;
    const size_t num_freq = fft_len / 2 + 1;
    const size_t num_mel = 80;

    size_t mel_node = g.mel_filter_bank(num_freq, num_mel, 0.0f, 8000.0f, 16000);

    size_t wav_input = g.input({n_samples}, Precision::FP32);
    std::vector<float> waveform(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        waveform[i] = std::sin(2.0f * static_cast<float>(M_PI) * 440.0f * i / 16000.0f);
    }
    g.set_input(wav_input, waveform.data(), Precision::FP32);

    size_t spec = g.spectrogram(wav_input, mel_node, frame_length, hop, fft_len,
                                 2.0f, true, 0, 1e-10f, 1);
    g.execute();

    float* output = static_cast<float*>(g.get_output(spec));
    const auto& buf = g.get_output_buffer(spec);

    bool has_negative = false;
    for (size_t i = 0; i < buf.total_size; i++) {
        if (!std::isfinite(output[i])) return false;
        if (output[i] < 0.0f) has_negative = true;
    }
    return has_negative;
}

bool run_benchmarks() {
    auto bench = [](const char* label, auto setup, auto run) {
        setup();
        run();
        TestUtils::Timer t;
        for (int i = 0; i < 100; i++) run();
        double ms = t.elapsed_ms() / 100.0;
        std::cout << "  ⚡ " << std::left << std::setw(30) << label
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    };

    {
        const size_t n = 512;
        std::vector<float> data(n, 1.0f);

        bench("rfft graph op 512", []{}, [&]{
            CactusGraph g;
            size_t inp = g.input({n}, Precision::FP32);
            g.set_input(inp, data.data(), Precision::FP32);
            g.rfft(inp);
            g.execute();
        });
    }

    {
        const size_t n_samples = 16000;
        const size_t num_freq = 201;
        const size_t num_mel = 80;
        std::vector<float> waveform(n_samples, 0.1f);

        CactusGraph g;
        size_t mel = g.mel_filter_bank(num_freq, num_mel, 0.0f, 8000.0f, 16000);
        size_t wav = g.input({n_samples}, Precision::FP32);
        g.set_input(wav, waveform.data(), Precision::FP32);
        g.spectrogram(wav, mel, 400, 160, 400);

        bench("spectrogram 1s@16kHz", []{}, [&]{
            g.execute();
        });
    }

    return true;
}

int main() {
    TestUtils::TestRunner runner("DSP Graph Ops Tests");

    runner.run_test("RFFT Graph Op", test_rfft_graph_op());
    runner.run_test("IRFFT Roundtrip Graph Op", test_irfft_graph_op());
    runner.run_test("Mel Filter Bank Graph Op", test_mel_filter_bank_graph_op());
    runner.run_test("Spectrogram Graph Op", test_spectrogram_graph_op());
    runner.run_test("Spectrogram with Log", test_spectrogram_with_log());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
