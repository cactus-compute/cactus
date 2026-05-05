#ifndef WAV_LOADER_H
#define WAV_LOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <cmath>


struct AudioFP32 {
    int sample_rate;
    std::vector<float> samples;
};

inline AudioFP32 load_wav_fp32(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open WAV file: " + path);

    char riff[4];
    file.read(riff, 4);
    if (std::string(riff, 4) != "RIFF") throw std::runtime_error("Not RIFF");

    uint32_t chunk_size;
    file.read(reinterpret_cast<char*>(&chunk_size), 4);

    char wave[4];
    file.read(wave, 4);
    if (std::string(wave, 4) != "WAVE") throw std::runtime_error("Not WAVE");
    
    char fmt_id[4];
    uint32_t fmt_size;
    file.read(fmt_id, 4);
    file.read(reinterpret_cast<char*>(&fmt_size), 4);
    if (std::string(fmt_id, 4) != "fmt ")
        throw std::runtime_error("Missing fmt chunk");

    uint16_t audio_format, num_channels, bits_per_sample;
    uint32_t sample_rate, byte_rate;
    uint16_t block_align;

    file.read(reinterpret_cast<char*>(&audio_format), 2);
    file.read(reinterpret_cast<char*>(&num_channels), 2);
    file.read(reinterpret_cast<char*>(&sample_rate), 4);
    file.read(reinterpret_cast<char*>(&byte_rate), 4);
    file.read(reinterpret_cast<char*>(&block_align), 2);
    file.read(reinterpret_cast<char*>(&bits_per_sample), 2);

    if (audio_format != 1 || bits_per_sample != 16)
        throw std::runtime_error("Only 16-bit PCM WAV supported");

    if (fmt_size > 16)
        file.seekg(fmt_size - 16, std::ios::cur);

    char data_id[4];
    uint32_t data_size;

    while (true) {
        file.read(data_id, 4);
        file.read(reinterpret_cast<char*>(&data_size), 4);
        if (!file) throw std::runtime_error("Malformed WAV: missing data chunk");

        if (std::string(data_id, 4) == "data")
            break;

        file.seekg(data_size, std::ios::cur);
    }

    size_t num_samples = data_size / 2;
    std::vector<int16_t> raw(num_samples);
    file.read(reinterpret_cast<char*>(raw.data()), data_size);

    std::vector<float> tmp(num_samples);
    constexpr float scale = 1.0f / 32768.0f;
    for (size_t i = 0; i < num_samples; i++) {
        tmp[i] = float(raw[i]) * scale;
    }

    std::vector<float> mono;

    if (num_channels == 1) {
        mono = std::move(tmp);
    } else if (num_channels == 2) {
        mono.reserve(num_samples / 2);
        for (size_t i = 0; i < num_samples; i += 2)
            mono.push_back(0.5f * (tmp[i] + tmp[i + 1]));
    } else {
        throw std::runtime_error("Unsupported channel count");
    }

    return AudioFP32{ (int)sample_rate, std::move(mono) };
}

inline AudioFP32 load_wav(const std::string& path) {
    return load_wav_fp32(path);
}

inline std::vector<float> resample_to_16k_fp32(
    const std::vector<float>& in, int sr_in)
{
    const int sr_out = 16000;
    if (sr_in == sr_out) return in;

    const double ratio = double(sr_out) / double(sr_in);
    const size_t out_len = size_t(in.size() * ratio);
    const size_t in_len = in.size();

    if (sr_in <= sr_out) {
        std::vector<float> out(out_len);
        for (size_t i = 0; i < out_len; i++) {
            double pos = i / ratio;
            size_t i0 = (size_t)pos;
            double frac = pos - i0;
            out[i] = (i0 + 1 < in_len)
                ? float((1.0 - frac) * in[i0] + frac * in[i0 + 1])
                : in.back();
        }
        return out;
    }

    const double cutoff = ratio;  
    constexpr int SINC_HALF_WIDTH = 16;
    const double M_PI_VAL = 3.14159265358979323846;

    std::vector<float> out(out_len);

    for (size_t i = 0; i < out_len; i++) {
        double center = i / ratio;
        int left = (int)std::ceil(center) - SINC_HALF_WIDTH;
        int right = (int)std::floor(center) + SINC_HALF_WIDTH;

        double sum = 0.0;
        double weight_sum = 0.0;

        for (int j = left; j <= right; j++) {
            if (j < 0 || j >= (int)in_len) continue;

            double x = center - j;
            double sinc_val;
            if (std::fabs(x) < 1e-9) {
                sinc_val = cutoff;
            } else {
                double pix = M_PI_VAL * x * cutoff;
                sinc_val = cutoff * std::sin(pix) / pix;
            }

            double window_pos = (x + SINC_HALF_WIDTH) / (2.0 * SINC_HALF_WIDTH);
            double window = 0.5 * (1.0 - std::cos(2.0 * M_PI_VAL * window_pos));

            double w = sinc_val * window;
            sum += in[j] * w;
            weight_sum += w;
        }

        out[i] = (weight_sum > 1e-9) ? float(sum / weight_sum) : 0.0f;
    }
    return out;
}

#endif
