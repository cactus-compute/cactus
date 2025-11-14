#include "engine.h"
#include "../kernel/kernel_utils.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <numbers>
#include <limits>

namespace cactus {
namespace engine {

static void to_db(
    float* spectrogram,
    size_t size,
    float reference,
    float min_value,
    const float* db_range,
    float multiplier)
{
    if (reference <= 0.0f) {
        throw std::invalid_argument("reference must be greater than zero");
    }
    if (min_value <= 0.0f) {
        throw std::invalid_argument("min_value must be greater than zero");
    }

    reference = std::max(min_value, reference);
    const float log_ref = std::log10(reference);

    CactusThreading::parallel_for(size, 10000, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            float value = std::max(min_value, spectrogram[i]);
            spectrogram[i] = multiplier * (std::log10(value) - log_ref);
        }
    });

    if (db_range != nullptr) {
        if (*db_range <= 0.0f) {
            throw std::invalid_argument("db_range must be greater than zero");
        }

        float max_db = CactusThreading::parallel_reduce<std::function<float(size_t, size_t)>, float, std::function<float(float, float)>>(
            size, 10000,
            [&](size_t start, size_t end) {
                float local_max = -std::numeric_limits<float>::infinity();
                for (size_t i = start; i < end; i++) {
                    local_max = std::max(local_max, spectrogram[i]);
                }
                return local_max;
            },
            -std::numeric_limits<float>::infinity(),
            [](float a, float b) { return std::max(a, b); }
        );

        float min_db = max_db - *db_range;
        CactusThreading::parallel_for(size, 10000, [&](size_t start, size_t end) {
            for (size_t i = start; i < end; i++) {
                spectrogram[i] = std::max(min_db, spectrogram[i]);
            }
        });
    }
}

static void rfft_f32_1d(const float* input, float* output, const size_t n, const char* norm) {
    const size_t out_len = n / 2 + 1;
    const float two_pi_over_n = 2.0f * std::numbers::pi / static_cast<float>(n);

    float norm_factor = 1.0f;
    if (norm) {
        if (std::strcmp(norm, "backward") == 0) {
            norm_factor = 1.0f;
        } else if (std::strcmp(norm, "forward") == 0) {
            norm_factor = 1.0f / static_cast<float>(n);
        } else if (std::strcmp(norm, "ortho") == 0) {
            norm_factor = 1.0f / std::sqrt(static_cast<float>(n));
        } else {
            throw std::invalid_argument("norm must be one of {\"backward\",\"forward\",\"ortho\"}");
        }
    }

    for (size_t i = 0; i < out_len; i++) {
        float re = 0.0f;
        float im = 0.0f;
        const float base = -two_pi_over_n * static_cast<float>(i);
        for (size_t j = 0; j < n; j++) {
            const float angle = base * static_cast<float>(j);
            const float input_val = input[j];
            re += input_val * std::cos(angle);
            im += input_val * std::sin(angle);
        }

        output[i * 2] = re * norm_factor;
        output[i * 2 + 1] = im * norm_factor;
    }
}

static float hertz_to_mel(float freq, const char* mel_scale) {
    if (std::strcmp(mel_scale, "htk") == 0) {
        return 2595.0f * std::log10(1.0f + (freq / 700.0f));
    } else if (std::strcmp(mel_scale, "kaldi") == 0) {
        return 1127.0f * std::log(1.0f + (freq / 700.0f));
    }

    const float min_log_hertz = 1000.0f;
    const float min_log_mel = 15.0f;
    const float logstep = 27.0f / std::log(6.4f);
    float mels = 3.0f * freq / 200.0f;

    if (freq >= min_log_hertz) {
        mels = min_log_mel + std::log(freq / min_log_hertz) * logstep;
    }

    return mels;
}

static float mel_to_hertz(float mels, const char* mel_scale) {
    if (std::strcmp(mel_scale, "htk") == 0) {
        return 700.0f * (std::pow(10.0f, mels / 2595.0f) - 1.0f);
    } else if (std::strcmp(mel_scale, "kaldi") == 0) {
        return 700.0f * (std::exp(mels / 1127.0f) - 1.0f);
    }

    const float min_log_hertz = 1000.0f;
    const float min_log_mel = 15.0f;
    const float logstep = std::log(6.4f) / 27.0f;
    float freq = 200.0f * mels / 3.0f;

    if (mels >= min_log_mel) {
        freq = min_log_hertz * std::exp(logstep * (mels - min_log_mel));
    }

    return freq;
}

static void generate_mel_filter_bank(
    float* mel_filters,
    const int num_frequency_bins,
    const int num_mel_filters,
    const float min_frequency,
    const float max_frequency,
    const int sampling_rate,
    const char* norm,
    const char* mel_scale,
    const bool triangularize_in_mel_space)
{
    if (norm != nullptr && std::strcmp(norm, "slaney") != 0) {
        throw std::invalid_argument("norm must be one of None or \"slaney\"");
    }

    if (std::strcmp(mel_scale, "htk") != 0 && std::strcmp(mel_scale, "kaldi") != 0 && std::strcmp(mel_scale, "slaney") != 0) {
        throw std::invalid_argument("mel_scale should be one of \"htk\", \"slaney\" or \"kaldi\".");
    }

    if (num_frequency_bins < 2) {
        throw std::invalid_argument(
            "Require num_frequency_bins: " + std::to_string(num_frequency_bins) + " >= 2");
    }

    if (min_frequency > max_frequency) {
        throw std::invalid_argument(
            "Require min_frequency: " + std::to_string(min_frequency) +
            " <= max_frequency: " + std::to_string(max_frequency));
    }

    const float mel_min = hertz_to_mel(min_frequency, mel_scale);
    const float mel_max = hertz_to_mel(max_frequency, mel_scale);

    std::vector<float> mel_freqs(num_mel_filters + 2);
    for (int i = 0; i < num_mel_filters + 2; i++) {
        mel_freqs[i] = mel_min + (mel_max - mel_min) * i / (num_mel_filters + 1);
    }

    std::vector<float> filter_freqs(num_mel_filters + 2);
    for (int i = 0; i < num_mel_filters + 2; i++) {
        filter_freqs[i] = mel_to_hertz(mel_freqs[i], mel_scale);
    }

    std::vector<float> fft_freqs(num_frequency_bins);
    if (triangularize_in_mel_space) {
        float fft_bin_width = static_cast<float>(sampling_rate) / ((num_frequency_bins - 1) * 2);
        for (int i = 0; i < num_frequency_bins; i++) {
            fft_freqs[i] = hertz_to_mel(fft_bin_width * i, mel_scale);
        }
        filter_freqs = mel_freqs;
    } else {
        for (int i = 0; i < num_frequency_bins; i++) {
            fft_freqs[i] = (static_cast<float>(sampling_rate) / 2.0f) * i / (num_frequency_bins - 1);
        }
    }

    for (int i = 0; i < num_mel_filters; i++) {
        float left_edge = filter_freqs[i];
        float center = filter_freqs[i + 1];
        float right_edge = filter_freqs[i + 2];

        for (int j = 0; j < num_frequency_bins; j++) {
            float freq = fft_freqs[j];
            float down_slope = (freq - left_edge) / (center - left_edge);
            float up_slope = (right_edge - freq) / (right_edge - center);

            mel_filters[i * num_frequency_bins + j] = std::max(0.0f, std::min(down_slope, up_slope));
        }
    }

    if (norm != nullptr && std::strcmp(norm, "slaney") == 0) {
        for (int i = 0; i < num_mel_filters; i++) {
            float enorm = 2.0f / (filter_freqs[i + 2] - filter_freqs[i]);
            for (int j = 0; j < num_frequency_bins; j++) {
                mel_filters[i * num_frequency_bins + j] *= enorm;
            }
        }
    }
}

static void compute_spectrogram_f32(
    const float* waveform,
    size_t waveform_length,
    const float* window,
    size_t window_length,
    size_t frame_length,
    size_t hop_length,
    const size_t* fft_length,
    float* spectrogram,
    float power,
    bool center,
    const char* pad_mode,
    bool onesided [[maybe_unused]],
    float dither,
    const float* preemphasis,
    const float* mel_filters,
    size_t mel_filters_size,
    float mel_floor,
    const char* log_mel,
    float reference,
    float min_value,
    const float* db_range,
    bool remove_dc_offset)
{
    size_t actual_fft_length;
    if (fft_length == nullptr) {
        actual_fft_length = frame_length;
    } else {
        actual_fft_length = *fft_length;
    }

    if (frame_length > actual_fft_length) {
        throw std::invalid_argument(
            "frame_length (" + std::to_string(frame_length) +
            ") may not be larger than fft_length (" +
            std::to_string(actual_fft_length) + ")");
    }

    std::vector<float> hann_window;
    const float* actual_window = window;

    if (window == nullptr) {
        size_t length = frame_length + 1;
        hann_window.resize(frame_length);
        for (size_t i = 0; i < frame_length; i++) {
            hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * std::numbers::pi * i / (length - 1)));
        }
        actual_window = hann_window.data();
    } else if (window_length != frame_length) {
        throw std::invalid_argument(
            "Length of the window (" + std::to_string(window_length) +
            ") must equal frame_length (" + std::to_string(frame_length) + ")");
    }

    if (hop_length <= 0) {
        throw std::invalid_argument("hop_length must be greater than zero");
    }

    if (power == 0.0f && mel_filters != nullptr) {
        throw std::invalid_argument(
            "You have provided `mel_filters` but `power` is `None`. "
            "Mel spectrogram computation is not yet supported for complex-valued spectrogram. "
            "Specify `power` to fix this issue.");
    }

    std::vector<float> padded_waveform;
    const float* input_waveform = waveform;
    size_t input_length = waveform_length;

    if (center) {
        size_t pad_length = frame_length / 2;
        size_t padded_length = waveform_length + 2 * pad_length;
        padded_waveform.resize(padded_length);

        if (std::strcmp(pad_mode, "reflect") == 0) {
            for (size_t i = 0; i < pad_length; i++) {
                padded_waveform[i] = waveform[pad_length - i];
            }

            std::copy(waveform, waveform + waveform_length, padded_waveform.data() + pad_length);

            for (size_t i = 0; i < pad_length; i++) {
                padded_waveform[pad_length + waveform_length + i] = waveform[waveform_length - 2 - i];
            }
        } else {
            throw std::invalid_argument("Unsupported pad_mode: " + std::string(pad_mode));
        }

        input_waveform = padded_waveform.data();
        input_length = padded_length;
    }

    const size_t num_frames = 1 + (input_length - frame_length) / hop_length;
    const size_t num_frequency_bins = (actual_fft_length / 2) + 1;

    std::vector<float> buffer(actual_fft_length);
    std::vector<float> raw_complex_frequencies(num_frequency_bins * 2);

    const size_t num_mel_bins = mel_filters != nullptr ? mel_filters_size / num_frequency_bins : 0;
    const size_t spectrogram_bins = mel_filters != nullptr ? num_mel_bins : num_frequency_bins;

    std::vector<float> temp_spectrogram(num_frames * num_frequency_bins);

    CactusThreading::parallel_for(num_frames, 100, [&](size_t start_frame, size_t end_frame) {
        std::vector<float> local_buffer(actual_fft_length);
        std::vector<float> local_complex_frequencies(num_frequency_bins * 2);

        for (size_t frame_idx = start_frame; frame_idx < end_frame; frame_idx++) {
            size_t timestep = frame_idx * hop_length;
            std::fill(local_buffer.begin(), local_buffer.end(), 0.0f);

            size_t available_length = std::min(frame_length, input_length - timestep);
            std::copy(input_waveform + timestep, input_waveform + timestep + available_length, local_buffer.data());

            if (dither != 0.0f) {
                for (size_t i = 0; i < frame_length; i++) {
                    float u1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                    float u2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                    float randn = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * std::numbers::pi * u2);
                    local_buffer[i] += dither * randn;
                }
            }

            if (remove_dc_offset) {
                float mean = 0.0f;
                for (size_t i = 0; i < frame_length; i++) {
                    mean += local_buffer[i];
                }
                mean /= static_cast<float>(frame_length);

                for (size_t i = 0; i < frame_length; i++) {
                    local_buffer[i] -= mean;
                }
            }

            if (preemphasis != nullptr) {
                float preemph_coef = *preemphasis;
                for (size_t i = frame_length - 1; i > 0; i--) {
                    local_buffer[i] -= preemph_coef * local_buffer[i - 1];
                }
                local_buffer[0] *= (1.0f - preemph_coef);
            }

            for (size_t i = 0; i < frame_length; i++) {
                local_buffer[i] *= actual_window[i];
            }

            rfft_f32_1d(local_buffer.data(), local_complex_frequencies.data(), actual_fft_length, "backward");

            for (size_t i = 0; i < num_frequency_bins; i++) {
                float real = local_complex_frequencies[i * 2];
                float imag = local_complex_frequencies[i * 2 + 1];
                float magnitude = std::hypot(real, imag);
                temp_spectrogram[frame_idx * num_frequency_bins + i] = std::pow(magnitude, power);
            }
        }
    });

    if (mel_filters != nullptr) {
        CactusThreading::parallel_for_2d(num_mel_bins, num_frames, 1000, [&](size_t m, size_t t) {
            float sum = 0.0f;
            for (size_t f = 0; f < num_frequency_bins; f++) {
                sum += mel_filters[m * num_frequency_bins + f] * temp_spectrogram[t * num_frequency_bins + f];
            }
            spectrogram[m * num_frames + t] = std::max(mel_floor, sum);
        });
    } else {
        CactusThreading::parallel_for_2d(num_frames, num_frequency_bins, 1000, [&](size_t t, size_t f) {
            spectrogram[f * num_frames + t] = temp_spectrogram[t * num_frequency_bins + f];
        });
    }

    if (power != 0.0f && log_mel != nullptr) {
        const size_t total_elements = spectrogram_bins * num_frames;

        if (std::strcmp(log_mel, "log") == 0) {
            CactusThreading::parallel_for(total_elements, 10000, [&](size_t start, size_t end) {
                for (size_t i = start; i < end; i++) {
                    spectrogram[i] = std::log(spectrogram[i]);
                }
            });
        } else if (std::strcmp(log_mel, "log10") == 0) {
            CactusThreading::parallel_for(total_elements, 10000, [&](size_t start, size_t end) {
                for (size_t i = start; i < end; i++) {
                    spectrogram[i] = std::log10(spectrogram[i]);
                }
            });
        } else if (std::strcmp(log_mel, "dB") == 0) {
            if (power == 1.0f) {
                to_db(spectrogram, total_elements, reference, min_value, db_range, 20.0f);
            } else if (power == 2.0f) {
                to_db(spectrogram, total_elements, reference, min_value, db_range, 10.0f);
            } else {
                throw std::invalid_argument(
                    "Cannot use log_mel option 'dB' with power " + std::to_string(power));
            }
        } else {
            throw std::invalid_argument("Unknown log_mel option: " + std::string(log_mel));
        }
    }
}

AudioProcessor::AudioProcessor()
    : num_frequency_bins_(0), num_mel_filters_(0) {}

AudioProcessor::~AudioProcessor() = default;

void AudioProcessor::init_mel_filters(size_t num_frequency_bins,
                                      size_t num_mel_filters,
                                      float min_freq,
                                      float max_freq,
                                      size_t sampling_rate) {
    num_frequency_bins_ = num_frequency_bins;
    num_mel_filters_ = num_mel_filters;
    mel_filters_.resize(num_mel_filters * num_frequency_bins);

    generate_mel_filter_bank(
        mel_filters_.data(),
        num_frequency_bins,
        num_mel_filters,
        min_freq,
        max_freq,
        sampling_rate,
        "slaney",
        "slaney",
        false
    );
}

std::vector<float> AudioProcessor::compute_spectrogram(
    const std::vector<float>& waveform,
    const SpectrogramConfig& config) {

    if (mel_filters_.empty()) {
        throw std::runtime_error("Mel filters not initialized. Call init_mel_filters() first.");
    }

    const size_t n_samples = waveform.size();
    const size_t pad_length = config.center ? config.frame_length / 2 : 0;
    const size_t padded_length = n_samples + 2 * pad_length;
    const size_t num_frames = 1 + (padded_length - config.frame_length) / config.hop_length;

    std::vector<float> output(num_mel_filters_ * num_frames);

    compute_spectrogram_f32(
        waveform.data(),
        waveform.size(),
        nullptr,
        0,
        config.frame_length,
        config.hop_length,
        &config.n_fft,
        output.data(),
        config.power,
        config.center,
        config.pad_mode,
        config.onesided,
        config.dither,
        nullptr,
        mel_filters_.data(),
        mel_filters_.size(),
        config.mel_floor,
        config.log_mel,
        config.reference,
        config.min_value,
        nullptr,
        config.remove_dc_offset
    );

    return output;
}

}
}
