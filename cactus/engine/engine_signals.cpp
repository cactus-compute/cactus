#include "engine.h"
#include "../kernel/kernel_utils.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <limits>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    CactusThreading::parallel_for(size, CactusThreading::Thresholds::ALL_REDUCE, [&](size_t start, size_t end) {
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
            size, CactusThreading::Thresholds::ALL_REDUCE,
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
        CactusThreading::parallel_for(size, CactusThreading::Thresholds::ALL_REDUCE, [&](size_t start, size_t end) {
            for (size_t i = start; i < end; i++) {
                spectrogram[i] = std::max(min_db, spectrogram[i]);
            }
        });
    }
}

static size_t bit_reverse(size_t x, size_t log2n) {
    size_t result = 0;
    for (size_t i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

static bool is_power_of_two(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

static size_t log2_power_of_two(size_t n) {
    size_t log2n = 0;
    for (size_t temp = n; temp > 1; temp >>= 1) {
        log2n++;
    }
    return log2n;
}

static void fft_radix2(float* re, float* im, size_t n, bool inverse) {
    if (!is_power_of_two(n)) return;

    const size_t log2n = log2_power_of_two(n);

    for (size_t i = 0; i < n; i++) {
        size_t j = bit_reverse(i, log2n);
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    for (size_t s = 1; s <= log2n; s++) {
        size_t m = size_t{1} << s;
        size_t m2 = m >> 1;
        float w_re = 1.0f;
        float w_im = 0.0f;
        const float angle = static_cast<float>(M_PI) / static_cast<float>(m2);
        const float wm_re = std::cos(angle);
        const float wm_im = inverse ? std::sin(angle) : -std::sin(angle);

        for (size_t j = 0; j < m2; j++) {
            for (size_t k = j; k < n; k += m) {
                size_t k_m2 = k + m2;
                float t_re = w_re * re[k_m2] - w_im * im[k_m2];
                float t_im = w_re * im[k_m2] + w_im * re[k_m2];
                float u_re = re[k];
                float u_im = im[k];
                re[k] = u_re + t_re;
                im[k] = u_im + t_im;
                re[k_m2] = u_re - t_re;
                im[k_m2] = u_im - t_im;
            }
            float new_w_re = w_re * wm_re - w_im * wm_im;
            float new_w_im = w_re * wm_im + w_im * wm_re;
            w_re = new_w_re;
            w_im = new_w_im;
        }
    }
}

enum class FFTNorm {
    BACKWARD,
    FORWARD,
    ORTHO
};

static FFTNorm parse_fft_norm(const char* norm) {
    if (!norm || std::strcmp(norm, "backward") == 0) {
        return FFTNorm::BACKWARD;
    }
    if (std::strcmp(norm, "forward") == 0) {
        return FFTNorm::FORWARD;
    }
    if (std::strcmp(norm, "ortho") == 0) {
        return FFTNorm::ORTHO;
    }
    throw std::invalid_argument("norm must be one of {\"backward\",\"forward\",\"ortho\"}");
}

static size_t next_power_of_two(size_t n) {
    size_t padded_n = 1;
    while (padded_n < n) {
        padded_n <<= 1;
    }
    return padded_n;
}

static float fft_norm_factor(size_t n, FFTNorm norm_mode, bool inverse) {
    if (norm_mode == FFTNorm::ORTHO) {
        return 1.0f / std::sqrt(static_cast<float>(n));
    }
    if (norm_mode == FFTNorm::FORWARD) {
        return inverse ? 1.0f : (1.0f / static_cast<float>(n));
    }
    return inverse ? (1.0f / static_cast<float>(n)) : 1.0f;
}

static void ensure_irfft_scratch(std::vector<float>& re, std::vector<float>& im, size_t n) {
    if (re.size() < n) {
        re.resize(n);
    }
    if (im.size() < n) {
        im.resize(n);
    }
}

static void idft_r_f32_1d_dft(
    const float* input,
    float* output,
    const size_t n,
    const float norm_factor) {
    const size_t in_len = n / 2 + 1;
    const float two_pi_over_n = (2.0f * static_cast<float>(M_PI)) / static_cast<float>(n);
    for (size_t t = 0; t < n; ++t) {
        float sum = input[0];
        for (size_t k = 1; k < in_len; ++k) {
            const float re = input[k * 2];
            const float im = input[k * 2 + 1];
            const float angle = two_pi_over_n * static_cast<float>(k * t);
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const bool self_conjugate = (k * 2 == n);
            if (self_conjugate) {
                sum += re * c;
            } else {
                sum += 2.0f * (re * c - im * s);
            }
        }
        output[t] = sum * norm_factor;
    }
}

static void rfft_f32_1d(const float* input, float* output, const size_t n, const char* norm) {
    const size_t out_len = n / 2 + 1;
    const FFTNorm norm_mode = parse_fft_norm(norm);
    const float norm_factor = fft_norm_factor(n, norm_mode, false);

    if (n == 1) {
        output[0] = input[0] * norm_factor;
        output[1] = 0.0f;
        return;
    }

#ifdef __APPLE__
    {
        size_t log2n = 0;
        size_t padded_n = 1;
        while (padded_n < n) {
            padded_n <<= 1;
            log2n++;
        }

        FFTSetup fft_setup = vDSP_create_fftsetup(log2n, FFT_RADIX2);
        if (fft_setup) {
            std::vector<float> buffer(padded_n, 0.0f);
            std::copy(input, input + n, buffer.begin());

            DSPSplitComplex split;
            std::vector<float> real_part(padded_n / 2);
            std::vector<float> imag_part(padded_n / 2);
            split.realp = real_part.data();
            split.imagp = imag_part.data();

            vDSP_ctoz(reinterpret_cast<const DSPComplex*>(buffer.data()), 2, &split, 1, padded_n / 2);

            vDSP_fft_zrip(fft_setup, &split, 1, log2n, FFT_FORWARD);

            float scale = 0.5f * norm_factor;

            output[0] = split.realp[0] * scale * 2.0f;
            output[1] = 0.0f;

            if (out_len > 1) {
                size_t nyquist_idx = padded_n / 2;
                if (nyquist_idx < out_len) {
                    output[nyquist_idx * 2] = split.imagp[0] * scale * 2.0f;
                    output[nyquist_idx * 2 + 1] = 0.0f;
                }
            }

            for (size_t i = 1; i < out_len && i < padded_n / 2; i++) {
                output[i * 2] = split.realp[i] * scale * 2.0f;
                output[i * 2 + 1] = split.imagp[i] * scale * 2.0f;
            }

            vDSP_destroy_fftsetup(fft_setup);
            return;
        }
    }
#endif

    const size_t padded_n = next_power_of_two(n);

    std::vector<float> re(padded_n, 0.0f), im(padded_n, 0.0f);
    std::copy(input, input + n, re.begin());

    fft_radix2(re.data(), im.data(), padded_n, false);

    for (size_t i = 0; i < out_len; i++) {
        output[i * 2] = re[i] * norm_factor;
        output[i * 2 + 1] = im[i] * norm_factor;
    }
}

static void idft_r_f32_1d(const float* input, float* output, const size_t n, const char* norm) {
    const size_t in_len = n / 2 + 1;
    const FFTNorm norm_mode = parse_fft_norm(norm);
    const float norm_factor = fft_norm_factor(n, norm_mode, true);

    if (n == 1) {
        output[0] = input[0] * norm_factor;
        return;
    }

    if (!is_power_of_two(n)) {
        idft_r_f32_1d_dft(input, output, n, norm_factor);
        return;
    }

    const size_t padded_n = next_power_of_two(n);
    const size_t padded_half_plus_one = padded_n / 2 + 1;
    thread_local std::vector<float> re_scratch;
    thread_local std::vector<float> im_scratch;
    ensure_irfft_scratch(re_scratch, im_scratch, padded_n);
    std::fill(re_scratch.begin(), re_scratch.begin() + padded_n, 0.0f);
    std::fill(im_scratch.begin(), im_scratch.begin() + padded_n, 0.0f);
    float* re = re_scratch.data();
    float* im = im_scratch.data();
    for (size_t i = 0; i < in_len; ++i) {
        re[i] = input[i * 2];
        im[i] = input[i * 2 + 1];
    }

    im[0] = 0.0f;
    const bool has_padded_nyquist_bin = in_len == padded_half_plus_one;
    if (has_padded_nyquist_bin) {
        im[padded_n / 2] = 0.0f;
    }

    const size_t mirror_limit = has_padded_nyquist_bin ? (in_len - 1) : in_len;
    for (size_t i = 1; i < mirror_limit; ++i) {
        const size_t mirrored = padded_n - i;
        re[mirrored] = re[i];
        im[mirrored] = -im[i];
    }

    fft_radix2(re, im, padded_n, true);

    for (size_t i = 0; i < n; ++i) {
        output[i] = re[i] * norm_factor;
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
            hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (length - 1)));
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
        padded_waveform.assign(padded_length, 0.0f);

        if (std::strcmp(pad_mode, "reflect") == 0) {
            for (size_t i = 0; i < pad_length; i++) {
                padded_waveform[i] = waveform[pad_length - i];
            }

            std::copy(waveform, waveform + waveform_length, padded_waveform.data() + pad_length);

            for (size_t i = 0; i < pad_length; i++) {
                padded_waveform[pad_length + waveform_length + i] = waveform[waveform_length - 2 - i];
            }
        } else if (std::strcmp(pad_mode, "constant") == 0) {
            std::copy(waveform, waveform + waveform_length, padded_waveform.data() + pad_length);
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

    CactusThreading::parallel_for(num_frames, CactusThreading::Thresholds::SCALAR_EXPENSIVE, [&](size_t start_frame, size_t end_frame) {
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
                    float randn = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * static_cast<float>(M_PI) * u2);
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
        CactusThreading::parallel_for_2d(num_mel_bins, num_frames, CactusThreading::Thresholds::AXIS_REDUCE, [&](size_t m, size_t t) {
            float sum = 0.0f;
            for (size_t f = 0; f < num_frequency_bins; f++) {
                sum += mel_filters[m * num_frequency_bins + f] * temp_spectrogram[t * num_frequency_bins + f];
            }
            spectrogram[m * num_frames + t] = std::max(mel_floor, sum);
        });
    } else {
        CactusThreading::parallel_for_2d(num_frames, num_frequency_bins, CactusThreading::Thresholds::AXIS_REDUCE, [&](size_t t, size_t f) {
            spectrogram[f * num_frames + t] = temp_spectrogram[t * num_frequency_bins + f];
        });
    }

    if (power != 0.0f && log_mel != nullptr) {
        const size_t total_elements = spectrogram_bins * num_frames;

        if (std::strcmp(log_mel, "log") == 0) {
            CactusThreading::parallel_for(total_elements, CactusThreading::Thresholds::ALL_REDUCE, [&](size_t start, size_t end) {
                for (size_t i = start; i < end; i++) {
                    spectrogram[i] = std::log(spectrogram[i]);
                }
            });
        } else if (std::strcmp(log_mel, "log10") == 0) {
            CactusThreading::parallel_for(total_elements, CactusThreading::Thresholds::ALL_REDUCE, [&](size_t start, size_t end) {
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
    const size_t analysis_frame_length = config.n_fft;
    const size_t window_length = std::min(config.frame_length, analysis_frame_length);
    const size_t pad_length = config.center ? analysis_frame_length / 2 : 0;
    const size_t padded_length = n_samples + 2 * pad_length;
    if (analysis_frame_length == 0 || padded_length < analysis_frame_length || config.hop_length == 0) {
        return {};
    }
    const size_t num_frames = 1 + (padded_length - analysis_frame_length) / config.hop_length;

    std::vector<float> output(num_mel_filters_ * num_frames);
    std::vector<float> window(analysis_frame_length, 0.0f);
    if (window_length == 1) {
        window[analysis_frame_length / 2] = 1.0f;
    } else if (window_length > 1) {
        const float denom = config.hann_periodic
            ? static_cast<float>(window_length)
            : static_cast<float>(window_length - 1);
        const size_t left_pad = (analysis_frame_length - window_length) / 2;
        for (size_t i = 0; i < window_length; ++i) {
            const float w = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / denom));
            window[left_pad + i] = w;
        }
    }

    compute_spectrogram_f32(
        waveform.data(),
        waveform.size(),
        window.data(),
        window.size(),
        analysis_frame_length,
        config.hop_length,
        &config.n_fft,
        output.data(),
        config.power,
        config.center,
        config.pad_mode,
        config.onesided,
        config.dither,
        config.preemphasis != 0.0f ? &config.preemphasis : nullptr,
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

std::vector<float> AudioProcessor::compute_irfft(
    const std::vector<float>& complex_input,
    size_t n,
    const char* norm) {
    if (n == 0) {
        throw std::invalid_argument("compute_irfft: n must be greater than 0");
    }
    const size_t expected_in_len = (n / 2 + 1) * 2;
    if (complex_input.size() != expected_in_len) {
        throw std::invalid_argument("compute_irfft: complex_input must contain exactly (n/2+1)*2 values");
    }
    std::vector<float> output(n);
    idft_r_f32_1d(complex_input.data(), output.data(), n, norm);
    return output;
}

}
}
