#include "model.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace cactus {
namespace engine {

namespace {

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

size_t tensor_numel(const std::vector<size_t>& shape) {
    size_t count = 1;
    for (size_t dim : shape) {
        count *= dim;
    }
    return count;
}

constexpr int kCloudHandoffSampleRate = 16000;
constexpr size_t kOverlapNfft = 512;
constexpr size_t kOverlapHopLength = 128;
constexpr size_t kOverlapFrameLength = 512;
constexpr size_t kNoiseNfft = 1024;
constexpr size_t kNoiseHopLength = 256;
constexpr size_t kNoiseFrameLength = 1024;

void compute_noise_stft_power(
    const AudioProcessor& audio_processor,
    const std::vector<float>& waveform,
    std::vector<float>& stft_power,
    std::vector<float>& stft_freqs_hz,
    size_t& stft_num_frames) {

    AudioProcessor::SpectrogramConfig cfg;
    cfg.n_fft = kNoiseNfft;
    cfg.hop_length = kNoiseHopLength;
    cfg.frame_length = kNoiseFrameLength;
    audio_processor.compute_stft_power(
        waveform,
        kCloudHandoffSampleRate,
        cfg,
        stft_power,
        stft_freqs_hz,
        stft_num_frames);
}

} // namespace

WhisperCloudHandoffModel::WhisperCloudHandoffModel() : Model() {}

WhisperCloudHandoffModel::WhisperCloudHandoffModel(const Config& config) : Model(config) {}

WhisperCloudHandoffModel::~WhisperCloudHandoffModel() = default;

bool WhisperCloudHandoffModel::init(const std::string& model_folder, size_t context_size,
                                    const std::string& system_prompt, bool do_warmup) {
    (void)context_size;
    (void)system_prompt;
    (void)do_warmup;
    std::string error;
    return init(model_folder, &error);
}

bool WhisperCloudHandoffModel::parse_json_float(const std::string& json, const std::string& key, float& out_value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch m;
    if (!std::regex_search(json, m, re) || m.size() < 2) {
        return false;
    }
    try {
        out_value = std::stof(m[1].str());
        return true;
    } catch (...) {
        return false;
    }
}

void WhisperCloudHandoffModel::set_error(std::string* error, const std::string& message) const {
    if (error != nullptr) {
        *error = message;
    }
}

bool WhisperCloudHandoffModel::init(const std::string& model_folder, std::string* error) {
    if (initialized_) {
        return true;
    }

    initialized_ = false;
    feature_names_.clear();
    input_dim_ = 0;
    hidden_dim_ = 0;
    output_dim_ = 0;
    threshold_ = 0.5f;
    high_freq_cutoff_hz_ = 3000.0f;
    graph_.hard_reset();
    model_folder_path_ = model_folder;

    const std::string features_file = model_folder + "/classifier_features.json";
    const std::string meta_file = model_folder + "/classifier_meta.json";

    try {
        load_weights_to_graph(&graph_);
    } catch (const std::exception& e) {
        set_error(error, "Failed to map cloud_handoff graph weights: " + std::string(e.what()));
        return false;
    }

    const auto& fc1w_shape = graph_.get_output_buffer(weight_nodes_.fc1_weight).shape;
    if (fc1w_shape.size() != 2) {
        set_error(error, "cloud_handoff_fc1.weights must be rank-2");
        return false;
    }
    hidden_dim_ = fc1w_shape[0];
    input_dim_ = fc1w_shape[1];

    const auto& fc1b_shape = graph_.get_output_buffer(weight_nodes_.fc1_bias).shape;
    if (fc1b_shape.size() != 1 || fc1b_shape[0] != hidden_dim_) {
        set_error(error, "cloud_handoff_fc1.bias shape mismatch");
        return false;
    }

    const auto& fc2w_shape = graph_.get_output_buffer(weight_nodes_.fc2_weight).shape;
    if (fc2w_shape.size() != 2 || fc2w_shape[1] != hidden_dim_) {
        set_error(error, "cloud_handoff_fc2.weights shape mismatch");
        return false;
    }
    output_dim_ = fc2w_shape[0];

    const auto& fc2b_shape = graph_.get_output_buffer(weight_nodes_.fc2_bias).shape;
    if (fc2b_shape.size() != 1 || fc2b_shape[0] != output_dim_) {
        set_error(error, "cloud_handoff_fc2.bias shape mismatch");
        return false;
    }

    const auto& mean_shape = graph_.get_output_buffer(weight_nodes_.norm_mean).shape;
    if (tensor_numel(mean_shape) != input_dim_) {
        set_error(error, "cloud_handoff_feature_mean.weights shape mismatch");
        return false;
    }

    const auto& std_shape = graph_.get_output_buffer(weight_nodes_.norm_std).shape;
    if (tensor_numel(std_shape) != input_dim_) {
        set_error(error, "cloud_handoff_feature_std.weights shape mismatch");
        return false;
    }

    if (std::filesystem::exists(meta_file)) {
        const std::string meta_json = read_text_file(meta_file);
        float parsed = 0.0f;
        if (parse_json_float(meta_json, "threshold", parsed)) {
            threshold_ = parsed;
        }
    }

    if (!std::filesystem::exists(features_file)) {
        set_error(error, "Missing required classifier_features.json in: " + model_folder);
        return false;
    }

    const std::string json = read_text_file(features_file);
    size_t key_pos = json.find("\"features\"");
    if (key_pos == std::string::npos) {
        key_pos = json.find("\"feature_names\"");
    }
    size_t arr_begin = key_pos == std::string::npos ? std::string::npos : json.find('[', key_pos);
    if (arr_begin == std::string::npos) {
        set_error(error, "classifier_features.json missing feature_names array");
        return false;
    }

    int depth = 1;
    size_t arr_end = arr_begin + 1;
    while (arr_end < json.size() && depth > 0) {
        if (json[arr_end] == '[') {
            depth++;
        } else if (json[arr_end] == ']') {
            depth--;
        }
        arr_end++;
    }
    if (depth != 0 || arr_end <= arr_begin + 1) {
        set_error(error, "classifier_features.json has invalid feature_names array");
        return false;
    }

    const std::string arr_payload = json.substr(arr_begin + 1, arr_end - arr_begin - 2);
    const std::regex str_re("\"([^\"]+)\"");
    auto begin = std::sregex_iterator(arr_payload.begin(), arr_payload.end(), str_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        feature_names_.push_back((*it)[1].str());
    }

    if (feature_names_.size() != input_dim_) {
        set_error(
            error,
            "classifier_features.json feature count (" + std::to_string(feature_names_.size()) +
                ") does not match input_dim (" + std::to_string(input_dim_) + ")");
        return false;
    }

    build_graph();
    initialized_ = true;
    return true;
}

void WhisperCloudHandoffModel::load_weights_to_graph(CactusGraph* gb) {
    weight_nodes_.fc1_weight = gb->mmap_weights(model_folder_path_ + "/cloud_handoff_fc1.weights");
    weight_nodes_.fc1_bias = gb->mmap_weights(model_folder_path_ + "/cloud_handoff_fc1.bias");
    weight_nodes_.fc2_weight = gb->mmap_weights(model_folder_path_ + "/cloud_handoff_fc2.weights");
    weight_nodes_.fc2_bias = gb->mmap_weights(model_folder_path_ + "/cloud_handoff_fc2.bias");
    weight_nodes_.norm_mean = gb->mmap_weights(model_folder_path_ + "/cloud_handoff_feature_mean.weights");
    weight_nodes_.norm_std = gb->mmap_weights(model_folder_path_ + "/cloud_handoff_feature_std.weights");
}

void WhisperCloudHandoffModel::build_graph() {
    graph_nodes_.input = graph_.input({1, input_dim_}, Precision::FP16);

    size_t norm_mean = weight_nodes_.norm_mean;
    if (graph_.get_output_buffer(norm_mean).precision != Precision::FP16) {
        norm_mean = graph_.precision_cast(norm_mean, Precision::FP16);
    }

    size_t norm_std = weight_nodes_.norm_std;
    if (graph_.get_output_buffer(norm_std).precision != Precision::FP16) {
        norm_std = graph_.precision_cast(norm_std, Precision::FP16);
    }

    size_t normalized = graph_.subtract(graph_nodes_.input, norm_mean);
    norm_std = graph_.scalar_add(norm_std, 1e-8f);
    normalized = graph_.divide(normalized, norm_std);

    size_t hidden = graph_.matmul(normalized, weight_nodes_.fc1_weight, true, ComputeBackend::CPU);
    hidden = graph_.add(hidden, weight_nodes_.fc1_bias);

    hidden = graph_.relu(hidden);

    size_t logits = graph_.matmul(hidden, weight_nodes_.fc2_weight, true, ComputeBackend::CPU);
    logits = graph_.add(logits, weight_nodes_.fc2_bias);
    graph_nodes_.output = graph_.sigmoid(logits);
}

size_t WhisperCloudHandoffModel::forward(const std::vector<float>&, const std::vector<uint32_t>&, bool) {
    return 0;
}

float WhisperCloudHandoffModel::predict_probability(const std::vector<float>& features) const {
    if (!initialized_ || features.size() != input_dim_) {
        return 0.0f;
    }

    std::vector<__fp16> features_fp16(input_dim_);
    Quantization::fp32_to_fp16(features.data(), features_fp16.data(), input_dim_);
    graph_.set_input(graph_nodes_.input, features_fp16.data(), Precision::FP16);
    graph_.execute();

    const auto& out_buf = graph_.get_output_buffer(graph_nodes_.output);
    void* out_data = graph_.get_output(graph_nodes_.output);
    if (out_buf.total_size == 0 || out_data == nullptr) {
        return 0.0f;
    }

    float prob = 0.0f;
    if (out_buf.precision == Precision::FP32) {
        prob = static_cast<const float*>(out_data)[0];
    } else if (out_buf.precision == Precision::FP16) {
        prob = static_cast<float>(static_cast<const __fp16*>(out_data)[0]);
    } else {
        prob = static_cast<float>(static_cast<const int8_t*>(out_data)[0]);
    }

    return std::max(0.0f, std::min(1.0f, prob));
}

bool WhisperCloudHandoffModel::predict_handoff(const std::vector<float>& features, float* out_probability) const {
    const float prob = predict_probability(features);
    if (out_probability != nullptr) {
        *out_probability = prob;
    }
    return prob >= threshold_;
}

bool WhisperCloudHandoffModel::predict_handoff_from_audio(
    const std::vector<float>& waveform_features,
    const std::vector<float>& encoder_mean_features,
    bool* out_handoff,
    float* out_probability,
    std::string* error) const {

    if (out_handoff == nullptr) {
        set_error(error, "cloud_handoff classifier output pointer is null");
        return false;
    }
    *out_handoff = false;
    if (out_probability != nullptr) {
        *out_probability = 0.0f;
    }

    if (!initialized_) {
        set_error(error, "cloud_handoff classifier is not initialized");
        return false;
    }
    if (encoder_mean_features.empty()) {
        set_error(error, "encoder_mean_features are empty");
        return false;
    }

    float overlap_pitch_lag_cv = 0.0f;
    float overlap_spectral_peak_spacing_cv_mean = 0.0f;
    float overlap_yin_conf_p95 = 0.0f;
    float noise_hf_energy_ratio_mean = 0.0f;
    float noise_spectral_entropy_std = 0.0f;

    AudioProcessor feature_ap;
    overlap_pitch_lag_cv = feature_ap.overlap_pitch_lag_cv(
        waveform_features,
        kCloudHandoffSampleRate,
        kOverlapFrameLength,
        kOverlapHopLength);
    overlap_spectral_peak_spacing_cv_mean = feature_ap.overlap_spectral_peak_spacing_cv_mean(
        waveform_features,
        kCloudHandoffSampleRate,
        kOverlapFrameLength,
        kOverlapHopLength,
        kOverlapNfft);
    overlap_yin_conf_p95 = feature_ap.overlap_yin_conf_p95(
        waveform_features,
        kCloudHandoffSampleRate,
        kOverlapFrameLength,
        kOverlapHopLength);

    std::vector<float> stft_power;
    std::vector<float> stft_freqs_hz;
    size_t stft_num_frames = 0;
    compute_noise_stft_power(
        feature_ap,
        waveform_features,
        stft_power,
        stft_freqs_hz,
        stft_num_frames);
    if (stft_num_frames > 0) {
        noise_hf_energy_ratio_mean = feature_ap.high_freq_energy_ratio_mean(
            stft_power,
            stft_freqs_hz,
            stft_num_frames,
            high_freq_cutoff_hz_);
        noise_spectral_entropy_std = feature_ap.spectral_entropy_std(
            stft_power,
            stft_num_frames);
    }

    std::unordered_map<std::string, float> feature_values;
    feature_values.reserve(encoder_mean_features.size() + 8);
    for (size_t i = 0; i < encoder_mean_features.size(); i++) {
        feature_values["whisper_encoder_mean_" + std::to_string(i)] = encoder_mean_features[i];
    }
    feature_values["overlap::pitch_lag_cv"] = overlap_pitch_lag_cv;
    feature_values["overlap::spectral_peak_spacing_cv_mean"] = overlap_spectral_peak_spacing_cv_mean;
    feature_values["overlap::yin_conf_p95"] = overlap_yin_conf_p95;
    feature_values["noise::hf_energy_ratio_mean"] = noise_hf_energy_ratio_mean;
    feature_values["noise::spectral_entropy_std"] = noise_spectral_entropy_std;

    std::vector<float> classifier_input;
    classifier_input.reserve(feature_names_.size());
    for (const auto& feature_name : feature_names_) {
        const auto it = feature_values.find(feature_name);
        classifier_input.push_back(it != feature_values.end() ? it->second : 0.0f);
    }
    if (classifier_input.size() != input_dim_) {
        classifier_input.resize(input_dim_, 0.0f);
    }

    *out_handoff = predict_handoff(classifier_input, out_probability);
    return true;
}

} // namespace engine
} // namespace cactus
