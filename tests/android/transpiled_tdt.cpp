#include "cactus_graph.h"
#include "picojson.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Binding {
    std::string component;
    size_t node_id;
    std::string path;
};

struct Config {
    int predictor_hidden_dim = 640;
    int predictor_num_layers = 2;
    int blank_id = 8192;
    int num_durations = 5;
    std::vector<int> durations{0, 1, 2, 3, 4};
};

struct Times {
    double encoder_ms = 0.0;
    double decoder_ms = 0.0;
    int decoder_steps = 0;
};

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (b.front() == '/') return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

std::vector<std::string> split_tab(const std::string& line) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t pos = line.find('\t', start);
        if (pos == std::string::npos) {
            out.push_back(line.substr(start));
            return out;
        }
        out.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
}

std::vector<Binding> load_bindings(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open bindings file: " + path);
    }
    std::vector<Binding> bindings;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto parts = split_tab(line);
        if (parts.size() != 3) {
            throw std::runtime_error("invalid bindings line: " + line);
        }
        bindings.push_back({parts[0], static_cast<size_t>(std::stoull(parts[1])), parts[2]});
    }
    return bindings;
}

size_t load_component_output_node_id(const std::string& bundle_root, const std::string& component_name, size_t output_index) {
    std::ifstream in(join_path(bundle_root, "components/manifest.json"));
    if (!in) {
        throw std::runtime_error("failed to open manifest file");
    }
    picojson::value root;
    std::string err = picojson::parse(root, in);
    if (!err.empty() || !root.is<picojson::object>()) {
        throw std::runtime_error("failed to parse manifest file: " + err);
    }
    const auto& obj = root.get<picojson::object>();
    const auto& components = obj.at("components").get<picojson::array>();
    for (const auto& value : components) {
        const auto& component = value.get<picojson::object>();
        if (component.at("component").get<std::string>() != component_name) continue;
        const auto& output_ids = component.at("output_node_ids").get<picojson::array>();
        if (output_index >= output_ids.size()) {
            throw std::runtime_error("manifest output index out of range for component: " + component_name);
        }
        return static_cast<size_t>(output_ids[output_index].get<double>());
    }
    throw std::runtime_error("manifest component not found: " + component_name);
}

Config load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open config file: " + path);
    }
    Config cfg;
    std::string line;
    while (std::getline(in, line)) {
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        if (key == "predictor_hidden_dim") cfg.predictor_hidden_dim = std::stoi(value);
        else if (key == "predictor_num_layers") cfg.predictor_num_layers = std::stoi(value);
        else if (key == "tdt_blank_id") cfg.blank_id = std::stoi(value);
        else if (key == "tdt_num_durations") cfg.num_durations = std::stoi(value);
        else if (key == "tdt_durations") {
            cfg.durations.clear();
            std::stringstream ss(value);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) cfg.durations.push_back(std::stoi(item));
            }
        }
    }
    if (cfg.durations.empty()) {
        for (int i = 0; i < cfg.num_durations; ++i) cfg.durations.push_back(i);
    }
    return cfg;
}

std::vector<std::string> load_vocab(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open vocab file: " + path);
    }
    std::vector<std::string> vocab;
    std::string line;
    while (std::getline(in, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        size_t id = static_cast<size_t>(std::stoull(line.substr(0, tab)));
        std::string token = line.substr(tab + 1);
        if (token == "<blank>") continue;
        if (id >= vocab.size()) vocab.resize(id + 1);
        vocab[id] = token;
    }
    return vocab;
}

void bind_component(CactusGraph& graph, const std::vector<Binding>& bindings, const std::string& component, const std::string& bundle_root) {
    for (const auto& binding : bindings) {
        if (binding.component != component) continue;
        graph.bind_mmap_weights(binding.node_id, join_path(bundle_root, binding.path));
    }
}

float read_scalar(const void* data, Precision precision, size_t index) {
    if (precision == Precision::FP32) {
        return static_cast<const float*>(data)[index];
    }
    if (precision == Precision::FP16) {
        return static_cast<float>(static_cast<const __fp16*>(data)[index]);
    }
    throw std::runtime_error("unsupported scalar precision");
}

void write_scalar(void* data, Precision precision, size_t index, float value) {
    if (precision == Precision::FP32) {
        static_cast<float*>(data)[index] = value;
        return;
    }
    if (precision == Precision::FP16) {
        static_cast<__fp16*>(data)[index] = static_cast<__fp16>(value);
        return;
    }
    throw std::runtime_error("unsupported scalar precision");
}

void copy_tensor_converting(void* dst, Precision dst_precision, const void* src, Precision src_precision, size_t count) {
    if (dst_precision == src_precision) {
        std::memcpy(dst, src, count * PrecisionTraits::size_of(dst_precision));
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        write_scalar(dst, dst_precision, i, read_scalar(src, src_precision, i));
    }
}

int argmax_logits(const void* logits, Precision precision, int count) {
    int best = 0;
    float best_value = read_scalar(logits, precision, 0);
    for (int i = 1; i < count; ++i) {
        float value = read_scalar(logits, precision, static_cast<size_t>(i));
        if (value > best_value) {
            best_value = value;
            best = i;
        }
    }
    return best;
}

int effective_blank_id(const Config& cfg, int token_class_count, size_t vocab_size) {
    if (vocab_size > 0 && cfg.blank_id == static_cast<int>(vocab_size - 1) && token_class_count == static_cast<int>(vocab_size + 1)) {
        return token_class_count - 1;
    }
    if (cfg.blank_id >= 0 && cfg.blank_id < token_class_count) {
        return cfg.blank_id;
    }
    if (vocab_size > 0 && token_class_count == static_cast<int>(vocab_size + 1)) {
        return token_class_count - 1;
    }
    return std::max(0, token_class_count - 1);
}

std::string decode_tokens(const std::vector<std::string>& vocab, const std::vector<int>& ids) {
    std::string out;
    for (int id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= vocab.size()) continue;
        const std::string& token = vocab[static_cast<size_t>(id)];
        if (token.empty()) continue;
        if (token.size() >= 2 && token.front() == '<' && token.back() == '>') continue;
        for (size_t i = 0; i < token.size();) {
            if (i + 3 <= token.size() &&
                static_cast<unsigned char>(token[i]) == 0xE2 &&
                static_cast<unsigned char>(token[i + 1]) == 0x96 &&
                static_cast<unsigned char>(token[i + 2]) == 0x81) {
                out.push_back(' ');
                i += 3;
            } else {
                out.push_back(token[i]);
                ++i;
            }
        }
    }
    size_t first = out.find_first_not_of(' ');
    if (first == std::string::npos) return "";
    size_t last = out.find_last_not_of(' ');
    return out.substr(first, last - first + 1);
}

std::vector<int> run_decode(
    CactusGraph& decoder,
    const Config& cfg,
    const std::vector<std::string>& vocab,
    const BufferDesc& hidden_buf,
    const void* hidden_ptr,
    int max_frames,
    Times& times
) {
    const auto& input_frame_buf = decoder.get_output_buffer(1);
    const auto& token_buf = decoder.get_output_buffer(2);
    const size_t frame_dim = input_frame_buf.total_size;
    std::vector<uint8_t> frame_storage(input_frame_buf.byte_size);
    std::vector<uint8_t> token_storage(token_buf.byte_size);
    std::vector<std::vector<uint8_t>> states;
    const std::vector<size_t> state_output_ids{26, 27, 31, 32};
    if (state_output_ids.size() != static_cast<size_t>(cfg.predictor_num_layers * 2)) {
        throw std::runtime_error("state output metadata mismatch");
    }
    for (int i = 0; i < cfg.predictor_num_layers * 2; ++i) {
        const auto& state_buf = decoder.get_output_buffer(static_cast<size_t>(3 + i));
        states.emplace_back(state_buf.byte_size, 0);
        decoder.set_external_input(static_cast<size_t>(3 + i), states.back().data(), state_buf.precision);
    }
    decoder.set_external_input(1, frame_storage.data(), input_frame_buf.precision);
    decoder.set_external_input(2, token_storage.data(), token_buf.precision);

    int duration_count = std::max(static_cast<int>(cfg.durations.size()), cfg.num_durations);
    int last_token = cfg.blank_id;
    std::vector<int> emitted;
    int time_index = 0;
    int total_frames = hidden_buf.shape.size() >= 2 ? static_cast<int>(hidden_buf.shape[1]) : 0;
    if (max_frames > 0) total_frames = std::min(total_frames, max_frames);
    if (hidden_buf.shape.size() < 3 || hidden_buf.shape[0] != 1) {
        throw std::runtime_error("encoder output must have shape [1,T,D]");
    }
    if (hidden_buf.shape[2] != frame_dim) {
        throw std::runtime_error("decoder frame input shape does not match encoder output");
    }

    auto start = std::chrono::steady_clock::now();
    while (time_index < total_frames) {
        const size_t row_offset = static_cast<size_t>(time_index) * frame_dim;
        const uint8_t* row = static_cast<const uint8_t*>(hidden_ptr) + row_offset * PrecisionTraits::size_of(hidden_buf.precision);
        copy_tensor_converting(frame_storage.data(), input_frame_buf.precision, row, hidden_buf.precision, frame_dim);
        bool advanced = false;
        int symbols_added = 0;
        while (symbols_added < 10) {
            write_scalar(token_storage.data(), token_buf.precision, 0, static_cast<float>(last_token));
            decoder.execute();
            ++times.decoder_steps;

            const auto& logits_buf = decoder.get_output_buffer(56);
            const void* logits_ptr = decoder.get_output(56);
            int total_classes = static_cast<int>(logits_buf.total_size);
            int active_duration_count = duration_count;
            if (active_duration_count <= 0 || active_duration_count >= total_classes) {
                active_duration_count = std::max(1, std::min(total_classes - 1, duration_count > 0 ? duration_count : 1));
            }
            int token_class_count = total_classes - active_duration_count;
            int blank_id = effective_blank_id(cfg, token_class_count, vocab.size());
            if (last_token < 0 || last_token >= token_class_count) last_token = blank_id;
            int next_token = argmax_logits(logits_ptr, logits_buf.precision, token_class_count);
            const uint8_t* duration_ptr = static_cast<const uint8_t*>(logits_ptr) + static_cast<size_t>(token_class_count) * PrecisionTraits::size_of(logits_buf.precision);
            int duration_index = argmax_logits(duration_ptr, logits_buf.precision, active_duration_count);
            int skip = cfg.durations.empty() ? 1 : cfg.durations[static_cast<size_t>(std::min(duration_index, static_cast<int>(cfg.durations.size()) - 1))];

            if (next_token != blank_id) {
                emitted.push_back(next_token);
                last_token = next_token;
                for (int i = 0; i < cfg.predictor_num_layers * 2; ++i) {
                    const size_t state_output_id = state_output_ids[static_cast<size_t>(i)];
                    const auto& out_buf = decoder.get_output_buffer(state_output_id);
                    std::memcpy(states[static_cast<size_t>(i)].data(), decoder.get_output(state_output_id), out_buf.byte_size);
                }
            }
            ++symbols_added;
            if (skip > 0) {
                time_index += skip;
                advanced = true;
                break;
            }
            if (next_token == blank_id) {
                ++time_index;
                advanced = true;
                break;
            }
        }
        if (!advanced) ++time_index;
    }
    auto end = std::chrono::steady_clock::now();
    times.decoder_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return emitted;
}

void print_json(const std::vector<int>& token_ids, const std::string& transcript, const BufferDesc& hidden_buf, const Times& times) {
    std::cout << "{";
    std::cout << "\"transcript\":\"";
    for (char ch : transcript) {
        if (ch == '"' || ch == '\\') std::cout << '\\';
        std::cout << ch;
    }
    std::cout << "\",\"token_ids\":[";
    for (size_t i = 0; i < token_ids.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << token_ids[i];
    }
    std::cout << "],\"encoder_ms\":" << std::fixed << std::setprecision(3) << times.encoder_ms;
    std::cout << ",\"decoder_ms\":" << std::fixed << std::setprecision(3) << times.decoder_ms;
    std::cout << ",\"total_ms\":" << std::fixed << std::setprecision(3) << (times.encoder_ms + times.decoder_ms);
    std::cout << ",\"decoder_steps\":" << times.decoder_steps;
    std::cout << ",\"encoder_hidden_shape\":[";
    for (size_t i = 0; i < hidden_buf.shape.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << hidden_buf.shape[i];
    }
    std::cout << "]}";
    std::cout << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <bundle_root> <input_features.weights> <bindings.tsv> [max_frames]\n";
        return 2;
    }
    try {
        std::string bundle_root = argv[1];
        std::string input_features_path = argv[2];
        std::string bindings_path = argv[3];
        int max_frames = argc >= 5 ? std::stoi(argv[4]) : 0;

        auto bindings = load_bindings(bindings_path);
        auto config = load_config(join_path(bundle_root, "config.txt"));
        auto vocab = load_vocab(join_path(bundle_root, "vocab.txt"));

        CactusGraph encoder = CactusGraph::load(join_path(bundle_root, "components/audio_encoder/graph.cactus"));
        CactusGraph decoder = CactusGraph::load(join_path(bundle_root, "components/decoder/graph.cactus"));
        bind_component(encoder, bindings, "audio_encoder", bundle_root);
        bind_component(decoder, bindings, "decoder", bundle_root);

        GraphFile::MappedFile input_features(input_features_path);
        encoder.set_external_input(1, input_features.data(), input_features.precision());

        Times times;
        auto encoder_start = std::chrono::steady_clock::now();
        encoder.execute();
        auto encoder_end = std::chrono::steady_clock::now();
        times.encoder_ms = std::chrono::duration<double, std::milli>(encoder_end - encoder_start).count();

        const size_t encoder_hidden_node_id = load_component_output_node_id(bundle_root, "audio_encoder", 0);
        const auto& hidden_buf = encoder.get_output_buffer(encoder_hidden_node_id);
        const void* hidden_ptr = encoder.get_output(encoder_hidden_node_id);
        auto emitted = run_decode(decoder, config, vocab, hidden_buf, hidden_ptr, max_frames, times);
        print_json(emitted, decode_tokens(vocab, emitted), hidden_buf, times);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << "\n";
        return 1;
    }
}
