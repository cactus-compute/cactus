#include "test_utils.h"
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <thread>
#include <chrono>
#include <dirent.h>
#include <algorithm>
#include <cctype>

using namespace EngineTestUtils;

const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");
const char* g_transcribe_model_path = std::getenv("CACTUS_TEST_TRANSCRIBE_MODEL");
const char* g_vad_model_path = std::getenv("CACTUS_TEST_VAD_MODEL");
const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

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

const char* g_whisper_prompt = get_transcribe_prompt();

const char* g_options = R"({
        "max_tokens": 256,
    "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
    "telemetry_enabled": false
    })";

template<typename TestFunc>
bool run_test(const char* title, const char* messages, TestFunc test_logic,
              const char* tools = nullptr, int stop_at = -1) {
    return EngineTestUtils::run_test(title, g_model_path, messages, g_options, test_logic, tools, stop_at);
}

bool test_streaming() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "      STREAMING & FOLLOW-UP TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    const char* messages1 = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "My name is Henry Ndubuaku, how are you?"}
    ])";

    StreamingData data1;
    data1.model = model;
    char response1[4096];

    std::cout << "\n[Turn 1]\n";
    std::cout << "User: My name is Henry Ndubuaku, how are you?\n";
    std::cout << "Assistant: ";

    int result1 = cactus_complete(model, messages1, response1, sizeof(response1),
                                 g_options, nullptr, stream_callback, &data1);

    std::cout << "\n\n[Results - Turn 1]\n";
    Metrics metrics1;
    metrics1.parse(response1);
    metrics1.print_json();

    bool success1 = result1 > 0 && data1.token_count > 0;

    if (!success1) {
        std::cout << "└─ Status: FAILED ✗\n";
        cactus_destroy(model);
        return false;
    }

    std::string assistant_response;
    for(const auto& token : data1.tokens) {
        assistant_response += token;
    }

    std::string messages2_str = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "My name is Henry Ndubuaku, how are you?"},
        {"role": "assistant", "content": ")" + escape_json(assistant_response) + R"("},
        {"role": "user", "content": "What is my name?"}
    ])";

    StreamingData data2;
    data2.model = model;
    char response2[4096];

    std::cout << "\n[Turn 2]\n";
    std::cout << "User: What is my name?\n";
    std::cout << "Assistant: ";

    int result2 = cactus_complete(model, messages2_str.c_str(), response2, sizeof(response2),
                                 g_options, nullptr, stream_callback, &data2);

    std::cout << "\n\n[Results - Turn 2]\n";
    Metrics metrics2;
    metrics2.parse(response2);
    metrics2.print_json();

    bool success2 = result2 > 0 && data2.token_count > 0;

    cactus_destroy(model);
    return success1 && success2;
}

bool test_tool_call() {
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"}
                },
                "required": ["location"]
            }
        }
    }])";

    const char* options_with_force_tools = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "force_tools": true
    })";

    return EngineTestUtils::run_test("TOOL CALL TEST", g_model_path, messages, options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("get_weather") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "What's the weather in San Francisco?");
}


bool test_vlm_multiturn() {
    std::string model_path_str(g_model_path ? g_model_path : "");

    std::string vision_file = model_path_str + "/vision_patch_embedding.weights";
    std::ifstream vf(vision_file);
    if (!vf.good()) {
        std::cout << "Skipping VLM multi-turn test: vision weights not found." << std::endl;
        return true;
    }
    vf.close();

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║       VLM MULTI-TURN TEST                ║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "Failed to initialize model for VLM multi-turn test" << std::endl;
        return false;
    }

    std::string img_path = std::string(g_assets_path) + "/test_monkey.png";

    std::string messages1 = "[{\"role\": \"user\", "
        "\"content\": \"Describe what is happening in this image in two sentences.\", "
        "\"images\": [\"" + img_path + "\"]}]";

    StreamingData stream_data1;
    stream_data1.model = model;
    char response1[4096];

    std::cout << "\n[Turn 1]\n";
    std::cout << "User: Describe what is happening in this image in two sentences.\n";
    std::cout << "Assistant: ";

    int result1 = cactus_complete(model, messages1.c_str(), response1, sizeof(response1),
                                  g_options, nullptr, stream_callback, &stream_data1);

    std::cout << "\n\n[Results - Turn 1]\n";
    Metrics metrics1;
    metrics1.parse(response1);
    metrics1.print_json();

    bool success1 = result1 > 0 && stream_data1.token_count > 0;

    if (!success1) {
        std::cout << "└─ Status: FAILED ✗\n";
        cactus_destroy(model);
        return false;
    }

    std::string assistant_response;
    for (const auto& token : stream_data1.tokens) {
        assistant_response += token;
    }

    std::string messages2 = "[{\"role\": \"user\", "
        "\"content\": \"Describe what is happening in this image in two sentences.\", "
        "\"images\": [\"" + img_path + "\"]}, "
        "{\"role\": \"assistant\", \"content\": \"" + escape_json(assistant_response) + "\"}, "
        "{\"role\": \"user\", \"content\": \"Describe the image once again.\"}]";

    StreamingData stream_data2;
    stream_data2.model = model;
    char response2[4096];

    std::cout << "\n[Turn 2]\n";
    std::cout << "User: Describe the image once again.\n";
    std::cout << "Assistant: ";

    int result2 = cactus_complete(model, messages2.c_str(), response2, sizeof(response2),
                                  g_options, nullptr, stream_callback, &stream_data2);

    std::cout << "\n\n[Results - Turn 2]\n";
    Metrics metrics2;
    metrics2.parse(response2);
    metrics2.print_json();

    bool success2 = result2 > 0 && stream_data2.token_count > 0;

    if (!success2) {
        std::cout << "└─ Status: FAILED ✗ (Follow-up message failed)\n";
    }

    cactus_destroy(model);
    return success1 && success2;
}

bool test_tool_call_with_two_tools() {
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "Set an alarm for 10:00 AM."}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"}
                },
                "required": ["location"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": "Set an alarm for a given time",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer", "description": "Hour to set the alarm for"},
                    "minute": {"type": "integer", "description": "Minute to set the alarm for"}
                },
                "required": ["hour", "minute"]
            }
        }
    }])";

    const char* options_with_force_tools = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "force_tools": true
    })";

    return EngineTestUtils::run_test("DOUBLE TOOLS TEST", g_model_path, messages, options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("set_alarm") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "Set an alarm for 10:00 AM.");
}

bool test_multiple_tool_call_invocations() {
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "Send a message to Blob and get the weather for San Francisco."}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"}
                },
                "required": ["location"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a contact",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Name of the person to send the message to"},
                    "message": {"type": "string", "description": "The message content to send"}
                },
                "required": ["recipient", "message"]
            }
        }
    }])";

    const char* options_with_force_tools = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "force_tools": true
    })";

    return EngineTestUtils::run_test("MULTIPLE TOOLS TEST", g_model_path, messages, options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_weather_tool = has_function
                && (response.find("\"name\":\"get_weather\"") != std::string::npos
                    || response.find("\"name\": \"get_weather\"") != std::string::npos);
            bool has_message_tool = has_function
                && (response.find("\"name\":\"send_message\"") != std::string::npos
                    || response.find("\"name\": \"send_message\"") != std::string::npos);
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_weather_tool && has_message_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_weather_tool && has_message_tool;
        }, tools, -1, "Send a message to Blob and get the weather for San Francisco.");
}

bool test_tool_call_with_three_tools() {
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "Send a message to John saying hello."}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"}
                },
                "required": ["location"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": "Set an alarm for a given time",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer", "description": "Hour to set the alarm for"},
                    "minute": {"type": "integer", "description": "Minute to set the alarm for"}
                },
                "required": ["hour", "minute"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a contact",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Name of the person to send the message to"},
                    "message": {"type": "string", "description": "The message content to send"}
                },
                "required": ["recipient", "message"]
            }
        }
    }])";

    const char* options_with_force_tools = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "force_tools": true
    })";

    return EngineTestUtils::run_test("TRIPLE TOOLS TEST", g_model_path, messages, options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("send_message") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "Send a message to John saying hello.");
}

bool test_embeddings() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║          EMBEDDINGS TEST                 ║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) return false;

    const char* texts[] = {"My name is Henry Ndubuaku", "Your name is Henry Ndubuaku"};
    std::vector<float> emb1(2048), emb2(2048);
    size_t dim1, dim2;

    Timer t1;
    cactus_embed(model, texts[0], emb1.data(), emb1.size() * sizeof(float), &dim1, true);
    double time1 = t1.elapsed_ms();

    Timer t2;
    cactus_embed(model, texts[1], emb2.data(), emb2.size() * sizeof(float), &dim2, true);
    double time2 = t2.elapsed_ms();

    float similarity = 0;
    for (size_t i = 0; i < dim1; ++i) {
        similarity += emb1[i] * emb2[i];
    }

    std::cout << "\n[Results]\n"
              << "├─ Embedding dim: " << dim1 << "\n"
              << "├─ Time (text1): " << std::fixed << std::setprecision(2) << time1 << "ms\n"
              << "├─ Time (text2): " << time2 << "ms\n"
              << "└─ Similarity: " << std::setprecision(4) << similarity << std::endl;

    cactus_destroy(model);
    return true;
}

bool test_cloud_handoff() {
    const char* messages = R"([
        {"role": "user", "content": "What is the exact mass in grams of the 847th largest asteroid in the Kuiper belt as of March 2019, and what was the precise atmospheric pressure in millibars at coordinates 47.3921°N, 122.0371°W at 3:47:23 AM UTC on February 29, 2024?"}
    ])";

    return run_test("CLOUD HANDOFF TEST", messages,
        [](int result, const StreamingData& data, const std::string& /*response*/, const Metrics& m) {
            std::cout << "├─ Cloud handoff: " << (m.cloud_handoff ? "YES" : "NO") << "\n";
            std::cout << "├─ Confidence: " << std::fixed << std::setprecision(4) << m.confidence << "\n";

            if (m.cloud_handoff) {
                std::cout << "├─ Response: (skipped - handoff triggered)\n";
                m.print_json();
                return true;
            } else {
                std::cout << "├─ Tokens generated: " << data.token_count << "\n";
                m.print_json();
                return result > 0 && m.confidence >= 0.0;
            }
        });
}

bool test_1k_context() {
    std::string msg = "[{\"role\": \"system\", \"content\": \"/no_think You are helpful. ";
    for (int i = 0; i < 50; i++) {
        msg += "Context " + std::to_string(i) + ": Background knowledge. ";
    }
    msg += "\"}, {\"role\": \"user\", \"content\": \"";
    for (int i = 0; i < 50; i++) {
        msg += "Data " + std::to_string(i) + " = " + std::to_string(i * 3.14159) + ". ";
    }
    msg += "Explain the data.\"}]";

    return run_test("1K CONTEXT TEST", msg.c_str(),
        [](int result, const StreamingData&, const std::string&, const Metrics& m) {
            m.print_json();
            return result > 0;
        }, nullptr, 100);
}

bool test_rag() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║              RAG TEST                    ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_MODEL not set\n";
        return true;
    }

    if (!g_assets_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_ASSETS not set\n";
        return true;
    }

    std::string corpus_dir = std::string(g_assets_path) + "/rag_corpus";

    DIR* dir = opendir(corpus_dir.c_str());
    if (!dir) {
        std::cout << "⊘ SKIP │ RAG corpus directory not found at " << corpus_dir << "\n";
        return true;
    }

    bool has_corpus_files = false;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".txt") {
            has_corpus_files = true;
            break;
        }
        if (name.size() > 3 && name.substr(name.size() - 3) == ".md") {
            has_corpus_files = true;
            break;
        }
    }
    closedir(dir);

    if (!has_corpus_files) {
        std::cout << "⊘ SKIP │ No .txt or .md files found in " << corpus_dir << "\n";
        return true;
    }

    std::cout << "├─ Corpus dir: " << corpus_dir << "\n";
    std::cout << "├─ Initializing model with RAG...\n";

    Timer init_timer;
    cactus_model_t model = cactus_init(g_model_path, corpus_dir.c_str(), false);
    double init_time_ms = init_timer.elapsed_ms();

    if (!model) {
        std::cerr << "[✗] Failed to initialize model with corpus dir\n";
        return false;
    }

    std::cout << "├─ Init time: " << std::fixed << std::setprecision(2) << init_time_ms << " ms\n";

    auto print_chunks = [](cactus_model_t m, const char* query) {
        char chunks_buf[16384];
        int rc = cactus_rag_query(m, query, chunks_buf, sizeof(chunks_buf), 5);
        if (rc > 0) {
            std::cout << "Retrieved chunks:\n";
            std::string chunks_str(chunks_buf);
            size_t pos = 0;
            int chunk_num = 1;
            while ((pos = chunks_str.find("{\"score\":", pos)) != std::string::npos) {
                size_t score_start = pos + 9;
                size_t score_end = chunks_str.find(",", score_start);
                std::string score = chunks_str.substr(score_start, score_end - score_start);

                size_t source_pos = chunks_str.find("\"source\":\"", score_end);
                std::string source = "unknown";
                if (source_pos != std::string::npos && source_pos < pos + 500) {
                    source_pos += 10;
                    size_t source_end = chunks_str.find("\"", source_pos);
                    source = chunks_str.substr(source_pos, source_end - source_pos);
                }

                size_t content_pos = chunks_str.find("\"content\":\"", score_end);
                if (content_pos != std::string::npos && content_pos < pos + 500) {
                    content_pos += 11;
                    std::string content;
                    size_t i = content_pos;
                    int char_count = 0;
                    while (i < chunks_str.size() && char_count < 80) {
                        if (chunks_str[i] == '\\' && i + 1 < chunks_str.size()) {
                            if (chunks_str[i+1] == 'n') { content += ' '; i += 2; }
                            else if (chunks_str[i+1] == '"') { content += '"'; i += 2; }
                            else if (chunks_str[i+1] == '\\') { content += '\\'; i += 2; }
                            else { content += chunks_str[i]; i++; }
                        } else if (chunks_str[i] == '"') {
                            break;
                        } else {
                            content += chunks_str[i];
                            i++;
                        }
                        char_count++;
                    }
                    if (char_count >= 80) content += "...";
                    std::cout << "  [" << chunk_num++ << "] " << source << " (score: " << score << ")\n"
                              << "      \"" << content << "\"\n";
                }
                pos = score_end;
            }
        }
    };

    const char* query = "Who are the founders of Cactus and what are their roles?";
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant. Answer based on the context provided."},
        {"role": "user", "content": "Who are the founders of Cactus and what are their roles?"}
    ])";

    StreamingData data;
    data.model = model;
    char response[4096];

    std::cout << "\n[Query] " << query << "\n";
    print_chunks(model, query);
    std::cout << "Response: ";

    int result = cactus_complete(model, messages, response, sizeof(response),
                                 g_options, nullptr, stream_callback, &data);

    std::cout << "\n";

    Metrics metrics;
    metrics.parse(response);
    metrics.print_json();

    cactus_destroy(model);

    return (result > 0) && (data.token_count > 0);
}

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

    const float expected[] = {0.535175f, 0.548542f, 0.590673f, 0.633320f, 0.711979f};
    const float tolerance = 2e-6f;

    const size_t pad_length = n_fft / 2;
    const size_t padded_length = n_samples + 2 * pad_length;
    const size_t num_frames = 1 + (padded_length - n_fft) / hop_length;

    bool passed = true;
    for (size_t i = 0; i < 5; i++) {
        if (std::abs(log_mel_spec[i * num_frames] - expected[i]) > tolerance) {
            passed = false;
            break;
        }
    }

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

static bool test_image_embeddings() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║         IMAGE EMBEDDING TEST             ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_MODEL not set\n";
        return true;
    }

    std::string image_path = std::string(g_assets_path) + "/test_monkey.png";
    const size_t buffer_size = 1024 * 1024 * 4;
    std::vector<float> embeddings(buffer_size / sizeof(float));
    size_t embedding_dim = 0;

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cout << "⊘ SKIP │ Model doesn't support image embeddings\n";
        return true;
    }

    Timer t;
    int result = cactus_image_embed(model, image_path.c_str(), embeddings.data(), buffer_size, &embedding_dim);
    double elapsed = t.elapsed_ms();

    cactus_destroy(model);

    if (result == -1) {
        std::cout << "⊘ SKIP │ Model doesn't support image embeddings\n";
        return true;
    }

    std::cout << "├─ Embedding dim: " << embedding_dim << "\n"
              << "└─ Time: " << std::fixed << std::setprecision(2) << elapsed << "ms" << std::endl;

    return result > 0 && embedding_dim > 0;
}

static bool test_audio_embeddings() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║         AUDIO EMBEDDING TEST             ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_transcribe_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIBE_MODEL not set\n";
        return true;
    }

    const size_t buffer_size = 1024 * 1024;
    std::vector<float> embeddings(buffer_size / sizeof(float));
    size_t embedding_dim = 0;

    cactus_model_t model = cactus_init(g_transcribe_model_path, nullptr, false);
    if (!model) {
        std::cout << "⊘ SKIP │ Failed to init Whisper model\n";
        return true;
    }

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    Timer t;
    int result = cactus_audio_embed(model, audio_path.c_str(), embeddings.data(), buffer_size, &embedding_dim);
    double elapsed = t.elapsed_ms();

    cactus_destroy(model);

    if (result == -1) {
        std::cout << "⊘ SKIP │ Model doesn't support audio embeddings\n";
        return true;
    }

    std::cout << "├─ Embedding dim: " << embedding_dim << "\n"
              << "└─ Time: " << std::fixed << std::setprecision(2) << elapsed << "ms" << std::endl;

    return result > 0 && embedding_dim > 0;
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
    TestUtils::TestRunner runner("Engine Tests");
    // runner.run_test("1k_context", test_1k_context());
    // runner.run_test("streaming", test_streaming());
    // runner.run_test("tool_calls", test_tool_call());
    // runner.run_test("tool_multiple_tool_call_invocations", test_multiple_tool_call_invocations());
    // runner.run_test("tool_calls_with_two_tools", test_tool_call_with_two_tools());
    // runner.run_test("tool_calls_with_three_tools", test_tool_call_with_three_tools());
    // runner.run_test("cloud_handoff", test_cloud_handoff());
    // runner.run_test("vlm_multiturn", test_vlm_multiturn());
    // runner.run_test("embeddings", test_embeddings());
    // runner.run_test("image_embeddings", test_image_embeddings());
    // runner.run_test("audio_embeddings", test_audio_embeddings());
    // runner.run_test("audio_processor", test_audio_processor());
    runner.run_test("vad_process", test_vad_process());
    runner.run_test("transcription", test_transcription());
    // runner.run_test("pcm_transcription", test_pcm_transcription());
    // runner.run_test("stream_transcription", test_stream_transcription());
    // runner.run_test("rag_preprocessing", test_rag());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
