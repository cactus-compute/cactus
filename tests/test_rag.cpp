#include "test_utils.h"
#include <cstdlib>
#include <iostream>
#include <dirent.h>

using namespace EngineTestUtils;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");
static const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

static const char* g_options = R"({
        "max_tokens": 256,
    "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
    "telemetry_enabled": false
    })";

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

int main() {
    TestUtils::TestRunner runner("RAG Tests");
    runner.run_test("rag_preprocessing", test_rag());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
