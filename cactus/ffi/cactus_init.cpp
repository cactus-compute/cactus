#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "telemetry/telemetry.h"
#include <string>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>
#include <cstdlib>
#include <chrono>

using namespace cactus::engine;
using namespace cactus::ffi;

static constexpr size_t RAG_MAX_CHUNK_TOKENS = 128;
static constexpr size_t RAG_MIN_CHUNK_TOKENS = 24;
static constexpr size_t RAG_CHUNK_OVERLAP = 32;

static void apply_no_cloud_telemetry_env() {
    const char* env = std::getenv("CACTUS_NO_CLOUD_TELE");
    if (env && env[0] != '\0' && !(env[0] == '0' && env[1] == '\0')) {
        cactus::telemetry::setCloudDisabled(true);
    }
}

static time_t get_file_mtime(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return st.st_mtime;
    }
    return 0;
}

static bool corpus_is_stale(const std::string& corpus_dir) {
    std::string index_path = corpus_dir + "/index.bin";
    time_t index_mtime = get_file_mtime(index_path);
    if (index_mtime == 0) return true;

    DIR* dir = opendir(corpus_dir.c_str());
    if (!dir) return true;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        if (name == "index.bin" || name == "data.bin") continue;

        bool is_corpus_file = false;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".txt") is_corpus_file = true;
        if (name.size() > 3 && name.substr(name.size() - 3) == ".md") is_corpus_file = true;

        if (is_corpus_file) {
            std::string full_path = corpus_dir + "/" + name;
            time_t file_mtime = get_file_mtime(full_path);
            if (file_mtime > index_mtime) {
                closedir(dir);
                CACTUS_LOG_INFO("init", "Corpus file " << name << " is newer than index, rebuilding");
                return true;
            }
        }
    }
    closedir(dir);
    return false;
}

static std::string read_file_contents(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static std::vector<std::string> scan_corpus_files(const std::string& corpus_dir) {
    std::vector<std::string> files;
    DIR* dir = opendir(corpus_dir.c_str());
    if (!dir) return files;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        if (name == "index.bin" || name == "data.bin") continue;

        std::string full_path = corpus_dir + "/" + name;
        struct stat st;
        if (stat(full_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            if (name.size() > 4 && (name.substr(name.size() - 4) == ".txt" || name.substr(name.size() - 3) == ".md")) {
                files.push_back(full_path);
            }
        }
    }
    closedir(dir);
    return files;
}

static std::vector<std::string> split_into_paragraphs(const std::string& content) {
    std::vector<std::string> paragraphs;
    std::string current;

    size_t i = 0;
    while (i < content.size()) {
        // Check for markdown header
        if (content[i] == '#' && (i == 0 || content[i-1] == '\n')) {
            if (!current.empty()) {
                paragraphs.push_back(current);
                current.clear();
            }
            // Include the header line
            while (i < content.size() && content[i] != '\n') {
                current += content[i++];
            }
            if (i < content.size()) current += content[i++];
            continue;
        }

        // Check for double newline (paragraph break)
        if (content[i] == '\n' && i + 1 < content.size() && content[i+1] == '\n') {
            current += content[i];
            if (!current.empty() && current != "\n") {
                paragraphs.push_back(current);
                current.clear();
            }
            i++;
            // Skip multiple blank lines
            while (i < content.size() && content[i] == '\n') i++;
            continue;
        }

        current += content[i++];
    }

    if (!current.empty()) {
        paragraphs.push_back(current);
    }

    return paragraphs;
}

static std::vector<std::pair<std::string, std::string>> chunk_corpus(
    const std::vector<std::string>& file_paths,
    Tokenizer* tokenizer
) {
    std::vector<std::pair<std::string, std::string>> chunks;

    for (const auto& path : file_paths) {
        std::string content = read_file_contents(path);
        if (content.empty()) continue;

        std::string filename = path;
        size_t last_slash = path.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            filename = path.substr(last_slash + 1);
        }

        auto paragraphs = split_into_paragraphs(content);

        std::string current_chunk;
        size_t current_tokens = 0;

        for (const auto& para : paragraphs) {
            std::vector<uint32_t> para_tokens = tokenizer->encode(para);
            size_t para_token_count = para_tokens.size();

            // If paragraph alone is too large, split it by tokens
            if (para_token_count > RAG_MAX_CHUNK_TOKENS) {
                // Flush current chunk first
                if (!current_chunk.empty()) {
                    chunks.emplace_back(current_chunk, filename);
                    current_chunk.clear();
                    current_tokens = 0;
                }

                // Split large paragraph by sentences or fixed tokens
                size_t stride = RAG_MAX_CHUNK_TOKENS - RAG_CHUNK_OVERLAP;
                for (size_t i = 0; i < para_tokens.size(); i += stride) {
                    size_t end = std::min(i + RAG_MAX_CHUNK_TOKENS, para_tokens.size());
                    std::vector<uint32_t> chunk_tokens(para_tokens.begin() + i, para_tokens.begin() + end);
                    std::string chunk_text = tokenizer->decode(chunk_tokens);
                    chunks.emplace_back(chunk_text, filename);
                    if (end >= para_tokens.size()) break;
                }
                continue;
            }

            // Would adding this paragraph exceed max?
            if (current_tokens + para_token_count > RAG_MAX_CHUNK_TOKENS && !current_chunk.empty()) {
                chunks.emplace_back(current_chunk, filename);
                current_chunk.clear();
                current_tokens = 0;
            }

            // Add paragraph to current chunk
            if (!current_chunk.empty()) current_chunk += "\n";
            current_chunk += para;
            current_tokens += para_token_count;
        }

        // Don't forget the last chunk
        if (!current_chunk.empty() && current_tokens >= RAG_MIN_CHUNK_TOKENS) {
            chunks.emplace_back(current_chunk, filename);
        } else if (!current_chunk.empty() && !chunks.empty()) {
            // Append small remaining chunk to previous
            chunks.back().first += "\n" + current_chunk;
        } else if (!current_chunk.empty()) {
            chunks.emplace_back(current_chunk, filename);
        }
    }

    return chunks;
}

static bool build_corpus_index(CactusModelHandle* handle, const std::string& corpus_dir) {
    CACTUS_LOG_INFO("init", "Building corpus index from: " << corpus_dir);

    auto* tokenizer = handle->model->get_tokenizer();
    if (!tokenizer) {
        CACTUS_LOG_ERROR("init", "No tokenizer available for corpus indexing");
        return false;
    }

    auto file_paths = scan_corpus_files(corpus_dir);
    if (file_paths.empty()) {
        CACTUS_LOG_WARN("init", "No .txt or .md files found in corpus directory");
        return false;
    }

    CACTUS_LOG_INFO("init", "Found " << file_paths.size() << " corpus files");

    auto chunks = chunk_corpus(file_paths, tokenizer);
    if (chunks.empty()) {
        CACTUS_LOG_WARN("init", "No chunks generated from corpus");
        return false;
    }

    CACTUS_LOG_INFO("init", "Generated " << chunks.size() << " chunks from corpus");

    std::vector<uint32_t> test_tokens = tokenizer->encode("test");
    std::vector<float> test_embedding = handle->model->get_embeddings(test_tokens, true, true);
    if (test_embedding.empty()) {
        CACTUS_LOG_ERROR("init", "Failed to get embedding dimension");
        return false;
    }
    size_t embedding_dim = test_embedding.size();
    handle->corpus_embedding_dim = embedding_dim;

    CACTUS_LOG_INFO("init", "Embedding dimension: " << embedding_dim);

    std::string index_path = corpus_dir + "/index.bin";
    std::string data_path = corpus_dir + "/data.bin";

    try {
        handle->corpus_index = std::make_unique<index::Index>(index_path, data_path, embedding_dim);
    } catch (const std::exception& e) {
        CACTUS_LOG_ERROR("init", "Failed to create index: " << e.what());
        return false;
    }

    std::vector<index::Document> docs;
    docs.reserve(chunks.size());

    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& [chunk_text, source_file] = chunks[i];

        std::vector<uint32_t> tokens = tokenizer->encode(chunk_text);
        std::vector<float> embedding = handle->model->get_embeddings(tokens, true, true);

        if (embedding.size() != embedding_dim) {
            CACTUS_LOG_WARN("init", "Skipping chunk " << i << " - embedding dimension mismatch");
            continue;
        }

        docs.push_back(index::Document{
            static_cast<int>(i),
            std::move(embedding),
            chunk_text,
            source_file
        });

        if ((i + 1) % 50 == 0) {
            CACTUS_LOG_INFO("init", "Embedded " << (i + 1) << "/" << chunks.size() << " chunks");
        }
    }

    if (docs.empty()) {
        CACTUS_LOG_ERROR("init", "No documents to add to index");
        return false;
    }

    try {
        handle->corpus_index->add_documents(docs);
    } catch (const std::exception& e) {
        CACTUS_LOG_ERROR("init", "Failed to add documents to index: " << e.what());
        return false;
    }

    CACTUS_LOG_INFO("init", "Corpus index built successfully with " << docs.size() << " chunks");
    return true;
}

static bool load_corpus_index(CactusModelHandle* handle, const std::string& corpus_dir) {
    std::string index_path = corpus_dir + "/index.bin";
    std::string data_path = corpus_dir + "/data.bin";

    struct stat st;
    if (stat(index_path.c_str(), &st) != 0 || stat(data_path.c_str(), &st) != 0) {
        return false;
    }

    if (corpus_is_stale(corpus_dir)) {
        return false;
    }

    auto* tokenizer = handle->model->get_tokenizer();
    std::vector<uint32_t> test_tokens = tokenizer->encode("test");
    std::vector<float> test_embedding = handle->model->get_embeddings(test_tokens, true, true);
    if (test_embedding.empty()) {
        CACTUS_LOG_ERROR("init", "Failed to get embedding dimension for index loading");
        return false;
    }
    size_t embedding_dim = test_embedding.size();
    handle->corpus_embedding_dim = embedding_dim;

    try {
        handle->corpus_index = std::make_unique<index::Index>(index_path, data_path, embedding_dim);
        CACTUS_LOG_INFO("init", "Loaded existing corpus index from: " << corpus_dir);
        return true;
    } catch (const std::exception& e) {
        CACTUS_LOG_WARN("init", "Failed to load existing index: " << e.what());
        return false;
    }
}

std::string last_error_message;

bool matches_stop_sequence(const std::vector<uint32_t>& generated_tokens,
                           const std::vector<std::vector<uint32_t>>& stop_sequences) {
    for (const auto& stop_seq : stop_sequences) {
        if (stop_seq.empty()) continue;
        if (generated_tokens.size() >= stop_seq.size()) {
            if (std::equal(stop_seq.rbegin(), stop_seq.rend(), generated_tokens.rbegin()))
                return true;
        }
    }
    return false;
}

extern "C" {

const char* cactus_get_last_error() {
    return last_error_message.c_str();
}

cactus_model_t cactus_init(const char* model_path, const char* corpus_dir, bool cache_index) {
    constexpr size_t DEFAULT_CONTEXT_SIZE = 512;  // matches default sliding window size

    std::string model_path_str = model_path ? std::string(model_path) : "unknown";

    std::string model_name = model_path_str;
    size_t last_slash = model_path_str.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        model_name = model_path_str.substr(last_slash + 1);
    }

    CACTUS_LOG_INFO("init", "Loading model: " << model_name << " from " << model_path_str);

    apply_no_cloud_telemetry_env();
    cactus::telemetry::init(nullptr, model_path_str.c_str(), nullptr);

    auto __cactus_init_start = std::chrono::steady_clock::now();

    try {
        auto* handle = new CactusModelHandle();
        handle->model = create_model(model_path);
        handle->model_name = model_name;

        if (!handle->model) {
            last_error_message = "Failed to create model - check config.txt exists at: " + model_path_str;
            CACTUS_LOG_ERROR("init", last_error_message);
            {
                auto __cactus_init_err_dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - __cactus_init_start).count();
                cactus::telemetry::recordInit(model_name.c_str(), false, static_cast<double>(__cactus_init_err_dur), last_error_message.c_str());
            }
            delete handle;
            return nullptr;
        }

        if (!handle->model->init(model_path, DEFAULT_CONTEXT_SIZE)) {
            last_error_message = "Failed to initialize model - check weight files at: " + model_path_str;
            CACTUS_LOG_ERROR("init", last_error_message);
            {
                auto __cactus_init_err_dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - __cactus_init_start).count();
                cactus::telemetry::recordInit(model_name.c_str(), false, static_cast<double>(__cactus_init_err_dur), last_error_message.c_str());
            }
            delete handle;
            return nullptr;
        }

        auto model_type = handle->model->get_config().model_type;
        if (model_type == Config::ModelType::WHISPER || model_type == Config::ModelType::MOONSHINE) {
            std::string vad_path = model_path_str + "/vad";
            handle->vad_model = create_model(vad_path);
            if (!handle->vad_model) {
                last_error_message = "Failed to create VAD model - check VAD weights at: " + vad_path;
                CACTUS_LOG_ERROR("init", last_error_message);
                delete handle;
                return nullptr;
            }
            if (!handle->vad_model->init(vad_path, 0, "", false)) {
                last_error_message = "Failed to initialize VAD model - check VAD weight files at: " + vad_path;
                CACTUS_LOG_ERROR("init", last_error_message);
                delete handle;
                return nullptr;
            }
        }

        if (corpus_dir != nullptr && strlen(corpus_dir) > 0) {
            handle->corpus_dir = std::string(corpus_dir);

            bool loaded = false;
            if (cache_index) {
                loaded = load_corpus_index(handle, handle->corpus_dir);
            }

            if (!loaded) {
                CACTUS_LOG_INFO("init", (cache_index ? "No existing index found, building new corpus index" : "Building fresh corpus index (caching disabled)"));
                if (!build_corpus_index(handle, handle->corpus_dir)) {
                    CACTUS_LOG_WARN("init", "Failed to build corpus index - RAG disabled");
                }
            }
        }

        CACTUS_LOG_INFO("init", "Model loaded successfully: " << model_name);
        {
            auto __cactus_init_ok_dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - __cactus_init_start).count();
            cactus::telemetry::recordInit(model_name.c_str(), true, static_cast<double>(__cactus_init_ok_dur), "");
        }

        return handle;
    } catch (const std::exception& e) {
        last_error_message = "Exception during init: " + std::string(e.what());
        CACTUS_LOG_ERROR("init", last_error_message);
        {
            auto __cactus_init_err_dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - __cactus_init_start).count();
            cactus::telemetry::recordInit(model_name.c_str(), false, static_cast<double>(__cactus_init_err_dur), last_error_message.c_str());
        }
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown exception during model initialization";
        CACTUS_LOG_ERROR("init", last_error_message);
        {
            auto __cactus_init_err_dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - __cactus_init_start).count();
            cactus::telemetry::recordInit(model_name.c_str(), false, static_cast<double>(__cactus_init_err_dur), last_error_message.c_str());
        }
        return nullptr;
    }
}

void cactus_destroy(cactus_model_t model) {
    if (model) delete static_cast<CactusModelHandle*>(model);
}

void cactus_reset(cactus_model_t model) {
    if (!model) return;
    auto* handle = static_cast<CactusModelHandle*>(model);
    handle->model->reset_cache();
    handle->processed_tokens.clear();
}

void cactus_stop(cactus_model_t model) {
    if (!model) return;
    auto* handle = static_cast<CactusModelHandle*>(model);
    handle->should_stop = true;
}

}
