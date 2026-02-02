#include "cactus_ffi.h"
#include "cactus_utils.h"
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <cctype>
#include <cmath>
#include <sstream>
#include <iomanip>

using namespace cactus::engine;
using namespace cactus::ffi;

static constexpr size_t RAG_TOP_K = 5;
static constexpr size_t RAG_CANDIDATE_K = 20;
static constexpr float RRF_K = 60.0f;
static constexpr float RRF_EMB_WEIGHT = 0.8f;
static constexpr float RRF_BM25_WEIGHT = 0.2f;
static constexpr float BM25_K1 = 1.5f;
static constexpr float BM25_B = 0.75f;

static std::vector<std::pair<float, size_t>> compute_rrf_scores(
    const std::vector<std::pair<float, size_t>>& emb_ranked,
    const std::vector<std::pair<float, size_t>>& bm25_ranked,
    size_t num_items
) {
    std::vector<std::pair<float, size_t>> emb_sorted = emb_ranked;
    std::vector<std::pair<float, size_t>> bm25_sorted = bm25_ranked;

    std::sort(emb_sorted.begin(), emb_sorted.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    std::sort(bm25_sorted.begin(), bm25_sorted.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::unordered_map<size_t, size_t> emb_rank_map, bm25_rank_map;
    for (size_t r = 0; r < emb_sorted.size(); ++r) {
        emb_rank_map[emb_sorted[r].second] = r + 1;
    }
    for (size_t r = 0; r < bm25_sorted.size(); ++r) {
        bm25_rank_map[bm25_sorted[r].second] = r + 1;
    }

    std::vector<std::pair<float, size_t>> rrf_scored;
    rrf_scored.reserve(num_items);
    for (size_t i = 0; i < num_items; ++i) {
        float rrf = RRF_EMB_WEIGHT / (RRF_K + emb_rank_map[i]) +
                    RRF_BM25_WEIGHT / (RRF_K + bm25_rank_map[i]);
        rrf_scored.emplace_back(rrf, i);
    }

    std::sort(rrf_scored.begin(), rrf_scored.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    return rrf_scored;
}

static std::vector<std::string> tokenize_words(const std::string& text) {
    std::vector<std::string> words;
    std::string current;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current += std::tolower(static_cast<unsigned char>(c));
        } else if (!current.empty()) {
            if (current.length() > 2) {
                words.push_back(current);
            }
            current.clear();
        }
    }
    if (!current.empty() && current.length() > 2) {
        words.push_back(current);
    }
    return words;
}

static float compute_bm25_score(
    const std::vector<std::string>& query_words,
    const std::string& doc_content,
    float avg_doc_len,
    const std::unordered_map<std::string, int>& doc_freqs,
    int total_docs
) {
    auto doc_words = tokenize_words(doc_content);
    float doc_len = static_cast<float>(doc_words.size());

    std::unordered_map<std::string, int> tf;
    for (const auto& w : doc_words) {
        tf[w]++;
    }

    float score = 0.0f;
    std::unordered_set<std::string> query_set(query_words.begin(), query_words.end());

    for (const auto& term : query_set) {
        auto tf_it = tf.find(term);
        if (tf_it == tf.end()) continue;

        float term_freq = static_cast<float>(tf_it->second);
        float df = 1.0f;
        auto df_it = doc_freqs.find(term);
        if (df_it != doc_freqs.end()) {
            df = static_cast<float>(df_it->second);
        }

        float idf = std::log((total_docs - df + 0.5f) / (df + 0.5f) + 1.0f);

        float tf_component = (term_freq * (BM25_K1 + 1.0f)) /
            (term_freq + BM25_K1 * (1.0f - BM25_B + BM25_B * (doc_len / avg_doc_len)));

        score += idf * tf_component;
    }

    return score;
}

std::string retrieve_rag_context(CactusModelHandle* handle, const std::string& query) {
    if (!handle->corpus_index || handle->corpus_embedding_dim == 0) {
        return "";
    }

    auto* tokenizer = handle->model->get_tokenizer();
    if (!tokenizer) return "";

    std::vector<uint32_t> query_tokens = tokenizer->encode(query);
    if (query_tokens.empty()) return "";

    std::vector<float> query_embedding = handle->model->get_embeddings(query_tokens, true, true);
    if (query_embedding.size() != handle->corpus_embedding_dim) {
        CACTUS_LOG_WARN("rag", "Query embedding dimension mismatch");
        return "";
    }

    index::QueryOptions options;
    options.top_k = RAG_CANDIDATE_K;
    options.score_threshold = 0.0f;

    std::vector<std::vector<float>> query_embeddings = {query_embedding};

    try {
        auto results = handle->corpus_index->query(query_embeddings, options);
        if (results.empty() || results[0].empty()) {
            return "";
        }

        std::vector<int> doc_ids;
        std::vector<float> embedding_scores;
        for (const auto& result : results[0]) {
            doc_ids.push_back(result.doc_id);
            embedding_scores.push_back(result.score);
        }

        auto docs = handle->corpus_index->get_documents(doc_ids);
        if (docs.empty()) {
            return "";
        }

        auto query_words = tokenize_words(query);

        float total_len = 0.0f;
        std::unordered_map<std::string, int> doc_freqs;
        for (const auto& doc : docs) {
            auto words = tokenize_words(doc.content);
            total_len += words.size();
            std::unordered_set<std::string> unique_words(words.begin(), words.end());
            for (const auto& w : unique_words) {
                doc_freqs[w]++;
            }
        }
        float avg_doc_len = total_len / docs.size();

        std::vector<std::pair<float, size_t>> emb_ranked;
        std::vector<std::pair<float, size_t>> bm25_ranked;
        for (size_t i = 0; i < docs.size(); ++i) {
            emb_ranked.emplace_back(embedding_scores[i], i);
            float bm25 = compute_bm25_score(
                query_words, docs[i].content, avg_doc_len, doc_freqs, docs.size()
            );
            bm25_ranked.emplace_back(bm25, i);
        }

        auto rrf_scored = compute_rrf_scores(emb_ranked, bm25_ranked, docs.size());

        std::string context = "[Retrieved Context - Use ONLY this information to answer. If the answer is not in the context, say \"I don't have enough information to answer that.\"]\n";
        size_t count = std::min(RAG_TOP_K, rrf_scored.size());
        for (size_t i = 0; i < count; ++i) {
            size_t idx = rrf_scored[i].second;
            context += "---\n";
            context += docs[idx].content;
            if (!docs[idx].metadata.empty()) {
                context += "\n(Source: " + docs[idx].metadata + ")";
            }
            context += "\n";
        }
        context += "---\n\n";

        CACTUS_LOG_DEBUG("rag", "Retrieved " << count << " RAG chunks (hybrid BM25+embedding)");
        return context;

    } catch (const std::exception& e) {
        CACTUS_LOG_WARN("rag", "RAG retrieval failed: " << e.what());
        return "";
    }
}

static std::string tool_to_text(const ToolFunction& tool) {
    return tool.name + " " + tool.description;
}

static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom > 0.0f ? dot / denom : 0.0f;
}

std::vector<cactus::ffi::ToolFunction> select_relevant_tools(
    CactusModelHandle* handle,
    const std::string& query,
    const std::vector<cactus::ffi::ToolFunction>& all_tools,
    size_t top_k
) {
    using cactus::ffi::ToolFunction;
    if (!handle || all_tools.empty() || top_k == 0) return all_tools;

    // If we have fewer tools than top_k, just return all
    if (all_tools.size() <= top_k) return all_tools;

    auto* tokenizer = handle->model->get_tokenizer();
    if (!tokenizer) {
        CACTUS_LOG_WARN("tool_rag", "No tokenizer available, returning all tools");
        return all_tools;
    }

    // Build tool texts and embeddings if not cached or tool set changed
    bool need_recompute = (handle->tool_texts.size() != all_tools.size());
    if (!need_recompute) {
        for (size_t i = 0; i < all_tools.size(); ++i) {
            if (tool_to_text(all_tools[i]) != handle->tool_texts[i]) {
                need_recompute = true;
                break;
            }
        }
    }

    if (need_recompute) {
        CACTUS_LOG_DEBUG("tool_rag", "Computing embeddings for " << all_tools.size() << " tools");
        handle->tool_texts.clear();
        handle->tool_embeddings.clear();
        handle->tool_texts.reserve(all_tools.size());
        handle->tool_embeddings.reserve(all_tools.size());

        for (const auto& tool : all_tools) {
            std::string text = tool_to_text(tool);
            handle->tool_texts.push_back(text);

            std::vector<uint32_t> tokens = tokenizer->encode(text);
            if (!tokens.empty()) {
                std::vector<float> emb = handle->model->get_embeddings(tokens, true, true);
                handle->tool_embeddings.push_back(std::move(emb));
            } else {
                handle->tool_embeddings.push_back({});
            }
        }
    }

    // Embed the query
    std::vector<uint32_t> query_tokens = tokenizer->encode(query);
    if (query_tokens.empty()) {
        CACTUS_LOG_WARN("tool_rag", "Empty query, returning all tools");
        return all_tools;
    }

    std::vector<float> query_embedding = handle->model->get_embeddings(query_tokens, true, true);
    if (query_embedding.empty()) {
        CACTUS_LOG_WARN("tool_rag", "Failed to get query embedding, returning all tools");
        return all_tools;
    }

    auto query_words = tokenize_words(query);

    float total_len = 0.0f;
    std::unordered_map<std::string, int> doc_freqs;
    for (const auto& text : handle->tool_texts) {
        auto words = tokenize_words(text);
        total_len += words.size();
        std::unordered_set<std::string> unique_words(words.begin(), words.end());
        for (const auto& w : unique_words) {
            doc_freqs[w]++;
        }
    }
    float avg_doc_len = total_len / handle->tool_texts.size();

    std::vector<std::pair<float, size_t>> emb_ranked;
    for (size_t i = 0; i < handle->tool_embeddings.size(); ++i) {
        float sim = cosine_similarity(query_embedding, handle->tool_embeddings[i]);
        emb_ranked.emplace_back(sim, i);
    }

    std::vector<std::pair<float, size_t>> bm25_ranked;
    for (size_t i = 0; i < handle->tool_texts.size(); ++i) {
        float bm25 = compute_bm25_score(
            query_words, handle->tool_texts[i], avg_doc_len, doc_freqs, handle->tool_texts.size()
        );
        bm25_ranked.emplace_back(bm25, i);
    }

    auto rrf_scored = compute_rrf_scores(emb_ranked, bm25_ranked, all_tools.size());

    // Select top-k tools
    std::vector<ToolFunction> selected;
    selected.reserve(top_k);
    for (size_t i = 0; i < top_k && i < rrf_scored.size(); ++i) {
        size_t idx = rrf_scored[i].second;
        selected.push_back(all_tools[idx]);
        CACTUS_LOG_DEBUG("tool_rag", "Selected tool: " << all_tools[idx].name << " (RRF: " << rrf_scored[i].first << ")");
    }

    CACTUS_LOG_INFO("tool_rag", "Selected " << selected.size() << " of " << all_tools.size() << " tools using hybrid ranking");
    return selected;
}

extern "C" {

int cactus_rag_query(
    cactus_model_t model,
    const char* query,
    char* response_buffer,
    size_t buffer_size,
    size_t top_k
) {
    if (!model || !query || !response_buffer || buffer_size == 0) {
        return -1;
    }

    auto* handle = static_cast<CactusModelHandle*>(model);

    if (!handle->corpus_index || handle->corpus_embedding_dim == 0) {
        std::strcpy(response_buffer, "{\"chunks\":[],\"error\":\"No corpus index loaded\"}");
        return 0;
    }

    try {
        auto* tokenizer = handle->model->get_tokenizer();
        if (!tokenizer) {
            std::strcpy(response_buffer, "{\"chunks\":[],\"error\":\"No tokenizer\"}");
            return 0;
        }

        std::vector<uint32_t> query_tokens = tokenizer->encode(query);
        if (query_tokens.empty()) {
            std::strcpy(response_buffer, "{\"chunks\":[],\"error\":\"Empty query\"}");
            return 0;
        }

        std::vector<float> query_embedding = handle->model->get_embeddings(query_tokens, true, true);
        if (query_embedding.size() != handle->corpus_embedding_dim) {
            std::strcpy(response_buffer, "{\"chunks\":[],\"error\":\"Embedding dimension mismatch\"}");
            return 0;
        }

        index::QueryOptions options;
        options.top_k = RAG_CANDIDATE_K;
        options.score_threshold = 0.0f;

        std::vector<std::vector<float>> query_embeddings = {query_embedding};
        auto results = handle->corpus_index->query(query_embeddings, options);

        if (results.empty() || results[0].empty()) {
            std::strcpy(response_buffer, "{\"chunks\":[]}");
            return 0;
        }

        std::vector<int> doc_ids;
        std::vector<float> embedding_scores;
        for (const auto& result : results[0]) {
            doc_ids.push_back(result.doc_id);
            embedding_scores.push_back(result.score);
        }

        auto docs = handle->corpus_index->get_documents(doc_ids);

        auto query_words = tokenize_words(query);

        float total_len = 0.0f;
        std::unordered_map<std::string, int> doc_freqs;
        for (const auto& doc : docs) {
            auto words = tokenize_words(doc.content);
            total_len += words.size();
            std::unordered_set<std::string> unique_words(words.begin(), words.end());
            for (const auto& w : unique_words) {
                doc_freqs[w]++;
            }
        }
        float avg_doc_len = docs.size() > 0 ? total_len / docs.size() : 1.0f;

        std::vector<std::pair<float, size_t>> emb_ranked;
        std::vector<std::pair<float, size_t>> bm25_ranked;
        for (size_t i = 0; i < docs.size(); ++i) {
            emb_ranked.emplace_back(embedding_scores[i], i);
            float bm25 = compute_bm25_score(
                query_words, docs[i].content, avg_doc_len, doc_freqs, docs.size()
            );
            bm25_ranked.emplace_back(bm25, i);
        }

        auto rrf_scored = compute_rrf_scores(emb_ranked, bm25_ranked, docs.size());

        size_t result_count = std::min(top_k > 0 ? top_k : RAG_TOP_K, rrf_scored.size());

        std::ostringstream oss;
        oss << "{\"chunks\":[";
        for (size_t i = 0; i < result_count; ++i) {
            size_t idx = rrf_scored[i].second;
            float final_score = rrf_scored[i].first;

            if (i > 0) oss << ",";
            oss << "{\"score\":" << std::setprecision(4) << final_score
                << ",\"source\":\"" << docs[idx].metadata << "\""
                << ",\"content\":\"";
            for (char c : docs[idx].content) {
                switch (c) {
                    case '"': oss << "\\\""; break;
                    case '\\': oss << "\\\\"; break;
                    case '\n': oss << "\\n"; break;
                    case '\r': oss << "\\r"; break;
                    case '\t': oss << "\\t"; break;
                    default: oss << c;
                }
            }
            oss << "\"}";
        }
        oss << "]}";

        std::string result = oss.str();
        if (result.size() >= buffer_size) {
            std::strcpy(response_buffer, "{\"chunks\":[],\"error\":\"Buffer too small\"}");
            return -1;
        }

        std::strcpy(response_buffer, result.c_str());
        return static_cast<int>(result.size());

    } catch (const std::exception& e) {
        std::string error = "{\"chunks\":[],\"error\":\"" + std::string(e.what()) + "\"}";
        if (error.size() < buffer_size) {
            std::strcpy(response_buffer, error.c_str());
        }
        return -1;
    }
}

}
