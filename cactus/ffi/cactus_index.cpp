#include "cactus_ffi.h"
#include "cactus_utils.h"
#include <cstring>

using namespace cactus::ffi;

struct CactusIndexHandle {
    std::unique_ptr<cactus::engine::index::Index> index;
};

static cactus::engine::index::QueryOptions parse_query_options_json(const std::string& json) {
    cactus::engine::index::QueryOptions options;

    if (json.empty()) return options;

    size_t pos = json.find("\"top_k\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        options.top_k = std::stoul(json.substr(pos));
    }

    pos = json.find("\"score_threshold\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        options.score_threshold = std::stof(json.substr(pos));
    }

    return options;
}

extern "C" {

cactus_index_t cactus_index_init(const char* index_dir, size_t embedding_dim) {
    if (!index_dir) {
        last_error_message = "Index directory path cannot be null";
        CACTUS_LOG_ERROR("index_init", last_error_message);
        return nullptr;
    }

    if (embedding_dim == 0) {
        last_error_message = "Embedding dimension must be greater than 0";
        CACTUS_LOG_ERROR("index_init", last_error_message);
        return nullptr;
    }

    std::string dir_path(index_dir);
    if (dir_path.empty()) {
        last_error_message = "Index directory path cannot be empty";
        CACTUS_LOG_ERROR("index_init", last_error_message);
        return nullptr;
    }

    std::string index_path_str = dir_path + "/index.bin";
    std::string data_path_str = dir_path + "/data.bin";

    CACTUS_LOG_INFO("index_init", "Initializing index in directory: " << dir_path << ", dim: " << embedding_dim);

    try {
        auto* handle = new CactusIndexHandle();
        handle->index = std::make_unique<cactus::engine::index::Index>(
            index_path_str,
            data_path_str,
            embedding_dim
        );

        if (!handle->index) {
            last_error_message = "Failed to create index instance";
            CACTUS_LOG_ERROR("index_init", last_error_message);

            delete handle;
            return nullptr;
        }

        CACTUS_LOG_INFO("index_init", "Index initialized successfully");
        return handle;

    } catch (const std::exception& e) {
        last_error_message = "Exception during index init: " + std::string(e.what());
        CACTUS_LOG_ERROR("index_init", last_error_message);
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown exception during index initialization";
        CACTUS_LOG_ERROR("index_init", last_error_message);
        return nullptr;
    }
}

int cactus_index_add(
    cactus_index_t index,
    const int* ids,
    const char** documents,
    const char** metadatas,
    const float** embeddings,
    size_t count,
    size_t embedding_dim
) {
    if (!index) {
        last_error_message = "Index not initialized";
        CACTUS_LOG_ERROR("index_add", last_error_message);
        return -1;
    }

    if (!ids || !documents || !embeddings) {
        last_error_message = "Invalid parameters: ids, documents, or embeddings is null";
        CACTUS_LOG_ERROR("index_add", last_error_message);
        return -1;
    }

    if (count == 0) {
        last_error_message = "Invalid parameter: count must be greater than 0";
        CACTUS_LOG_ERROR("index_add", last_error_message);
        return -1;
    }

    if (embedding_dim == 0) {
        last_error_message = "Invalid parameter: embedding_dim must be greater than 0";
        CACTUS_LOG_ERROR("index_add", last_error_message);
        return -1;
    }

    try {
        auto* handle = static_cast<CactusIndexHandle*>(index);

        std::vector<cactus::engine::index::Document> docs;
        docs.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            if (!embeddings[i]) {
                last_error_message = "Embedding at index " + std::to_string(i) + " is null";
                CACTUS_LOG_ERROR("index_add", last_error_message);
                return -1;
            }

            docs.emplace_back(
                ids[i],
                std::vector<float>(embeddings[i], embeddings[i] + embedding_dim),
                documents[i] ? std::string(documents[i]) : "",
                metadatas && metadatas[i] ? std::string(metadatas[i]) : ""
            );
        }

        CACTUS_LOG_INFO("index_add", "Adding " << docs.size() << " documents");
        handle->index->add_documents(docs);

        return 0;

    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        CACTUS_LOG_ERROR("index_add", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during add";
        CACTUS_LOG_ERROR("index_add", last_error_message);
        return -1;
    }
}

int cactus_index_delete(
    cactus_index_t index,
    const int* ids,
    size_t ids_count
) {
    if (!index) {
        last_error_message = "Index not initialized";
        CACTUS_LOG_ERROR("index_delete", last_error_message);
        return -1;
    }

    if (!ids || ids_count == 0) {
        last_error_message = "Invalid parameters";
        CACTUS_LOG_ERROR("index_delete", last_error_message);
        return -1;
    }

    try {
        auto* handle = static_cast<CactusIndexHandle*>(index);

        std::vector<int> doc_ids(ids, ids + ids_count);

        CACTUS_LOG_INFO("index_delete", "Deleting " << doc_ids.size() << " documents");
        handle->index->delete_documents(doc_ids);

        return 0;

    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        CACTUS_LOG_ERROR("index_delete", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during delete";
        CACTUS_LOG_ERROR("index_delete", last_error_message);
        return -1;
    }
}

int cactus_index_get(
    cactus_index_t index,
    const int* ids,
    size_t ids_count,
    char** document_buffers,
    size_t* document_buffer_sizes,
    char** metadata_buffers,
    size_t* metadata_buffer_sizes,
    float** embedding_buffers,
    size_t* embedding_buffer_sizes
) {
    if (!index) {
        last_error_message = "Index not initialized";
        CACTUS_LOG_ERROR("index_get", last_error_message);
        return -1;
    }

    if (!ids) {
        last_error_message = "Invalid parameter: ids is null";
        CACTUS_LOG_ERROR("index_get", last_error_message);
        return -1;
    }

    if (ids_count == 0) {
        last_error_message = "Invalid parameter: ids_count must be greater than 0";
        CACTUS_LOG_ERROR("index_get", last_error_message);
        return -1;
    }

    if (document_buffers && !document_buffer_sizes) {
        last_error_message = "Invalid parameters: document_buffers provided but document_buffer_sizes is null";
        CACTUS_LOG_ERROR("index_get", last_error_message);
        return -1;
    }

    if (metadata_buffers && !metadata_buffer_sizes) {
        last_error_message = "Invalid parameters: metadata_buffers provided but metadata_buffer_sizes is null";
        CACTUS_LOG_ERROR("index_get", last_error_message);
        return -1;
    }

    if (embedding_buffers && !embedding_buffer_sizes) {
        last_error_message = "Invalid parameters: embedding_buffers provided but embedding_buffer_sizes is null";
        CACTUS_LOG_ERROR("index_get", last_error_message);
        return -1;
    }

    try {
        auto* handle = static_cast<CactusIndexHandle*>(index);

        std::vector<int> doc_ids(ids, ids + ids_count);

        CACTUS_LOG_INFO("index_get", "Getting " << doc_ids.size() << " documents");
        std::vector<cactus::engine::index::Document> docs = handle->index->get_documents(doc_ids);

        for (size_t i = 0; i < docs.size(); ++i) {
            if (document_buffers) {
                size_t document_size = docs[i].content.size() + 1;
                if (document_buffer_sizes[i] < document_size) {
                    last_error_message = "Document buffer too small at index " + std::to_string(i) +
                                        " (required: " + std::to_string(document_size) +
                                        ", provided: " + std::to_string(document_buffer_sizes[i]) + ")";
                    CACTUS_LOG_ERROR("index_get", last_error_message);
                    return -1;
                }
            }

            if (metadata_buffers) {
                size_t metadata_size = docs[i].metadata.size() + 1;
                if (metadata_buffer_sizes[i] < metadata_size) {
                    last_error_message = "Metadata buffer too small at index " + std::to_string(i) +
                                        " (required: " + std::to_string(metadata_size) +
                                        ", provided: " + std::to_string(metadata_buffer_sizes[i]) + ")";
                    CACTUS_LOG_ERROR("index_get", last_error_message);
                    return -1;
                }
            }

            if (embedding_buffers) {
                size_t embedding_size = docs[i].embedding.size();
                if (embedding_buffer_sizes[i] < embedding_size) {
                    last_error_message = "Embedding buffer too small at index " + std::to_string(i) +
                                        " (required: " + std::to_string(embedding_size) +
                                        ", provided: " + std::to_string(embedding_buffer_sizes[i]) + ")";
                    CACTUS_LOG_ERROR("index_get", last_error_message);
                    return -1;
                }
            }
        }

        for (size_t i = 0; i < docs.size(); ++i) {
            if (document_buffers) {
                size_t document_size = docs[i].content.size() + 1;
                std::memcpy(document_buffers[i], docs[i].content.c_str(), document_size);
                document_buffer_sizes[i] = document_size;
            }

            if (metadata_buffers) {
                size_t metadata_size = docs[i].metadata.size() + 1;
                std::memcpy(metadata_buffers[i], docs[i].metadata.c_str(), metadata_size);
                metadata_buffer_sizes[i] = metadata_size;
            }

            if (embedding_buffers) {
                size_t embedding_size = docs[i].embedding.size();
                std::memcpy(embedding_buffers[i], docs[i].embedding.data(), embedding_size * sizeof(float));
                embedding_buffer_sizes[i] = embedding_size;
            }
        }

        return 0;

    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        CACTUS_LOG_ERROR("index_get", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during get";
        CACTUS_LOG_ERROR("index_get", last_error_message);
        return -1;
    }
}

int cactus_index_query(
    cactus_index_t index,
    const float** embeddings,
    size_t embeddings_count,
    size_t embedding_dim,
    const char* options_json,
    int** id_buffers,
    size_t* id_buffer_sizes,
    float** score_buffers,
    size_t* score_buffer_sizes
) {
    if (!index) {
        last_error_message = "Index not initialized";
        CACTUS_LOG_ERROR("index_query", last_error_message);
        return -1;
    }

    if (!embeddings) {
        last_error_message = "Invalid parameter: embeddings is null";
        CACTUS_LOG_ERROR("index_query", last_error_message);
        return -1;
    }

    if (!id_buffers || !score_buffers) {
        last_error_message = "Invalid parameters: id_buffers or score_buffers is null";
        CACTUS_LOG_ERROR("index_query", last_error_message);
        return -1;
    }

    if (!id_buffer_sizes || !score_buffer_sizes) {
        last_error_message = "Invalid parameters: buffer size arrays cannot be null";
        CACTUS_LOG_ERROR("index_query", last_error_message);
        return -1;
    }

    if (embeddings_count == 0) {
        last_error_message = "Invalid parameter: embeddings_count must be greater than 0";
        CACTUS_LOG_ERROR("index_query", last_error_message);
        return -1;
    }

    if (embedding_dim == 0) {
        last_error_message = "Invalid parameter: embedding_dim must be greater than 0";
        CACTUS_LOG_ERROR("index_query", last_error_message);
        return -1;
    }

    try {
        auto* handle = static_cast<CactusIndexHandle*>(index);

        std::vector<std::vector<float>> embeddings_vec;
        embeddings_vec.reserve(embeddings_count);

        for (size_t i = 0; i < embeddings_count; ++i) {
            if (!embeddings[i]) {
                last_error_message = "Embedding at index " + std::to_string(i) + " is null";
                CACTUS_LOG_ERROR("index_query", last_error_message);
                return -1;
            }
            embeddings_vec.emplace_back(embeddings[i], embeddings[i] + embedding_dim);
        }

        cactus::engine::index::QueryOptions options;
        if (options_json && std::strlen(options_json) > 0) {
            options = parse_query_options_json(options_json);
        }

        CACTUS_LOG_DEBUG("index_query", "Querying with " << embeddings_count << " queries, dim: " << embedding_dim << ", top_k: " << options.top_k);
        std::vector<std::vector<cactus::engine::index::QueryResult>> results = handle->index->query(embeddings_vec, options);

        for (size_t i = 0; i < results.size(); ++i) {
            size_t result_count = results[i].size();
            if (id_buffer_sizes[i] < result_count || score_buffer_sizes[i] < result_count) {
                last_error_message = "Result buffer too small at query index " + std::to_string(i) +
                                    " (required: " + std::to_string(result_count) +
                                    ", id_buffer: " + std::to_string(id_buffer_sizes[i]) +
                                    ", score_buffer: " + std::to_string(score_buffer_sizes[i]) + ")";
                CACTUS_LOG_ERROR("index_query", last_error_message);
                return -1;
            }
        }

        for (size_t i = 0; i < results.size(); ++i) {
            size_t result_count = results[i].size();

            for (size_t j = 0; j < result_count; ++j) {
                id_buffers[i][j] = results[i][j].doc_id;
                score_buffers[i][j] = results[i][j].score;
            }

            id_buffer_sizes[i] = result_count;
            score_buffer_sizes[i] = result_count;
        }

        return 0;

    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        CACTUS_LOG_ERROR("index_query", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during query";
        CACTUS_LOG_ERROR("index_query", last_error_message);
        return -1;
    }
}

int cactus_index_compact(cactus_index_t index) {
    if (!index) {
        last_error_message = "Index not initialized";
        CACTUS_LOG_ERROR("index_compact", last_error_message);
        return -1;
    }

    try {
        auto* handle = static_cast<CactusIndexHandle*>(index);

        CACTUS_LOG_INFO("index_compact", "Compacting index");
        handle->index->compact();

        return 0;

    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        CACTUS_LOG_ERROR("index_compact", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during compact";
        CACTUS_LOG_ERROR("index_compact", last_error_message);
        return -1;
    }
}

void cactus_index_destroy(cactus_index_t index) {
    if (index) delete static_cast<CactusIndexHandle*>(index);
}

}
