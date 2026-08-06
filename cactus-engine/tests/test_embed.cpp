#include "test_utils.h"
#include <cstdlib>
#include <iostream>

using namespace EngineTestUtils;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");
static const char* g_transcription_model_path = std::getenv("CACTUS_TEST_TRANSCRIPTION_MODEL");
static const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

bool test_embeddings() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║          EMBEDDINGS TEST                 ║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) return false;

    const char* texts[] = {"My name is Henry Ndubuaku", "Your name is Henry Ndubuaku"};
    std::vector<float> emb1(2048), emb2(2048);
    size_t dim1 = 0, dim2 = 0;

    int rc1 = cactus_embed(model, texts[0], emb1.data(), emb1.size() * sizeof(float), &dim1, true);
    int rc2 = cactus_embed(model, texts[1], emb2.data(), emb2.size() * sizeof(float), &dim2, true);

    if (rc1 < 0 || rc2 < 0 || dim1 == 0 || dim1 != dim2 || dim1 > emb1.size()) {
        cactus_destroy(model);
        return false;
    }

    float similarity = 0;
    for (size_t i = 0; i < dim1; ++i) {
        similarity += emb1[i] * emb2[i];
    }

    std::cout << "\n[Results]\n"
              << "├─ Embedding dim: " << dim1 << "\n"
              << "└─ Similarity: " << std::fixed << std::setprecision(4) << similarity << std::endl;

    cactus_destroy(model);
    return similarity > 0.5f;
}

static bool test_image_embeddings() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║         IMAGE EMBEDDING TEST             ║\n"
              << "╚══════════════════════════════════════════╝\n";

    std::string image_path = std::string(g_assets_path) + "/test_monkey.png";
    const size_t buffer_size = 1024 * 1024 * 4;
    std::vector<float> embeddings(buffer_size / sizeof(float));
    size_t embedding_dim = 0;

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) return false;

    int result = cactus_image_embed(model, image_path.c_str(), embeddings.data(), buffer_size, &embedding_dim);

    cactus_destroy(model);

    std::cout << "└─ Embedding dim: " << embedding_dim << std::endl;

    return result > 0 && embedding_dim > 0;
}

static bool test_audio_embeddings() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║         AUDIO EMBEDDING TEST             ║\n"
              << "╚══════════════════════════════════════════╝\n";

    const size_t buffer_size = 1024 * 1024;
    std::vector<float> embeddings(buffer_size / sizeof(float));
    size_t embedding_dim = 0;

    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) return false;

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    int result = cactus_audio_embed(model, audio_path.c_str(), embeddings.data(), buffer_size, &embedding_dim);

    cactus_destroy(model);

    std::cout << "└─ Embedding dim: " << embedding_dim << std::endl;

    return result > 0 && embedding_dim > 0;
}

int main() {
    TestUtils::TestRunner runner("Embedding Tests");
    runner.run_test("embeddings", test_embeddings());
    runner.run_test("image_embeddings", test_image_embeddings());
    runner.run_test("audio_embeddings", test_audio_embeddings());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
