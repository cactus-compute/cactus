#include "test_utils.h"
#include <cstdlib>
#include <iostream>

using namespace EngineTestUtils;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");
static const char* g_transcribe_model_path = std::getenv("CACTUS_TEST_TRANSCRIBE_MODEL");
static const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

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

int main() {
    TestUtils::TestRunner runner("Embedding Tests");
    runner.run_test("embeddings", test_embeddings());
    runner.run_test("image_embeddings", test_image_embeddings());
    runner.run_test("audio_embeddings", test_audio_embeddings());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
