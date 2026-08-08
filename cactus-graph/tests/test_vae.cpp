#include "test_utils.h"
#include "../../cactus-kernels/cactus_kernels.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace TestUtils;

namespace {

constexpr size_t IMAGE_HW = 512, LATENT_HW = 64, LATENT_C = 4, IMAGE_C = 3, CH = 64;

struct TaesdWeights {
    CactusGraph& graph;
    std::string dir;

    size_t weight(const std::string& name) { return graph.mmap_weights(dir + "/" + name + ".weights"); }
    size_t bias(const std::string& name) { return graph.mmap_weights(dir + "/" + name + ".bias"); }
};

size_t conv_biased(TaesdWeights& w, size_t x, const std::string& name) {
    return w.graph.conv2d_k3s1p1(x, w.weight(name), w.bias(name));
}

size_t block(TaesdWeights& w, size_t x, const std::string& prefix) {
    size_t h = w.graph.relu(conv_biased(w, x, prefix + "_conv_0"));
    h = w.graph.relu(conv_biased(w, h, prefix + "_conv_2"));
    h = conv_biased(w, h, prefix + "_conv_4");
    return w.graph.relu(w.graph.add(h, x));
}

size_t build_encoder(TaesdWeights& w, size_t image) {
    size_t x = conv_biased(w, image, "encoder_layers_0");
    x = block(w, x, "encoder_layers_1");
    for (int stage = 0; stage < 3; ++stage) {
        const int down = 2 + stage * 4;
        x = w.graph.conv2d_k3s2p1(x, w.weight("encoder_layers_" + std::to_string(down)));
        for (int j = 1; j <= 3; ++j) x = block(w, x, "encoder_layers_" + std::to_string(down + j));
    }
    return conv_biased(w, x, "encoder_layers_14");
}

size_t build_decoder(TaesdWeights& w, size_t latent) {
    size_t x = w.graph.scalar_multiply(w.graph.tanh(w.graph.scalar_divide(latent, 3.0f)), 3.0f);
    x = w.graph.relu(conv_biased(w, x, "decoder_layers_0"));
    for (int stage = 0; stage < 3; ++stage) {
        const int first = 2 + stage * 5;
        for (int j = 0; j < 3; ++j) x = block(w, x, "decoder_layers_" + std::to_string(first + j));
        x = w.graph.upsample_nearest2d(x, 2);
        x = w.graph.conv2d_k3s1p1(x, w.weight("decoder_layers_" + std::to_string(first + 4)));
    }
    x = block(w, x, "decoder_layers_17");
    return w.graph.clamp(conv_biased(w, x, "decoder_layers_18"), 0.0f, 1.0f);
}

std::vector<__fp16> load_image_chw(const std::string& path, std::vector<unsigned char>& rgb) {
    int w = 0, h = 0, c = 0;
    unsigned char* pixels = cactus_image_load(path.c_str(), &w, &h, &c, 3);
    if (!pixels) return {};
    rgb.assign(IMAGE_HW * IMAGE_HW * IMAGE_C, 0);
    if (w == (int)IMAGE_HW && h == (int)IMAGE_HW) {
        std::copy(pixels, pixels + rgb.size(), rgb.begin());
    } else {
        cactus_image_resize_uint8(pixels, w, h, rgb.data(), IMAGE_HW, IMAGE_HW, IMAGE_C);
    }
    cactus_image_free(pixels);

    std::vector<__fp16> chw(rgb.size());
    const size_t plane = IMAGE_HW * IMAGE_HW;
    for (size_t i = 0; i < plane; ++i)
        for (size_t ch = 0; ch < IMAGE_C; ++ch)
            chw[ch * plane + i] = static_cast<__fp16>(rgb[i * IMAGE_C + ch] / 255.0f);
    return chw;
}

void write_ppm(const std::string& path, const __fp16* chw) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << IMAGE_HW << " " << IMAGE_HW << "\n255\n";
    const size_t plane = IMAGE_HW * IMAGE_HW;
    for (size_t i = 0; i < plane; ++i)
        for (size_t c = 0; c < IMAGE_C; ++c) {
            float v = static_cast<float>(chw[c * plane + i]) * 255.0f;
            f.put(static_cast<char>(v < 0.0f ? 0 : (v > 255.0f ? 255 : static_cast<int>(v + 0.5f))));
        }
}

float psnr_vs_source(const __fp16* out, const std::vector<unsigned char>& rgb) {
    const size_t plane = IMAGE_HW * IMAGE_HW;
    double se = 0.0;
    for (size_t i = 0; i < plane; ++i)
        for (size_t c = 0; c < IMAGE_C; ++c) {
            double d = static_cast<double>(out[c * plane + i]) - rgb[i * IMAGE_C + c] / 255.0;
            se += d * d;
        }
    double mse = se / (plane * IMAGE_C);
    return mse > 0.0 ? static_cast<float>(10.0 * std::log10(1.0 / mse)) : 99.0f;
}

bool roundtrip(const std::string& dir, const std::string& image_path, const char* backend) {
    if (cactus_backend_select(backend) != 0) {
        std::cout << "  [" << backend << "] unavailable, skipping" << std::endl;
        return true;
    }

    std::vector<unsigned char> rgb;
    auto image = load_image_chw(image_path, rgb);
    if (image.empty()) {
        std::cout << "  could not read " << image_path << ": " << cactus_image_failure_reason() << std::endl;
        return false;
    }

    std::vector<__fp16> latent(LATENT_C * LATENT_HW * LATENT_HW);
    {
        CactusGraph g;
        TaesdWeights w{g, dir};
        size_t input = g.input({1, IMAGE_C, IMAGE_HW, IMAGE_HW}, Precision::FP16);
        size_t out = build_encoder(w, input);
        g.set_input(input, image.data(), Precision::FP16);
        g.execute();
        const __fp16* data = static_cast<const __fp16*>(g.get_output(out));
        std::copy(data, data + latent.size(), latent.begin());
        g.hard_reset();
    }

    CactusGraph g;
    TaesdWeights w{g, dir};
    size_t input = g.input({1, LATENT_C, LATENT_HW, LATENT_HW}, Precision::FP16);
    size_t out = build_decoder(w, input);
    g.set_input(input, latent.data(), Precision::FP16);

    g.execute();
    constexpr int runs = 5;
    Timer timer;
    for (int i = 0; i < runs; ++i) g.execute();
    double ms = timer.elapsed_ms() / runs;

    const __fp16* decoded = static_cast<const __fp16*>(g.get_output(out));
    float quality = psnr_vs_source(decoded, rgb);
    write_ppm(dir + "/taesd_roundtrip_" + backend + ".ppm", decoded);
    std::cout << "  [" << backend << "] decode " << ms << " ms, roundtrip PSNR " << quality << " dB" << std::endl;

    g.hard_reset();
    return quality > 20.0f;
}

bool test_taesd_roundtrip() {
    const char* bundle = std::getenv("CACTUS_TEST_VAE_MODEL");
    if (!bundle || !*bundle) {
        std::cout << "  skipped: no VAE weights. Run graph tests via 'cactus test', which prepares them."
                  << std::endl;
        return true;
    }
    const char* assets = std::getenv("CACTUS_TEST_ASSETS");
    std::string image_path = assets ? std::string(assets) + "/test_monkey.png"
                                    : "../../cactus-engine/tests/assets/test_monkey.png";

    bool ok = roundtrip(bundle, image_path, "cpu");
    ok = roundtrip(bundle, image_path, "metal") && ok;
    return ok;
}

}

int main() {
    TestRunner runner("VAE Tests");
    runner.run_test("taesd_encode_decode_roundtrip", test_taesd_roundtrip());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
