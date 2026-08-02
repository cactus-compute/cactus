#include "test_utils.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace TestUtils;

namespace {

constexpr size_t LATENT_C = 4, LATENT_HW = 64, IMAGE_C = 3, IMAGE_HW = 512, CH = 64;

std::vector<__fp16> read_fp16_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t bytes = static_cast<size_t>(f.tellg());
    std::vector<__fp16> data(bytes / sizeof(__fp16));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

std::vector<float> read_fp32_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t bytes = static_cast<size_t>(f.tellg());
    std::vector<float> data(bytes / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

struct WeightStream {
    const std::vector<__fp16>& blob;
    size_t offset = 0;

    size_t next(CactusGraph& g, const std::vector<size_t>& shape) {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        size_t id = g.input(shape, Precision::FP16);
        g.set_input(id, blob.data() + offset, Precision::FP16);
        offset += n;
        return id;
    }
};

size_t taesd_conv(CactusGraph& g, WeightStream& w, size_t x, size_t c_in, size_t c_out, bool bias = true) {
    size_t weight = w.next(g, {c_out, c_in, 3, 3});
    if (!bias) return g.conv2d_k3s1p1(x, weight);
    size_t b = w.next(g, {c_out});
    return g.conv2d_k3s1p1(x, weight, b);
}

size_t taesd_block(CactusGraph& g, WeightStream& w, size_t x) {
    size_t h = g.relu(taesd_conv(g, w, x, CH, CH));
    h = g.relu(taesd_conv(g, w, h, CH, CH));
    h = taesd_conv(g, w, h, CH, CH);
    return g.relu(g.add(h, x));
}

size_t build_taesd_decoder(CactusGraph& g, WeightStream& w, size_t latent) {
    size_t x = g.scalar_multiply(g.tanh(g.scalar_divide(latent, 3.0f)), 3.0f);
    x = g.relu(taesd_conv(g, w, x, LATENT_C, CH));
    for (int stage = 0; stage < 3; ++stage) {
        x = taesd_block(g, w, x);
        x = taesd_block(g, w, x);
        x = taesd_block(g, w, x);
        x = g.upsample_nearest2d(x, 2);
        x = taesd_conv(g, w, x, CH, CH, false);
    }
    x = taesd_block(g, w, x);
    x = taesd_conv(g, w, x, CH, IMAGE_C);
    return g.clamp(x, 0.0f, 1.0f);
}

void write_ppm(const std::string& path, const __fp16* chw) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << IMAGE_HW << " " << IMAGE_HW << "\n255\n";
    const size_t plane = IMAGE_HW * IMAGE_HW;
    for (size_t i = 0; i < plane; ++i) {
        for (size_t c = 0; c < IMAGE_C; ++c) {
            float v = static_cast<float>(chw[c * plane + i]) * 255.0f;
            f.put(static_cast<char>(v < 0.0f ? 0 : (v > 255.0f ? 255 : static_cast<int>(v + 0.5f))));
        }
    }
}

float psnr_vs_reference(const __fp16* out, const std::vector<float>& ref) {
    double se = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = static_cast<double>(out[i]) - static_cast<double>(ref[i]);
        se += d * d;
    }
    double mse = se / ref.size();
    return mse > 0.0 ? static_cast<float>(10.0 * std::log10(1.0 / mse)) : 99.0f;
}

bool decode_and_check(const std::string& dir, const char* backend, const std::vector<__fp16>& weights,
                      const std::vector<__fp16>& latent, const std::vector<float>& ref) {
    if (cactus_backend_select(backend) != 0) {
        std::cout << "  [" << backend << "] unavailable, skipping" << std::endl;
        return true;
    }

    CactusGraph g;
    size_t latent_id = g.input({1, LATENT_C, LATENT_HW, LATENT_HW}, Precision::FP16);
    WeightStream w{weights};
    size_t out_id = build_taesd_decoder(g, w, latent_id);
    if (w.offset != weights.size()) {
        std::cout << "  weight blob mismatch: consumed " << w.offset << " of " << weights.size() << std::endl;
        return false;
    }
    g.set_input(latent_id, latent.data(), Precision::FP16);

    g.execute();
    constexpr int runs = 5;
    Timer timer;
    for (int i = 0; i < runs; ++i) g.execute();
    double ms = timer.elapsed_ms() / runs;

    const __fp16* out = static_cast<const __fp16*>(g.get_output(out_id));
    float psnr = psnr_vs_reference(out, ref);
    write_ppm(dir + "/cactus_decode_" + backend + ".ppm", out);
    std::cout << "  [" << backend << "] " << ms << " ms/decode, PSNR vs torch fp32: " << psnr << " dB" << std::endl;

    g.hard_reset();
    return psnr > 30.0f;
}

bool test_taesd_decoder() {
    const char* env = std::getenv("CACTUS_TAESD_DIR");
    if (!env) {
        std::cout << "  skipped: set CACTUS_TAESD_DIR to a dir with decoder.f16.bin/latent.f16.bin/ref_image.f32.bin"
                  << std::endl;
        return true;
    }
    std::string dir(env);

    auto weights = read_fp16_file(dir + "/decoder.f16.bin");
    auto latent = read_fp16_file(dir + "/latent.f16.bin");
    auto ref = read_fp32_file(dir + "/ref_image.f32.bin");
    if (weights.empty() || latent.size() != LATENT_C * LATENT_HW * LATENT_HW ||
        ref.size() != IMAGE_C * IMAGE_HW * IMAGE_HW) {
        std::cout << "  missing or malformed TAESD artifacts in " << dir << std::endl;
        return false;
    }

    bool ok = decode_and_check(dir, "cpu", weights, latent, ref);
    ok = decode_and_check(dir, "metal", weights, latent, ref) && ok;
    return ok;
}

}

int main() {
    TestRunner runner("VAE Decoder Tests");
    runner.run_test("taesd_decoder", test_taesd_decoder());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
