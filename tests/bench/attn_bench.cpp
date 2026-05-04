#include "bench_common.h"
#include "bench_driver.h"

#include <iostream>

int main(int argc, char** argv) {
    bench::AttnBenchOptions opt;
    std::string err;
    if (!bench::parse_attn_bench_args(argc, argv, opt, err)) {
        std::cerr << "Error: " << err << "\n"
                  << "Usage: " << argv[0]
                  << " [--iterations N] [--warmup N]\n"
                  << "       [--graphs attn_prefill_s,attn_decode_cache]\n"
                  << "       [--dims 128,256,...] [--model_dim N] [--heads N] [--kv_heads N] [--head_dim N]\n"
                  << "       [--backends fw1,fw2] [--threads N|max] [--csv path]\n";
        return 1;
    }

    const auto& backends = bench::get_attn_backends();
    std::cout << "Registered attention backends: " << backends.size() << "\n";
    for (const auto& b : backends) {
        std::cout << "  " << b.name << " (" << b.framework << ", "
                  << (b.mode == bench::AttnMode::PREFILL ? "prefill" : "decode") << ")\n";
    }

    if (!bench::run_attn_benchmark(opt)) return 1;
    return 0;
}
