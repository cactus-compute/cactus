#include "bench_common.h"
#include "bench_driver.h"

#include <iostream>

int main(int argc, char** argv) {
    bench::MatmulBenchOptions opt;
    std::string err;
    if (!bench::parse_matmul_bench_args(argc, argv, opt, err)) {
        std::cerr << "Error: " << err << "\n"
                  << "Usage: " << argv[0]
                  << " [--iterations N] [--warmup N]\n"
                  << "       [--graphs gemv_d,gemm_d,gemm_mn]\n"
                  << "       [--dims 128,256,...]\n"
                  << "       [--backends fw1,fw2] [--threads N|max] [--csv path]\n";
        return 1;
    }

    const auto& backends = bench::get_matmul_backends();
    std::cout << "Registered matmul backends: " << backends.size() << "\n";
    for (const auto& b : backends)
        std::cout << "  " << b.name << " (" << b.framework << ")\n";

    if (!bench::run_matmul_benchmark(opt)) return 1;
    return 0;
}
