#include "e2e_driver.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <numeric>
#include <unordered_map>

namespace e2e {

static std::vector<E2EBackend>& backend_registry() {
    static std::vector<E2EBackend> backends;
    return backends;
}

void register_e2e_backend(E2EBackend b) {
    backend_registry().push_back(b);
}

const std::vector<E2EBackend>& get_e2e_backends() {
    return backend_registry();
}

std::vector<ScheduledRun> build_schedule(int n_backends, int n_rounds) {
    std::vector<ScheduledRun> schedule;
    for (int round = 0; round < n_rounds; round++) {
        for (int slot = 0; slot < n_backends; slot++) {
            int backend = (slot + round) % n_backends;
            schedule.push_back({backend, round});
        }
    }
    return schedule;
}

void write_csv(const std::string& path, const std::vector<E2ERunRecord>& records) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open " << path << " for writing\n";
        return;
    }
    f << "backend,model,rep,prefill_tokens,decode_tokens,prefill_tps,decode_tps,ttft_ms,total_ms\n";
    f << std::fixed;
    for (const auto& r : records) {
        f << r.backend << ","
          << r.model << ","
          << r.rep << ","
          << r.result.prefill_tokens << ","
          << r.result.decode_tokens << ","
          << std::setprecision(1) << r.result.prefill_tps << ","
          << std::setprecision(1) << r.result.decode_tps << ","
          << std::setprecision(1) << r.result.ttft_ms << ","
          << std::setprecision(1) << r.result.total_ms << "\n";
    }
}

static double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = mean(v);
    double sq_sum = 0.0;
    for (double x : v) sq_sum += (x - m) * (x - m);
    return std::sqrt(sq_sum / (v.size() - 1));
}

void print_summary(const std::string& model_name, const std::vector<E2ERunRecord>& records, int n_rounds) {
    // Group by backend
    struct Stats {
        std::vector<double> decode_tps;
        std::vector<double> prefill_tps;
        std::vector<double> ttft_ms;
        int prefill_tokens = 0;
        int decode_tokens = 0;
    };

    // Preserve insertion order
    std::vector<std::string> backend_order;
    std::unordered_map<std::string, Stats> stats;

    for (const auto& r : records) {
        if (r.model != model_name) continue;
        if (stats.find(r.backend) == stats.end()) {
            backend_order.push_back(r.backend);
        }
        auto& s = stats[r.backend];
        s.decode_tps.push_back(r.result.decode_tps);
        s.prefill_tps.push_back(r.result.prefill_tps);
        s.ttft_ms.push_back(r.result.ttft_ms);
        s.prefill_tokens = r.result.prefill_tokens;
        s.decode_tokens = r.result.decode_tokens;
    }

    if (backend_order.empty()) return;

    int prompt_tokens = stats[backend_order[0]].prefill_tokens;
    int gen_tokens = stats[backend_order[0]].decode_tokens;

    std::cerr << "\nModel: " << model_name
              << "  |  Prompt: " << prompt_tokens << " tokens"
              << "  |  Generate: " << gen_tokens << " tokens"
              << "  |  " << n_rounds << " rounds\n\n";

    std::cerr << std::left << std::setw(18) << "Backend"
              << std::right << std::setw(12) << "Decode TPS"
              << std::setw(8) << "+/-"
              << std::setw(14) << "Prefill TPS"
              << std::setw(8) << "+/-"
              << std::setw(10) << "TTFT"
              << std::setw(14) << "vs Cactus" << "\n";

    std::string ruler(84, '-');
    std::cerr << ruler << "\n";

    double cactus_decode = 0.0;
    if (stats.count("cactus")) {
        cactus_decode = mean(stats["cactus"].decode_tps);
    }

    std::cerr << std::fixed;
    for (const auto& name : backend_order) {
        const auto& s = stats[name];
        double d_mean = mean(s.decode_tps);
        double d_std = stddev(s.decode_tps);
        double p_mean = mean(s.prefill_tps);
        double p_std = stddev(s.prefill_tps);
        double t_mean = mean(s.ttft_ms);

        std::cerr << std::left << std::setw(18) << name
                  << std::right
                  << std::setprecision(1) << std::setw(12) << d_mean
                  << std::setprecision(1) << std::setw(8) << d_std
                  << std::setprecision(1) << std::setw(14) << p_mean
                  << std::setprecision(1) << std::setw(8) << p_std
                  << std::setprecision(0) << std::setw(7) << t_mean << "ms";

        if (name == "cactus") {
            std::cerr << std::setw(14) << "baseline";
        } else if (cactus_decode > 0.0) {
            double pct = ((d_mean - cactus_decode) / cactus_decode) * 100.0;
            std::ostringstream pct_str;
            pct_str << std::fixed << std::setprecision(1) << std::showpos << pct << "%";
            std::cerr << std::setw(14) << pct_str.str();
        } else {
            std::cerr << std::setw(14) << "n/a";
        }
        std::cerr << "\n";
    }
    std::cerr << "\n";
}

} // namespace e2e
