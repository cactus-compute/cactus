#ifndef E2E_DRIVER_H
#define E2E_DRIVER_H

#include <string>
#include <vector>
#include <functional>

namespace e2e {

struct E2EResult {
    double prefill_tps;      // prompt tokens / prefill time
    double decode_tps;       // generated tokens / decode time
    double ttft_ms;          // time to first token (ms)
    int    prefill_tokens;   // number of prompt tokens
    int    decode_tokens;    // number of generated tokens
    double total_ms;         // wall clock total
};

struct E2EBackend {
    const char* name;        // "cactus", "llama_cpp", etc.
    const char* framework;   // for --backends filtering

    bool (*available)();
    void* (*load)(const char* model_path, int threads);
    E2EResult (*generate)(void* handle, const char* prompt, int max_tokens);
    void (*unload)(void* handle);
};

struct ScheduledRun {
    int backend_idx;
    int round;
};

std::vector<ScheduledRun> build_schedule(int n_backends, int n_rounds = 10);

void register_e2e_backend(E2EBackend b);
const std::vector<E2EBackend>& get_e2e_backends();

struct E2ERunRecord {
    std::string backend;
    std::string model;
    int rep;
    E2EResult result;
};

void write_csv(const std::string& path, const std::vector<E2ERunRecord>& records);
void print_summary(const std::string& model_name, const std::vector<E2ERunRecord>& records, int n_rounds);

} // namespace e2e

#endif
