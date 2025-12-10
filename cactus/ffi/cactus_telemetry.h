#ifndef CACTUS_TELEMETRY_H
#define CACTUS_TELEMETRY_H

#include <string>

namespace cactus {
namespace ffi {

class TelemetryCollector {
public:
    static TelemetryCollector& getInstance();

    void recordInit(const std::string& model, bool success, const std::string& message = "");

    void recordCompletion(const std::string& model, bool success,
                         double ttft_ms, double tps, double response_time_ms,
                         int tokens, const std::string& message = "");

    void recordEmbedding(const std::string& model, bool success,
                        const std::string& message = "");

    void recordTranscription(const std::string& model, bool success,
                           double response_time_ms, int audio_duration_ms,
                           const std::string& message = "");
};

} // namespace ffi
} // namespace cactus

#ifdef __cplusplus
extern "C" {
#endif

void cactus_set_telemetry_enabled(int enabled);
void cactus_set_telemetry_token(const char* token);

#ifdef __cplusplus
}
#endif

#endif // CACTUS_TELEMETRY_H
