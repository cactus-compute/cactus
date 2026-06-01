#include "telemetry.h"

namespace cactus {
namespace telemetry {

void init(const char*, const char*, const char*) {}
void setEnabled(bool) {}
void setCloudDisabled(bool) {}
void setAppId(const char*) {}
void setTelemetryEnvironment(const char*, const char*, const char*) {}
void setCloudKey(const char*) {}
void cacheCloudApiKey(const char*) {}
std::string loadCachedCloudApiKey() { return {}; }
void recordInit(const char*, bool, double, const char*) {}
void recordCompletion(const char*, const CompletionMetrics&) {}
void recordCompletion(const char*, bool, double, double, double, int, const char*) {}
void recordEmbedding(const char*, bool, const char*) {}
void recordTranscription(const char*, bool, double, double, double, int, double, const char*) {}
void recordStreamTranscription(const char*, bool, double, double, double, int, double, double, double, int, const char*) {}
void setStreamMode(bool) {}
bool isStreamMode() { return false; }
void markInference(bool) {}
void flush() {}
void shutdown() {}

} // namespace telemetry
} // namespace cactus
