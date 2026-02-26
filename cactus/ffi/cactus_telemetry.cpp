#include "cactus_ffi.h"
#include "telemetry/telemetry.h"

extern "C" {

void cactus_set_telemetry_environment(const char* framework, const char* cache_location) {
    cactus::telemetry::setTelemetryEnvironment(framework, cache_location);
}

void cactus_telemetry_flush(void) {
    cactus::telemetry::flush();
}

void cactus_telemetry_shutdown(void) {
    cactus::telemetry::shutdown();
}

}
