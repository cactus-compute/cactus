#include "cactus_ffi.h"
#include "telemetry/telemetry.h"

extern "C" {

void cactus_set_telemetry_environment(const char* framework, const char* cache_location, const char* version) {
    cactus::telemetry::setTelemetryEnvironment(framework, cache_location, version);
}

void cactus_set_app_id(const char* app_id) {
    cactus::telemetry::setAppId(app_id);
}

void cactus_telemetry_flush(void) {
    cactus::telemetry::flush();
}

void cactus_telemetry_shutdown(void) {
    cactus::telemetry::shutdown();
}

}
