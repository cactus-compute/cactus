// Network stub implementation for platforms without native HTTP support
// On Android, net_android.cpp provides the real implementation via JNI
// On Apple, net_apple.mm provides the real implementation via NSURLSession

#include "net.h"

namespace cactus {
namespace net {

// Global cloud configuration
static CloudConfig g_cloud_config;

HttpResponse http_post(const HttpRequest& /*request*/) {
    HttpResponse response;
    response.status_code = -1;
    response.success = false;
    response.error = "Network not available: HTTP support not compiled for this platform";
    return response;
}

bool is_network_available() {
    return false;
}

void set_cloud_config(const CloudConfig& config) {
    g_cloud_config = config;
}

CloudConfig get_cloud_config() {
    return g_cloud_config;  // Returns copy for API consistency
}

bool is_cloud_configured() {
    return g_cloud_config.is_configured();
}

HttpResponse cloud_complete(const std::string& /*messages_json*/,
                            const std::string& /*options_json*/) {
    HttpResponse response;
    response.status_code = -1;
    response.success = false;
    response.error = "Cloud fallback not available: HTTP support not compiled for this platform";
    return response;
}

} // namespace net
} // namespace cactus
