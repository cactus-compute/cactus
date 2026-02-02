#ifndef CACTUS_NET_H
#define CACTUS_NET_H

#include <string>
#include <functional>
#include <memory>

namespace cactus {
namespace net {

// HTTP response structure
struct HttpResponse {
    int status_code;          // HTTP status code (200, 400, 500, etc.) or -1 for network error
    std::string body;         // Response body
    std::string error;        // Error message if status_code < 0
    bool success;             // true if status_code >= 200 && status_code < 300
};

// HTTP request configuration
struct HttpRequest {
    std::string url;
    std::string body;
    std::string content_type;  // Default: "application/json"
    std::string authorization; // Bearer token or API key header value
    int timeout_ms;            // Request timeout in milliseconds (default: 30000)

    HttpRequest() : content_type("application/json"), timeout_ms(30000) {}
};

// Cloud provider configuration
struct CloudConfig {
    std::string provider;      // "openai", "anthropic", "gcp", "custom"
    std::string api_key;
    std::string endpoint_url;  // Optional: override default endpoint
    std::string model;         // Cloud model to use (e.g., "gpt-4", "claude-3-opus")

    bool is_configured() const {
        return !provider.empty() && !api_key.empty();
    }
};

// Synchronous HTTP POST
// Returns HttpResponse with result or error
// NOTE (Android): Uses JNI Modified UTF-8 for string passing. Supplementary Unicode
// characters (code points > U+FFFF, like emoji) may not be handled correctly.
// For typical API requests (ASCII URLs, JSON bodies), this is not an issue.
HttpResponse http_post(const HttpRequest& request);

// Check if network/HTTP is available on this platform
bool is_network_available();

// Cloud configuration management
void set_cloud_config(const CloudConfig& config);
CloudConfig get_cloud_config();  // Returns copy for thread safety
bool is_cloud_configured();

// High-level cloud completion
// Takes messages JSON (same format as cactus_complete) and returns cloud response
// This handles provider-specific API formatting internally
HttpResponse cloud_complete(const std::string& messages_json,
                            const std::string& options_json = "");

} // namespace net
} // namespace cactus

#endif // CACTUS_NET_H
