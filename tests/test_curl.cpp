#include "test_utils.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#if __has_include(<curl/curl.h>)
#include <curl/curl.h>
#define CACTUS_TEST_HAS_CURL 1
#else
#define CACTUS_TEST_HAS_CURL 0
#endif

bool test_curl_version_info() {
#if !CACTUS_TEST_HAS_CURL
    return false;
#else
    curl_version_info_data* info = curl_version_info(CURLVERSION_NOW);
    if (!info) return false;
    if (!info->version || std::string(info->version).empty()) return false;
    if (!info->host || std::string(info->host).empty()) return false;
    return true;
#endif
}

bool test_curl_easy_init() {
#if !CACTUS_TEST_HAS_CURL
    return false;
#else
    CURL* handle = curl_easy_init();
    if (!handle) return false;
    curl_easy_cleanup(handle);
    return true;
#endif
}

bool test_curl_url_api() {
#if !CACTUS_TEST_HAS_CURL
    return false;
#else
    CURL* handle = curl_easy_init();
    if (!handle) return false;

    bool ok = true;
    ok = ok && (curl_easy_setopt(handle, CURLOPT_URL, "https://example.com/api/v1/ping?x=1") == CURLE_OK);
    ok = ok && (curl_easy_setopt(handle, CURLOPT_NOBODY, 1L) == CURLE_OK);
    ok = ok && (curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, 200L) == CURLE_OK);
    ok = ok && (curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, 200L) == CURLE_OK);

    curl_easy_cleanup(handle);
    return ok;
#endif
}

bool test_curl_http_request() {
#if !CACTUS_TEST_HAS_CURL
    return false;
#else
    CURL* handle = curl_easy_init();
    if (!handle) return false;

    bool ok = true;
    ok = ok && (curl_easy_setopt(handle, CURLOPT_URL, "https://example.com/") == CURLE_OK);
    ok = ok && (curl_easy_setopt(handle, CURLOPT_NOBODY, 1L) == CURLE_OK);
    ok = ok && (curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L) == CURLE_OK);
    ok = ok && (curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, 3000L) == CURLE_OK);
    ok = ok && (curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, 2000L) == CURLE_OK);
    if (!ok) {
        curl_easy_cleanup(handle);
        return false;
    }

    CURLcode rc = curl_easy_perform(handle);
    long http_code = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(handle);

    return (rc == CURLE_OK) && (http_code >= 100);
#endif
}

struct CurlHttpCheck {
    bool pass = false;
    bool skip = false;
    std::string reason;
    long http_code = 0;
    std::string body_preview;
};

static size_t curl_write_to_string(char* ptr, size_t size, size_t nmemb, void* userdata) {
    if (!userdata) return 0;
    std::string* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

static bool curl_supports_protocol(const char* name) {
#if !CACTUS_TEST_HAS_CURL
    (void)name;
    return false;
#else
    curl_version_info_data* info = curl_version_info(CURLVERSION_NOW);
    if (!info || !info->protocols) return false;
    for (const char* const* p = info->protocols; *p; ++p) {
        if (std::string(*p) == name) return true;
    }
    return false;
#endif
}

static void configure_curl_tls(CURL* handle) {
#if CACTUS_TEST_HAS_CURL
    if (!handle) return;
    curl_easy_setopt(handle, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(handle, CURLOPT_SSL_VERIFYHOST, 2L);
    const char* ca_bundle = std::getenv("CACTUS_CA_BUNDLE");
    if (ca_bundle && ca_bundle[0] != '\0') {
        curl_easy_setopt(handle, CURLOPT_CAINFO, ca_bundle);
    }
#if defined(__ANDROID__)
    const char* ca_path = std::getenv("CACTUS_CA_PATH");
    if (ca_path && ca_path[0] != '\0') {
        curl_easy_setopt(handle, CURLOPT_CAPATH, ca_path);
    } else {
        curl_easy_setopt(handle, CURLOPT_CAPATH, "/system/etc/security/cacerts");
    }
#endif
#else
    (void)handle;
#endif
}

CurlHttpCheck run_curl_http_request_check() {
#if !CACTUS_TEST_HAS_CURL
    return {false, true, "curl/curl.h not available in include path"};
#else
    if (!curl_supports_protocol("https")) {
        return {false, false, "libcurl built without HTTPS protocol support", 0, ""};
    }

    CURL* handle = curl_easy_init();
    if (!handle) return {false, false, "curl_easy_init returned null", 0, ""};

    bool opts_ok = true;
    std::string body;
    opts_ok = opts_ok && (curl_easy_setopt(handle, CURLOPT_URL, "https://sha256.badssl.com/") == CURLE_OK);
    opts_ok = opts_ok && (curl_easy_setopt(handle, CURLOPT_HTTPGET, 1L) == CURLE_OK);
    opts_ok = opts_ok && (curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L) == CURLE_OK);
    opts_ok = opts_ok && (curl_easy_setopt(handle, CURLOPT_USERAGENT, "cactus-curl-test/1.0") == CURLE_OK);
    opts_ok = opts_ok && (curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, curl_write_to_string) == CURLE_OK);
    opts_ok = opts_ok && (curl_easy_setopt(handle, CURLOPT_WRITEDATA, &body) == CURLE_OK);
    opts_ok = opts_ok && (curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, 3000L) == CURLE_OK);
    opts_ok = opts_ok && (curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, 2000L) == CURLE_OK);
    configure_curl_tls(handle);
    if (!opts_ok) {
        curl_easy_cleanup(handle);
        return {false, false, "failed to configure curl options", 0, ""};
    }

    CURLcode rc = curl_easy_perform(handle);
    long http_code = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(handle);
    std::string body_preview = body.substr(0, std::min<size_t>(body.size(), 400));

    if (rc == CURLE_OK && http_code >= 200 && http_code < 400) {
        if (body.find("badssl.com") != std::string::npos) {
            return {true, false, "", http_code, body_preview};
        }
        std::ostringstream oss;
        oss << "https request succeeded but response body did not contain expected marker";
        return {false, false, oss.str(), http_code, body_preview};
    }

    std::ostringstream oss;
    oss << "request failed rc=" << static_cast<int>(rc)
        << " (" << curl_easy_strerror(rc) << "), http_code=" << http_code;
    return {false, false, oss.str(), http_code, body_preview};
#endif
}

int main() {
    TestUtils::TestRunner runner("Curl Tests");

#if !CACTUS_TEST_HAS_CURL
    runner.log_skip("curl_headers", "curl/curl.h not available in include path");
    runner.print_summary();
    return 0;
#else
    CURLcode init_rc = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (init_rc != CURLE_OK) {
        runner.run_test("global_init", false);
        runner.print_summary();
        return 1;
    }

    runner.run_test("version_info", test_curl_version_info());
    runner.run_test("easy_init", test_curl_easy_init());
    runner.run_test("url_api", test_curl_url_api());
    CurlHttpCheck http_check = run_curl_http_request_check();
    if (http_check.skip) {
        runner.log_skip("http_request", http_check.reason);
    } else {
        runner.run_test("http_request", http_check.pass);
        if (http_check.http_code > 0) {
            std::cout << "  http_code: " << http_check.http_code << "\n";
        }
        if (!http_check.body_preview.empty()) {
            std::cout << "  body_preview: " << http_check.body_preview << "\n";
        }
        if (!http_check.pass && !http_check.reason.empty()) {
            std::cout << "  reason: " << http_check.reason << "\n";
        }
    }

    curl_global_cleanup();
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
#endif
}
