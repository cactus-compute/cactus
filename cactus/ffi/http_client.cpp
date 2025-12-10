#ifdef __APPLE__
#include <curl/curl.h>
#include <iostream>
#endif

#include <string>
#include <map>

namespace cactus {
namespace ffi {

class HttpClient {
public:
    struct Response {
        bool success;
        int status_code;
        std::string body;
    };

    static Response postJson(
        const std::string& url,
        const std::map<std::string, std::string>& headers,
        const std::string& json_body
    );

private:
    static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp);
};

size_t HttpClient::writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::string* response = static_cast<std::string*>(userp);
    response->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

HttpClient::Response HttpClient::postJson(
    const std::string& url,
    const std::map<std::string, std::string>& headers,
    const std::string& json_body
) {
#ifdef __APPLE__
    Response response;
    response.success = false;
    response.status_code = 0;

    CURL* curl = curl_easy_init();
    if (!curl) {
        return response;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    curl_easy_setopt(curl, CURLOPT_POST, 1L);

    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_body.length());

    struct curl_slist* header_list = nullptr;
    for (const auto& header : headers) {
        std::string header_str = header.first + ": " + header.second;
        header_list = curl_slist_append(header_list, header_str.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response.body);

    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

    CURLcode res = curl_easy_perform(curl);

    if (res == CURLE_OK) {
        long response_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        response.status_code = static_cast<int>(response_code);
        response.success = (response_code >= 200 && response_code < 300);

        if (!response.success && !response.body.empty()) {
            std::cerr << "[Telemetry] Response body: " << response.body << std::endl;
        }
    } else {
        std::cerr << "[Telemetry] HTTP POST failed: " << curl_easy_strerror(res) << std::endl;
    }

    if (header_list) {
        curl_slist_free_all(header_list);
    }
    curl_easy_cleanup(curl);

    return response;
#else
    Response response;
    response.success = false;
    response.status_code = 0;
    return response;
#endif
}

} // namespace ffi
} // namespace cactus
