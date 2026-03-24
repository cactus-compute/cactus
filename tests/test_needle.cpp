#include "../cactus/ffi/cactus_ffi.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

constexpr size_t kResponseBufferSize = 65536;

void stream_callback(const char* token, uint32_t /*token_id*/, void* /*user_data*/) {
    if (token) {
        std::cout << token << std::flush;
    }
}

const char* last_error_or(const char* fallback) {
    const char* err = cactus_get_last_error();
    return err ? err : fallback;
}

} // namespace

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : std::getenv("CACTUS_TEST_MODEL");
    if (!model_path || std::strlen(model_path) == 0) {
        std::cerr << "Usage: " << argv[0] << " <needle_model_dir>\n";
        std::cerr << "   or: CACTUS_TEST_MODEL=/path/to/needle-model " << argv[0] << "\n";
        return 1;
    }

    const char* messages_json = R"([
  {
    "role": "user",
    "content": "check the weather in San Francisco this afternoon and tell me if I should bring an umbrella"
  }
])";

    const char* tools_json = R"([
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the weather forecast for a location and time.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": { "type": "string" },
          "time": { "type": "string" }
        },
        "required": ["location"]
      }
    }
  }
])";

    const char* options_json = R"({
  "max_tokens": 64,
  "temperature": 0.0,
  "top_p": 0.0,
  "top_k": 0,
  "telemetry_enabled": false
})";

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        std::cerr << "cactus_init failed: " << last_error_or("unknown error") << "\n";
        return 1;
    }

    char response[kResponseBufferSize] = {0};

    std::cout << "Model: " << model_path << "\n";
    std::cout << "Prompt:\n" << messages_json << "\n";
    std::cout << "Tools:\n" << tools_json << "\n";
    std::cout << "\nStreaming response:\n";

    int rc = cactus_complete(
        model,
        messages_json,
        response,
        sizeof(response),
        options_json,
        tools_json,
        stream_callback,
        nullptr
    );

    std::cout << "\n\n";

    if (rc < 0) {
        std::cerr << "cactus_complete failed: " << last_error_or("unknown error") << "\n";
        cactus_destroy(model);
        return 1;
    }

    std::cout << "Raw completion JSON:\n" << response << "\n";

    cactus_destroy(model);
    return 0;
}
