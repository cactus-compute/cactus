#include "../cactus_engine.h"
#include <cstdio>
#include <cstdlib>
#include <string>

static const char* MESSAGES = R"([
    {"role": "system", "content": "You are a helpful assistant that can use tools."},
    {"role": "user", "content": "Send a message to Bob and get the weather for San Francisco."}
])";

static const char* TOOLS = R"([{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City, State, Country"}},
            "required": ["location"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": "Send a message to a contact",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Name of the person"},
                "message": {"type": "string", "description": "The message content"}
            },
            "required": ["recipient", "message"]
        }
    }
}])";

static const char* OPTIONS = R"({
    "max_tokens": 40,
    "temperature": 0.0,
    "force_tools": true,
    "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
    "telemetry_enabled": false
})";

static int g_step = 0;

static void token_callback(const char* text, uint32_t token_id, void* /*ud*/) {
    fprintf(stderr, "[step %d] tok=%u text=%s\n", g_step++, token_id, text ? text : "");
    printf("%u\n", token_id);
    fflush(stdout); fflush(stderr);
}

int main() {
    const char* model_path = std::getenv("CACTUS_TEST_MODEL");
    if (!model_path) { fprintf(stderr, "CACTUS_TEST_MODEL not set\n"); return 1; }

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    char response[4096] = {};
    cactus_complete(model, MESSAGES, response, sizeof(response),
                    OPTIONS, TOOLS, token_callback, nullptr, nullptr, 0);

    cactus_destroy(model);
    return 0;
}
