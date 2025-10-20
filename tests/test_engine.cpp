#include "test_utils.h"
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>

const char* g_model_path = "../../weights/qwen3-600m-i8";

const char* g_options = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"]
    })";

struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1000.0;
    }
};

struct StreamingTestData {
    std::vector<std::string> tokens;
    std::vector<uint32_t> token_ids;
    int token_count;
};

void streaming_callback(const char* token, uint32_t token_id, void* user_data) {
    auto* data = static_cast<StreamingTestData*>(user_data);
    data->tokens.push_back(std::string(token));
    data->token_ids.push_back(token_id);
    data->token_count++;
    std::cout << token << std::flush;
}

bool test_ffi() {
    cactus_model_t model = cactus_init(g_model_path, 2048);
    
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise and friendly in your responses."},
        {"role": "user", "content": "What is your name?"}
    ])";
    
    StreamingTestData stream_data;
    stream_data.token_count = 0;

    char response[2048];

    std::cout << "\n=== Streaming ===" << std::endl;
    int result = cactus_complete(model, messages, response, sizeof(response), g_options, nullptr,
                                streaming_callback, &stream_data);
    
    std::cout << "\n=== End of Stream ===\n" << std::endl;
    std::cout << "Final Response JSON:\n" << response << "\n" << std::endl;
    
    cactus_destroy(model);

    return result > 0 && stream_data.token_count > 0;
}

bool test_embeddings() {
    cactus_model_t model = cactus_init(g_model_path, 2048);
    
    if (!model) {
        std::cerr << "Failed to initialize model" << std::endl;
        return false;
    }
    
    const char* text1 = "My name is Henry Ndubuaku";
    const char* text2 = "Your name is Henry Ndubuaku";
    
    std::vector<float> embeddings1(2048);  
    std::vector<float> embeddings2(2048);
    size_t embedding_dim1 = 0;
    size_t embedding_dim2 = 0;
    
    Timer timer1;
    int result1 = cactus_embed(model, text1, embeddings1.data(),
                               embeddings1.size() * sizeof(float), &embedding_dim1);
    (void)result1; 
    double time1 = timer1.elapsed_ms();

    Timer timer2;
    int result2 = cactus_embed(model, text2, embeddings2.data(),
                               embeddings2.size() * sizeof(float), &embedding_dim2);
    (void)result2;  
    double time2 = timer2.elapsed_ms();
    
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < embedding_dim1; ++i) {
        dot_product += embeddings1[i] * embeddings2[i];
        norm1 += embeddings1[i] * embeddings1[i];
        norm2 += embeddings2[i] * embeddings2[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    float cosine_similarity = dot_product / (norm1 * norm2);
    
    std::cout << "\n=== Embedding Test Results ===" << std::endl;
    std::cout << "Text 1: \"" << text1 << "\"" << std::endl;
    std::cout << "Text 2: \"" << text2 << "\"" << std::endl;
    std::cout << "Embedding dimension: " << embedding_dim1 << std::endl;
    std::cout << "Time for text1: " << time1 << " ms" << std::endl;
    std::cout << "Time for text2: " << time2 << " ms" << std::endl;
    std::cout << "Cosine similarity: " << cosine_similarity << std::endl;
    
    cactus_destroy(model);
    return true;
}

bool test_generation_control() {
    cactus_model_t model = cactus_init(g_model_path, 2048);
    
    if (!model) {
        std::cerr << "Failed to initialize model for control test" << std::endl;
        return false;
    }
    
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count to 10"}
    ])";
    
    char response[2048];
    
    std::cout << "\n=== Generation Control Test ===" << std::endl;
    
    struct ControlData {
        cactus_model_t model;
        int token_count = 0;
    } control_data;
    
    control_data.model = model;
    
    auto control_callback = [](const char* token, uint32_t token_id, void* user_data) {
        (void)token_id;  
        auto* data = static_cast<ControlData*>(user_data);
        std::cout << token << std::flush;
        data->token_count++;
        
        if (data->token_count >= 5) {
            std::cout << "\n[Stopping after 5 tokens...]" << std::endl;
            cactus_stop(data->model);
        }
    };
    
    int result = cactus_complete(model, messages, response, sizeof(response), 
                                g_options, nullptr, control_callback, &control_data);
    
    std::cout << "\n[Test complete]" << std::endl;
    std::cout << "Generated " << control_data.token_count << " tokens" << std::endl;
    
    cactus_destroy(model);
    return result > 0;
}

bool test_incremental_processing() {
    cactus_model_t model = cactus_init(g_model_path, 2048);
    
    if (!model) {
        std::cerr << "Failed to initialize model for incremental test" << std::endl;
        return false;
    }
    
    char response1[2048];
    char response2[2048];
    
    const char* first_messages = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Henry."}
    ])";
    
    std::cout << "\n=== Incremental Processing Test ===" << std::endl;
    int result1 = cactus_complete(model, first_messages, response1, sizeof(response1), g_options, nullptr, nullptr, nullptr);
    std::cout << "Response 1: " << response1 << "\n" << std::endl;
    
    const char* second_messages = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Henry."},
        {"role": "assistant", "content": "Nice to meet you, Henry! How can I help you today?"},
        {"role": "user", "content": "What is my name?"}
    ])";
    
    int result2 = cactus_complete(model, second_messages, response2, sizeof(response2), g_options, nullptr, nullptr, nullptr);
    std::cout << "Response 2: " << response2 << "\n" << std::endl;
    
    cactus_destroy(model);
    
    return result1 > 0 && result2 > 0;
}

bool test_ffi_with_tools() {
    cactus_model_t model = cactus_init(g_model_path, 2048);
    
    if (!model) {
        std::cerr << "Failed to initialize model for tools test" << std::endl;
        return false;
    }
    
        const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "Instruction: Limit your response to 100 words Set an alarm for 9 a.m"}
    ])";
    
    const char* tools = R"([
        {
            "function": {
                "name": "create_note",
                "description": "Create a new note with the given user text",
                "parameters": {
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text of the note"
                        }
                    },
                    "required": ["text"]
                }
            }
        },
        {
            "function": {
                "name": "set_timer",
                "description": "Set a timer for the specified duration, as a sum of the parameters",
                "parameters": {
                    "properties": {
                        "time_hours": {
                            "type": "number",
                            "description": "The number of hours on the timer"
                        },
                        "time_minutes": {
                            "type": "number",
                            "description": "The number of minutes on the timer"
                        },
                        "time_seconds": {
                            "type": "number",
                            "description": "The number of seconds on the timer"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "function": {
                "name": "set_alarm",
                "description": "Set an alarm for a specified time using 24-hour format",
                "parameters": {
                    "properties": {
                        "time_hours": {
                            "type": "number",
                            "description": "The hour component of the alarm time in 24 hour format from 0 to 23"
                        },
                        "time_minutes": {
                            "type": "number",
                            "description": "The minute component of the alarm time from 0 to 59"
                        }
                    },
                    "required": ["time_hours", "time_minutes"]
                }
            }
        },
        {
            "function": {
                "name": "set_timer_absolute",
                "description": "Set a timer that will end at the specified time such as 12pm or tomorrow at 7am",
                "parameters": {
                    "properties": {
                        "day_offset": {
                            "type": "string",
                            "description": "The offset of the day such as tomorrow, today, thursday, or a number of days"
                        },
                        "time_hours": {
                            "type": "number",
                            "description": "The hour component of the desired end time in 24 hour format"
                        },
                        "time_minutes": {
                            "type": "number",
                            "description": "The minute component of the desired end time from 0 to 59"
                        }
                    },
                    "required": ["time_hours", "time_minutes"]
                }
            }
        },
        {
            "function": {
                "name": "reminder_absolute",
                "description": "Create a reminder for the user at a specific time",
                "parameters": {
                    "properties": {
                        "day_offset": {
                            "type": "string",
                            "description": "The offset of the day to remind the user at such as tomorrow, today, or thursday"
                        },
                        "absolute_time_hour": {
                            "type": "number",
                            "description": "The absolute time to remind the user at as a 24 hour hour part such as 17"
                        },
                        "absolute_time_minute": {
                            "type": "number",
                            "description": "The absolute time to remind the user at as a minute part such as 30"
                        },
                        "date_month_day": {
                            "type": "string",
                            "description": "The date to remind the user at in month-day format such as 12-31"
                        },
                        "date_year": {
                            "type": "number",
                            "description": "The year to remind the user at such as 2022"
                        },
                        "message": {
                            "type": "string",
                            "description": "The message to remind the user such as Buy more milk"
                        }
                    },
                    "required": ["absolute_time_hour", "absolute_time_minute", "message"]
                }
            }
        },
        {
            "function": {
                "name": "reminder_next_time",
                "description": "Get the next AM/PM agnostic human instance of a given time, for example if 2:45 is given and it is 13:00, the next instance will be 14:45",
                "parameters": {
                    "properties": {
                        "time_hour": {
                            "type": "number",
                            "description": "The hour component specified by the user"
                        },
                        "time_minute": {
                            "type": "number",
                            "description": "The minute component specified by the user"
                        }
                    },
                    "required": ["time_hour", "time_minute"]
                }
            }
        },
        {
            "function": {
                "name": "reminder_relative",
                "description": "Create a reminder for the user at a relative time such as in 5 minutes",
                "parameters": {
                    "properties": {
                        "relative_time": {
                            "type": "number",
                            "description": "The relative time value such as 5 for 5 minutes"
                        },
                        "time_unit": {
                            "type": "string",
                            "description": "The unit of time: seconds, minutes, hours, days, weeks, months, years"
                        },
                        "message": {
                            "type": "string",
                            "description": "The reminder message"
                        }
                    },
                    "required": ["relative_time", "time_unit", "message"]
                }
            }
        },
        {
            "function": {
                "name": "evaluate_js",
                "description": "Evaluate JavaScript code and return the result",
                "parameters": {
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "JavaScript code to evaluate"
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    ])";
    
    StreamingTestData stream_data;
    stream_data.token_count = 0;
    
    char response[4096];
    
    std::cout << "User: What's the weather in San Francisco?" << std::endl;
    std::cout << "Assistant: ";
    
    int result = cactus_complete(model, messages, response, sizeof(response), g_options, tools,
                                streaming_callback, &stream_data);
    
    std::cout << "\n\n=== Tool Response ===" << std::endl;
    std::cout << "Final Response JSON:\n" << response << "\n" << std::endl;
    
    std::string response_str(response);
    bool has_tool_mention = response_str.find("tool_call") != std::string::npos ||
                            response_str.find("get_weather") != std::string::npos ||
                            response_str.find("San Francisco") != std::string::npos;
    
    std::cout << "Tool recognition: " << (has_tool_mention ? "PASSED" : "FAILED") << std::endl;
    
    cactus_destroy(model);
    return result > 0 && stream_data.token_count > 0;
}

int main() {
    TestUtils::TestRunner runner("Engine Tests");
    runner.run_test("engine_forward_decode_benchmark", test_ffi());  
    runner.run_test("text_embeddings", test_embeddings());
    runner.run_test("generation_control", test_generation_control());
    runner.run_test("incremental_processing", test_incremental_processing());
    runner.run_test("ffi_with_tools", test_ffi_with_tools());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}