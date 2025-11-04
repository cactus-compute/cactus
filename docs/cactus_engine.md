# Cactus Engine FFI Documentation

The Cactus Engine provides a clean C FFI (Foreign Function Interface) for integrating the LLM inference engine into various applications. This documentation covers all available functions, their parameters, and usage examples.

## Types

### `cactus_model_t`
An opaque pointer type representing a loaded model instance. This handle is used throughout the API to reference a specific model.

```c
typedef void* cactus_model_t;
```

### `cactus_token_callback`
Callback function type for streaming token generation. Called for each generated token during completion.

```c
typedef void (*cactus_token_callback)(
    const char* token,      // The generated token text
    uint32_t token_id,      // The token's ID in the vocabulary
    void* user_data        // User-provided context data
);
```

## Core Functions

### `cactus_init`
Initializes a model from disk and prepares it for inference.

```c
cactus_model_t cactus_init(
    const char* model_path,   // Path to the model directory
    size_t context_size      // Maximum context size (e.g., 2048)
);
```

**Returns:** Model handle on success, NULL on failure

**Example:**
```c
cactus_model_t model = cactus_init("../../weights/qwen3-600m", 2048);
if (!model) {
    fprintf(stderr, "Failed to initialize model\n");
    return -1;
}
```

### `cactus_complete`
Performs text completion with optional streaming and tool support.

```c
int cactus_complete(
    cactus_model_t model,           // Model handle
    const char* messages_json,      // JSON array of messages
    char* response_buffer,          // Buffer for response JSON
    size_t buffer_size,            // Size of response buffer
    const char* options_json,       // Optional generation options (can be NULL)
    const char* tools_json,         // Optional tools definition (can be NULL)
    cactus_token_callback callback, // Optional streaming callback (can be NULL)
    void* user_data               // User data for callback (can be NULL)
);
```

**Returns:** Number of tokens generated on success, negative value on error

**Message Format:**
```json
[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is your name?"}
]
```

**Options Format:**
```json
{
    "max_tokens": 256,
    "stop_sequences": ["<|im_end|>", "<end_of_turn>"]
}
```

**Response Format:**
The response is written as JSON to the response_buffer:
```json
{
    "content": "I am Claude, an AI assistant.",
    "token_count": 8,
    "function_call": null
}
```

When tools are provided and the model decides to use one:
```json
{
    "content": "",
    "token_count": 15,
    "function_call": {
        "name": "get_weather",
        "arguments": "{\"location\": \"San Francisco, CA, USA\"}"
    }
}
```

**Example with Streaming:**
```c
// Streaming callback implementation
void streaming_callback(const char* token, uint32_t token_id, void* user_data) {
    printf("%s", token);  // Print each token as it's generated
    fflush(stdout);
}

// Usage
const char* messages = R"([
    {"role": "user", "content": "Tell me a story"}
])";

char response[4096];
int tokens = cactus_complete(model, messages, response, sizeof(response), 
                            NULL, NULL, streaming_callback, NULL);
```

### `cactus_embed`
Generates embeddings for the given text.

```c
int cactus_embed(
    cactus_model_t model,        // Model handle
    const char* text,           // Text to embed
    float* embeddings_buffer,   // Buffer for embedding vector
    size_t buffer_size,        // Buffer size in bytes
    size_t* embedding_dim      // Output: actual embedding dimensions
);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
const char* text = "My name is Henry";
float embeddings[2048];  // Allocate for max expected dimensions
size_t actual_dim = 0;

int result = cactus_embed(model, text, embeddings, 
                         sizeof(embeddings), &actual_dim);
if (result == 0) {
    printf("Generated %zu-dimensional embedding\n", actual_dim);
}
```

### `cactus_stop`
Stops ongoing generation. Useful for implementing early stopping based on custom logic.

```c
void cactus_stop(cactus_model_t model);
```

**Example with Controlled Generation:**
```c
struct ControlData {
    cactus_model_t model;
    int token_count;
};

void control_callback(const char* token, uint32_t token_id, void* user_data) {
    struct ControlData* data = (struct ControlData*)user_data;
    printf("%s", token);
    data->token_count++;
    
    // Stop after 5 tokens
    if (data->token_count >= 5) {
        cactus_stop(data->model);
    }
}

// Usage
struct ControlData control = {model, 0};
cactus_complete(model, messages, response, sizeof(response),
               options, NULL, control_callback, &control);
```

### `cactus_reset`
Resets the model's internal state, clearing any cached context.

```c
void cactus_reset(cactus_model_t model);
```

**Use Cases:**
- Starting a new conversation
- Clearing context between unrelated requests
- Recovering from errors

### `cactus_destroy`
Releases all resources associated with the model.

```c
void cactus_destroy(cactus_model_t model);
```

**Important:** Always call this when done with a model to prevent memory leaks.

## Complete Examples

### Basic Conversation
```c
#include "cactus_ffi.h"
#include <stdio.h>
#include <string.h>

int main() {
    // Initialize model
    cactus_model_t model = cactus_init("path/to/model", 2048);
    if (!model) return -1;
    
    // Prepare conversation
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "What's 2+2?"}
    ])";
    
    // Generate response
    char response[2048];
    int result = cactus_complete(model, messages, response, 
                                sizeof(response), NULL, NULL, NULL, NULL);
    
    if (result > 0) {
        printf("Response: %s\n", response);
    }
    
    cactus_destroy(model);
    return 0;
}
```

### Tool Calling
```c
const char* tools = R"([
    {
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City, State, Country"
                    }
                },
                "required": ["location"]
            }
        }
    }
])";

const char* messages = R"([
    {"role": "user", "content": "What's the weather in Paris?"}
])";

char response[4096];
int result = cactus_complete(model, messages, response, sizeof(response),
                            NULL, tools, NULL, NULL);

// Parse response to check for function_call
printf("Response: %s\n", response);
```

### Computing Similarity with Embeddings
```c
#include <math.h>

float compute_cosine_similarity(cactus_model_t model, 
                               const char* text1, 
                               const char* text2) {
    float embeddings1[2048], embeddings2[2048];
    size_t dim1, dim2;
    
    // Generate embeddings
    cactus_embed(model, text1, embeddings1, sizeof(embeddings1), &dim1);
    cactus_embed(model, text2, embeddings2, sizeof(embeddings2), &dim2);
    
    // Calculate cosine similarity
    float dot_product = 0.0f;
    float norm1 = 0.0f, norm2 = 0.0f;
    
    for (size_t i = 0; i < dim1; i++) {
        dot_product += embeddings1[i] * embeddings2[i];
        norm1 += embeddings1[i] * embeddings1[i];
        norm2 += embeddings2[i] * embeddings2[i];
    }
    
    return dot_product / (sqrtf(norm1) * sqrtf(norm2));
}
```

## Best Practices

1. **Always Check Return Values**: Functions return negative values on error
2. **Buffer Sizes**: Ensure response buffers are large enough (4096 bytes recommended)
3. **Memory Management**: Always call `cactus_destroy()` when done
4. **Thread Safety**: Each model instance should be used from a single thread
5. **Context Management**: Use `cactus_reset()` between unrelated conversations
6. **Streaming**: Implement callbacks for better user experience with long generations

## Error Handling

Most functions return:
- Positive values or 0 on success
- Negative values on error

Common error scenarios:
- Invalid model path
- Insufficient buffer size
- Malformed JSON input
- Out of memory

## Performance Tips

1. **Reuse Model Instances**: Initialize once, use multiple times
2. **Appropriate Context Size**: Use the minimum context size needed
3. **Streaming for UX**: Use callbacks for responsive interfaces
4. **Early Stopping**: Use `cactus_stop()` to avoid unnecessary generation