# Cactus Engine FFI Documentation

The Cactus Engine provides a clean C FFI (Foreign Function Interface) for integrating the LLM inference engine into various applications. This documentation covers all available functions, their parameters, and usage examples.

## Getting Started

Before using the Cactus Engine, you need to download model weights:

```bash
# Setup the environment
./setup

# Download model weights (converts HuggingFace models to Cactus format)
cactus download LiquidAI/LFM2-1.2B

# Or download a vision-language model
cactus download LiquidAI/LFM2-VL-450M

# Or download a whisper model for transcription
cactus download openai/whisper-small
```

Weights are saved to the `weights/` directory and can be loaded using `cactus_init()`.

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
    void* user_data         // User-provided context data
);
```

## Core Functions

### `cactus_init`
Initializes a model from disk and prepares it for inference.

```c
cactus_model_t cactus_init(
    const char* model_path,   // Path to the model directory
    size_t context_size,      // Maximum context size (e.g., 2048)
    const char* corpus_dir    // Optional path to corpus directory for RAG (can be NULL)
);
```

**Returns:** Model handle on success, NULL on failure

**Example:**
```c
// Basic initialization
cactus_model_t model = cactus_init("../../weights/qwen3-600m", 2048, NULL);
if (!model) {
    fprintf(stderr, "Failed to initialize model\n");
    return -1;
}

// With RAG corpus (only works om LFM2 RAG version for now)
cactus_model_t rag_model = cactus_init("../../weights/lfm2-rag", 512, "./documents");
```

### `cactus_complete`
Performs text completion with optional streaming and tool support.

```c
int cactus_complete(
    cactus_model_t model,           // Model handle
    const char* messages_json,      // JSON array of messages
    char* response_buffer,          // Buffer for response JSON
    size_t buffer_size,             // Size of response buffer
    const char* options_json,       // Optional generation options (can be NULL)
    const char* tools_json,         // Optional tools definition (can be NULL)
    cactus_token_callback callback, // Optional streaming callback (can be NULL)
    void* user_data                 // User data for callback (can be NULL)
);
```

**Returns:** Number of bytes written to response_buffer on success, negative value on error

**Message Format:**
```json
[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is your name?"}
]
```

**Messages with Images (for VLM models):**
```json
[
    {"role": "user", "content": "Describe this image", "images": ["/path/to/image.jpg"]}
]
```

**Options Format:**
```json
{
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "stop_sequences": ["<|im_end|>", "<end_of_turn>"]
}
```

**Response Format:**
```json
{
    "success": true,
    "response": "I am an AI assistant.",
    "time_to_first_token_ms": 150.5,
    "total_time_ms": 1250.3,
    "tokens_per_second": 45.2,
    "prompt_tokens": 25,
    "completion_tokens": 8
}
```

**Response with Function Call:**
```json
{
    "success": true,
    "response": "",
    "function_calls": [
        {
            "name": "get_weather",
            "arguments": "{\"location\": \"San Francisco, CA, USA\"}"
        }
    ],
    "time_to_first_token_ms": 120.0,
    "total_time_ms": 450.5,
    "tokens_per_second": 38.5,
    "prompt_tokens": 45,
    "completion_tokens": 15
}
```

**Example with Streaming:**
```c
void streaming_callback(const char* token, uint32_t token_id, void* user_data) {
    printf("%s", token);
    fflush(stdout);
}

const char* messages = "[{\"role\": \"user\", \"content\": \"Tell me a story\"}]";

char response[8192];
int result = cactus_complete(model, messages, response, sizeof(response),
                             NULL, NULL, streaming_callback, NULL);
```

### `cactus_transcribe`
Transcribes audio to text using a Whisper model.

```c
int cactus_transcribe(
    cactus_model_t model,           // Model handle (must be Whisper model)
    const char* audio_file_path,    // Path to audio file (WAV, MP3, etc.)
    const char* prompt,             // Optional prompt to guide transcription (can be NULL)
    char* response_buffer,          // Buffer for response JSON
    size_t buffer_size,             // Size of response buffer
    const char* options_json,       // Optional transcription options (can be NULL)
    cactus_token_callback callback, // Optional streaming callback (can be NULL)
    void* user_data                 // User data for callback (can be NULL)
);
```

**Returns:** Number of bytes written to response_buffer on success, negative value on error

**Example:**
```c
cactus_model_t whisper = cactus_init("../../weights/whisper-base", 448, NULL);

char response[16384];
int result = cactus_transcribe(whisper, "audio.wav", NULL,
                                response, sizeof(response), NULL, NULL, NULL);
if (result > 0) {
    printf("Transcription: %s\n", response);
}
```

### `cactus_embed`
Generates text embeddings for semantic search, similarity, and RAG applications.

```c
int cactus_embed(
    cactus_model_t model,        // Model handle
    const char* text,            // Text to embed
    float* embeddings_buffer,    // Buffer for embedding vector
    size_t buffer_size,          // Buffer size in bytes
    size_t* embedding_dim        // Output: actual embedding dimensions
);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
const char* text = "The quick brown fox jumps over the lazy dog";
float embeddings[2048];
size_t actual_dim = 0;

int result = cactus_embed(model, text, embeddings, sizeof(embeddings), &actual_dim);
if (result == 0) {
    printf("Generated %zu-dimensional embedding\n", actual_dim);
}
```

### `cactus_image_embed`
Generates embeddings for images, useful for multimodal retrieval tasks.

```c
int cactus_image_embed(
    cactus_model_t model,        // Model handle (must support vision)
    const char* image_path,      // Path to image file
    float* embeddings_buffer,    // Buffer for embedding vector
    size_t buffer_size,          // Buffer size in bytes
    size_t* embedding_dim        // Output: actual embedding dimensions
);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
float image_embeddings[1024];
size_t dim = 0;

int result = cactus_image_embed(model, "photo.jpg", image_embeddings,
                                 sizeof(image_embeddings), &dim);
if (result == 0) {
    printf("Image embedding dimension: %zu\n", dim);
}
```

### `cactus_audio_embed`
Generates embeddings for audio files, useful for audio retrieval and classification.

```c
int cactus_audio_embed(
    cactus_model_t model,        // Model handle (must support audio)
    const char* audio_path,      // Path to audio file
    float* embeddings_buffer,    // Buffer for embedding vector
    size_t buffer_size,          // Buffer size in bytes
    size_t* embedding_dim        // Output: actual embedding dimensions
);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
float audio_embeddings[768];
size_t dim = 0;

int result = cactus_audio_embed(model, "speech.wav", audio_embeddings,
                                 sizeof(audio_embeddings), &dim);
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
    int max_tokens;
};

void control_callback(const char* token, uint32_t token_id, void* user_data) {
    struct ControlData* data = (struct ControlData*)user_data;
    printf("%s", token);
    data->token_count++;

    // Stop after reaching limit
    if (data->token_count >= data->max_tokens) {
        cactus_stop(data->model);
    }
}

struct ControlData control = {model, 0, 50};
cactus_complete(model, messages, response, sizeof(response),
                NULL, NULL, control_callback, &control);
```

### `cactus_reset`
Resets the model's internal state, clearing KV cache and any cached context.

```c
void cactus_reset(cactus_model_t model);
```

**Use Cases:**
- Starting a new conversation
- Clearing context between unrelated requests
- Recovering from errors
- Freeing memory after long conversations

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
    cactus_model_t model = cactus_init("path/to/model", 2048, NULL);
    if (!model) return -1;

    // Prepare conversation
    const char* messages =
        "[{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},"
        " {\"role\": \"user\", \"content\": \"Hello!\"},"
        " {\"role\": \"assistant\", \"content\": \"Hello! How can I help you today?\"},"
        " {\"role\": \"user\", \"content\": \"What's 2+2?\"}]";

    // Generate response
    char response[4096];
    int result = cactus_complete(model, messages, response,
                                 sizeof(response), NULL, NULL, NULL, NULL);

    if (result > 0) {
        printf("Response: %s\n", response);
    }

    cactus_destroy(model);
    return 0;
}
```

### Vision-Language Model (VLM)
```c
#include "cactus_ffi.h"

int main() {
    cactus_model_t vlm = cactus_init("path/to/lfm2-vlm", 4096, NULL);
    if (!vlm) return -1;

    // Message with image
    const char* messages =
        "[{\"role\": \"user\","
        "  \"content\": \"What do you see in this image?\","
        "  \"images\": [\"/path/to/photo.jpg\"]}]";

    char response[8192];
    int result = cactus_complete(vlm, messages, response, sizeof(response),
                                 NULL, NULL, NULL, NULL);

    if (result > 0) {
        printf("%s\n", response);
    }

    cactus_destroy(vlm);
    return 0;
}
```

### Tool Calling
```c
const char* tools =
    "[{\"function\": {"
    "    \"name\": \"get_weather\","
    "    \"description\": \"Get weather for a location\","
    "    \"parameters\": {"
    "        \"type\": \"object\","
    "        \"properties\": {"
    "            \"location\": {\"type\": \"string\", \"description\": \"City, State, Country\"}"
    "        },"
    "        \"required\": [\"location\"]"
    "    }"
    "}}]";

const char* messages = "[{\"role\": \"user\", \"content\": \"What's the weather in Paris?\"}]";

char response[4096];
int result = cactus_complete(model, messages, response, sizeof(response),
                             NULL, tools, NULL, NULL);

printf("Response: %s\n", response);
// Parse response JSON to check for function_calls array
```

### Computing Similarity with Embeddings
```c
#include <math.h>

float compute_cosine_similarity(cactus_model_t model,
                                const char* text1,
                                const char* text2) {
    float embeddings1[2048], embeddings2[2048];
    size_t dim1, dim2;

    cactus_embed(model, text1, embeddings1, sizeof(embeddings1), &dim1);
    cactus_embed(model, text2, embeddings2, sizeof(embeddings2), &dim2);

    float dot_product = 0.0f;
    float norm1 = 0.0f, norm2 = 0.0f;

    for (size_t i = 0; i < dim1; i++) {
        dot_product += embeddings1[i] * embeddings2[i];
        norm1 += embeddings1[i] * embeddings1[i];
        norm2 += embeddings2[i] * embeddings2[i];
    }

    return dot_product / (sqrtf(norm1) * sqrtf(norm2));
}

// Usage
float similarity = compute_cosine_similarity(embed_model,
    "The cat sat on the mat",
    "A feline rested on the rug");
printf("Similarity: %.4f\n", similarity);
```

### Audio Transcription with Whisper
```c
#include "cactus_ffi.h"
#include <stdio.h>

void transcription_callback(const char* token, uint32_t token_id, void* user_data) {
    printf("%s", token);
    fflush(stdout);
}

int main() {
    cactus_model_t whisper = cactus_init("path/to/whisper-base", 448, NULL);
    if (!whisper) return -1;

    char response[32768];

    // Transcribe with streaming output
    int result = cactus_transcribe(whisper, "meeting.wav", NULL,
                                    response, sizeof(response), NULL,
                                    transcription_callback, NULL);

    printf("\n\nFull response: %s\n", response);

    cactus_destroy(whisper);
    return 0;
}
```

### Multimodal Retrieval
```c
#include "cactus_ffi.h"
#include <math.h>

// Find most similar image to a text query
int find_similar_image(cactus_model_t model,
                       const char* query,
                       const char** image_paths,
                       int num_images) {
    float query_embed[1024];
    size_t query_dim;
    cactus_embed(model, query, query_embed, sizeof(query_embed), &query_dim);

    float best_score = -1.0f;
    int best_idx = -1;

    for (int i = 0; i < num_images; i++) {
        float img_embed[1024];
        size_t img_dim;
        cactus_image_embed(model, image_paths[i], img_embed, sizeof(img_embed), &img_dim);

        // Compute cosine similarity
        float dot = 0, norm_q = 0, norm_i = 0;
        for (size_t j = 0; j < query_dim; j++) {
            dot += query_embed[j] * img_embed[j];
            norm_q += query_embed[j] * query_embed[j];
            norm_i += img_embed[j] * img_embed[j];
        }
        float score = dot / (sqrtf(norm_q) * sqrtf(norm_i));

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return best_idx;
}
```

## Supported Model Types

| Model Type | Text | Vision | Audio | Embeddings | Description |
|------------|------|--------|-------|------------|-------------|
| Qwen | ✓ | - | - | ✓ | Qwen/Qwen2/Qwen3 language models |
| Gemma | ✓ | - | - | ✓ | Google Gemma models |
| LFM2 | ✓ | ✓ | - | ✓ | Liquid Foundation Models |
| Smol | ✓ | - | - | ✓ | SmolLM compact models |
| Nomic | - | - | - | ✓ | Nomic embedding models |
| Whisper | - | - | ✓ | ✓ | OpenAI Whisper transcription |
| Siglip2 | - | ✓ | - | ✓ | Vision encoder for embeddings |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CACTUS_KV_WINDOW_SIZE` | 512 | Sliding window size for KV cache |
| `CACTUS_KV_SINK_SIZE` | 4 | Number of attention sink tokens to preserve |

**Example:**
```bash
export CACTUS_KV_WINDOW_SIZE=1024
export CACTUS_KV_SINK_SIZE=8
./my_app
```

## Best Practices

1. **Always Check Return Values**: Functions return negative values on error
2. **Buffer Sizes**: Use large response buffers (8192+ bytes recommended)
3. **Memory Management**: Always call `cactus_destroy()` when done
4. **Thread Safety**: Each model instance should be used from a single thread
5. **Context Management**: Use `cactus_reset()` between unrelated conversations
6. **Streaming**: Implement callbacks for better user experience with long generations
7. **Reuse Models**: Initialize once, use multiple times for efficiency

## Error Handling

Most functions return:
- Positive values or 0 on success
- Negative values on error

Common error scenarios:
- Invalid model path
- Insufficient buffer size
- Malformed JSON input
- Unsupported operation for model type
- Out of memory

## Performance Tips

1. **Reuse Model Instances**: Initialize once, use multiple times
2. **Appropriate Context Size**: Use the minimum context size needed for your use case
3. **Streaming for UX**: Use callbacks for responsive user interfaces
4. **Early Stopping**: Use `cactus_stop()` to avoid unnecessary generation
5. **Batch Embeddings**: When possible, process multiple texts in sequence without resetting
6. **KV Cache Tuning**: Adjust `CACTUS_KV_WINDOW_SIZE` based on your context needs