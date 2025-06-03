#include "test_core_api.h"
#include "../cactus/cactus.h"
#include "../cactus/cactus_stt.h" // For STT tests
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cstring> // For strcmp

// Test hook from cactus_stt.cpp
extern const char* g_last_initial_prompt_for_test;

// Test basic model loading and initialization
void test_model_loading() {
    std::cout << "Testing model loading..." << std::endl;
    
    common_params params;
    params.model.path = "../llm.gguf";
    params.n_ctx = 1024;
    params.n_batch = 512;
    params.cpuparams.n_threads = 4;
    params.use_mmap = true;
    params.warmup = false;

    cactus::cactus_context ctx;
    bool success = ctx.loadModel(params);
    assert(success && "Model loading failed");
    std::cout << "Model loading test passed" << std::endl;
}

// Test basic text completion
void test_basic_completion() {
    std::cout << "Testing basic completion..." << std::endl;
    
    common_params params;
    params.model.path = "../llm.gguf";
    params.prompt = "Hello, how are you?";
    params.n_predict = 50;
    params.n_ctx = 1024;
    params.n_batch = 512;
    params.cpuparams.n_threads = 4;
    params.use_mmap = true;
    params.warmup = false;

    cactus::cactus_context ctx;
    assert(ctx.loadModel(params) && "Model loading failed");
    assert(ctx.initSampling() && "Sampling initialization failed");

    ctx.loadPrompt();
    ctx.beginCompletion();

    std::string response;
    const llama_vocab * vocab = llama_model_get_vocab(ctx.model);
    while (ctx.has_next_token) {
        auto tok = ctx.nextToken();
        if (tok.tok < 0) break;
        if (tok.tok == llama_vocab_eos(vocab)) break;

        char buffer[64];
        int length = llama_token_to_piece(vocab, tok.tok, buffer, sizeof(buffer), false, false);
        if (length > 0) {
            response += std::string(buffer, length);
        }
    }

    assert(!response.empty() && "Response should not be empty");
    std::cout << "Basic completion test passed" << std::endl;
}

// Test chat formatting
void test_chat_formatting() {
    std::cout << "Testing chat formatting..." << std::endl;
    
    common_params params;
    params.model.path = "../llm.gguf";
    params.n_ctx = 1024;
    params.n_batch = 512;
    params.cpuparams.n_threads = 4;
    params.use_mmap = true;
    params.warmup = false;

    cactus::cactus_context ctx;
    assert(ctx.loadModel(params) && "Model loading failed");

    // Test standard chat formatting
    std::string messages = R"([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ])";
    
    auto contains_local = [](const std::string& str, const std::string& substr) {
        return str.find(substr) != std::string::npos;
    };

    std::string formatted = ctx.getFormattedChat(messages, "");
    assert(!formatted.empty() && "Formatted chat should not be empty");
    assert(contains_local(formatted, "Hello") && "Formatted chat should contain the message");
    
    std::cout << "Chat formatting test passed" << std::endl;
}

// Test prompt truncation
void test_prompt_truncation() {
    std::cout << "Testing prompt truncation..." << std::endl;
    
    common_params params;
    params.model.path = "../llm.gguf";
    params.n_ctx = 64;
    params.n_batch = 512;
    params.cpuparams.n_threads = 4;
    params.use_mmap = true;
    params.warmup = false;

    std::string long_prompt = "This is a very long prompt that should be truncated because it exceeds the context size. ";
    for (int i = 0; i < 100; i++) {
        long_prompt += "This is additional text to make the prompt longer. ";
    }
    params.prompt = long_prompt; 
    std::cout << "Prompt length: " << long_prompt.length() << " characters" << std::endl;

    cactus::cactus_context ctx;
    assert(ctx.loadModel(params) && "Model loading failed"); 
    assert(ctx.initSampling() && "Sampling initialization failed");

    ctx.loadPrompt();
    
    std::cout << "Number of prompt tokens: " << ctx.num_prompt_tokens << std::endl;
    std::cout << "Context size: " << params.n_ctx << std::endl;

    assert(ctx.truncated && "Prompt should be truncated");

    std::cout << "Prompt truncation test passed" << std::endl;
}

// Test stopping criteria
void test_stopping_criteria() {
    std::cout << "Testing stopping criteria..." << std::endl;
    
    common_params params;
    params.model.path = "../llm.gguf";
    params.prompt = "Write a short story about a cat.";
    params.n_predict = 100;
    params.n_ctx = 1024;
    params.n_batch = 512;
    params.cpuparams.n_threads = 4;
    params.use_mmap = true;
    params.warmup = false;

    cactus::cactus_context ctx;
    assert(ctx.loadModel(params) && "Model loading failed");
    assert(ctx.initSampling() && "Sampling initialization failed");

    ctx.loadPrompt();
    ctx.beginCompletion();

    std::string response;
    const llama_vocab * vocab = llama_model_get_vocab(ctx.model);
    while (ctx.has_next_token) {
        auto tok = ctx.nextToken();
        if (tok.tok < 0) break;
        if (tok.tok == llama_vocab_eos(vocab)) {
            assert(ctx.stopped_eos && "Should stop on EOS token");
            break;
        }

        char buffer[64];
        int length = llama_token_to_piece(vocab, tok.tok, buffer, sizeof(buffer), false, false);
        if (length > 0) {
            response += std::string(buffer, length);
        }
    }

    assert(!response.empty() && "Response should not be empty");
    std::cout << "Stopping criteria test passed" << std::endl;
}

// Test embedding generation
void test_embedding_generation() {
    std::cout << "Testing embedding generation..." << std::endl;
    
    common_params params;
    params.model.path = "../llm.gguf";
    params.embedding = true; 
    params.n_ctx = 1024;
    params.n_batch = 512;
    params.cpuparams.n_threads = 4;
    params.use_mmap = true;
    params.warmup = false;
    params.prompt = "Generate embeddings for this text.";

    cactus::cactus_context ctx;
    assert(ctx.loadModel(params) && "Model loading failed");
    std::vector<float> embeddings = ctx.getEmbedding(params);
    assert(!embeddings.empty() && "Embeddings should not be empty");
    std::cout << "Embedding generation test passed" << std::endl;
}

// Test benchmarking function
void test_benchmarking() {
    std::cout << "Testing benchmarking..." << std::endl;
    
    common_params params;
    params.model.path = "../llm.gguf";
    params.n_ctx = 1024; 
    params.n_batch = 512;
    params.cpuparams.n_threads = 4;
    params.use_mmap = true;
    params.warmup = false; 

    cactus::cactus_context ctx;
    assert(ctx.loadModel(params) && "Model loading failed for benchmarking");

    int pp = 8; 
    int tg = 8; 
    int pl = 1; 
    int nr = 1;

    std::string bench_results = ctx.bench(pp, tg, pl, nr);
    assert(!bench_results.empty() && "Benchmarking results string should not be empty");
    
    std::cout << "Benchmarking results (JSON): " << bench_results << std::endl;
    std::cout << "Benchmarking test passed" << std::endl;
}

// Test Jinja chat formatting
void test_jinja_chat_formatting() {
    std::cout << "Testing Jinja chat formatting..." << std::endl;
    
    common_params params;
    params.model.path = "../llm.gguf";
    params.n_ctx = 1024;
    params.n_batch = 512;
    params.cpuparams.n_threads = 4;
    params.use_mmap = true;
    params.warmup = false;

    cactus::cactus_context ctx;
    assert(ctx.loadModel(params) && "Model loading failed for Jinja test");

    std::string messages_json = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"}
    ])";
    std::string empty_template = "";
    std::string empty_schema = "";
    std::string empty_tools = "";
    bool parallel_calls = false;
    std::string tool_choice = "";

    // Call the Jinja formatting function
    common_chat_params chat_result = ctx.getFormattedChatWithJinja(
        messages_json,
        empty_template, 
        empty_schema,
        empty_tools,
        parallel_calls,
        tool_choice
    );
    
    // Same as above regarding `contains` helper
    auto contains_local = [](const std::string& str, const std::string& substr) {
        return str.find(substr) != std::string::npos;
    };

    assert(!chat_result.prompt.empty() && "Formatted Jinja prompt should not be empty");
    assert(contains_local(chat_result.prompt, "helpful assistant") && "Formatted prompt should contain system message");
    assert(contains_local(chat_result.prompt, "Hello, world!") && "Formatted prompt should contain user message");

    std::cout << "Formatted Jinja Prompt: " << chat_result.prompt << std::endl; // Print for inspection
    std::cout << "Jinja chat formatting test passed" << std::endl;
}

// Test KV cache type string conversion
void test_kv_cache_type() {
    std::cout << "Testing KV cache type conversion..." << std::endl;

    // Test valid types
    lm_ggml_type f16_type = cactus::kv_cache_type_from_str("f16");
    assert(f16_type == LM_GGML_TYPE_F16 && "KV cache type 'f16' conversion failed");

    lm_ggml_type f32_type = cactus::kv_cache_type_from_str("f32");
    assert(f32_type == LM_GGML_TYPE_F32 && "KV cache type 'f32' conversion failed");

    // Test invalid type (should throw)
    bool caught_exception = false;
    try {
        cactus::kv_cache_type_from_str("invalid_type");
    } catch (const std::runtime_error& e) {
        caught_exception = true;
        std::cout << "Caught expected exception for invalid type: " << e.what() << std::endl;
    } catch (...) {
        assert(false && "Caught unexpected exception type for invalid KV cache type");
    }
    assert(caught_exception && "Expected std::runtime_error was not thrown for invalid KV cache type");

    std::cout << "KV cache type conversion test passed" << std::endl;
}

// --- STT Test Cases ---

void test_stt_set_user_vocabulary_stores_vocabulary() {
    std::cout << "Testing STT_SetUserVocabulary_StoresVocabulary..." << std::endl;
    cactus::STT stt;
    std::string vocab = "expected vocabulary, custom words";
    stt.setUserVocabulary(vocab);
    assert(stt.getUserVocabularyForTest() == vocab && "User vocabulary not stored correctly.");
    std::cout << "STT_SetUserVocabulary_StoresVocabulary test passed" << std::endl;
}

void test_stt_set_user_vocabulary_empty_clears_vocabulary() {
    std::cout << "Testing STT_SetUserVocabulary_EmptyClearsVocabulary..." << std::endl;
    cactus::STT stt;
    stt.setUserVocabulary("initial vocabulary"); // Set some initial
    stt.setUserVocabulary("");                   // Clear it
    assert(stt.getUserVocabularyForTest().empty() && "User vocabulary not cleared by empty string.");
    std::cout << "STT_SetUserVocabulary_EmptyClearsVocabulary test passed" << std::endl;
}

void test_stt_process_audio_uses_vocabulary() {
    std::cout << "Testing STT_ProcessAudio_UsesVocabularyAsInitialPrompt..." << std::endl;
    cactus::STT stt;
    // Initialize STT - assuming dummy_model.gguf allows initialization
    // We won't check transcription quality, just that the prompt is passed.
    bool initialized = stt.initialize("../dummy_model.gguf", "en", false); // use_gpu=false
    assert(initialized && "STT initialization failed for vocabulary test. Ensure dummy_model.gguf exists.");

    std::string vocab = "my custom prompt for STT";
    stt.setUserVocabulary(vocab);

    std::vector<float> dummy_samples = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f}; // Minimal audio
    stt.processAudio(dummy_samples); // This will trigger the test hook

    assert(g_last_initial_prompt_for_test != nullptr && "g_last_initial_prompt_for_test was not set.");
    if (g_last_initial_prompt_for_test) { // Defensive check
      assert(strcmp(g_last_initial_prompt_for_test, vocab.c_str()) == 0 && "User vocabulary was not used as initial_prompt.");
    }
    std::cout << "STT_ProcessAudio_UsesVocabularyAsInitialPrompt test passed" << std::endl;
}

void test_stt_process_audio_no_vocabulary_prompt_is_null() {
    std::cout << "Testing STT_ProcessAudio_NoVocabulary_InitialPromptIsNull..." << std::endl;
    cactus::STT stt;
    bool initialized = stt.initialize("../dummy_model.gguf", "en", false);
    assert(initialized && "STT initialization failed for no-vocabulary test. Ensure dummy_model.gguf exists.");

    stt.setUserVocabulary(""); // Ensure it's empty

    std::vector<float> dummy_samples = {0.0f, 0.1f, 0.2f};
    stt.processAudio(dummy_samples);

    assert(g_last_initial_prompt_for_test == nullptr && "initial_prompt was not null when no vocabulary was set.");
    std::cout << "STT_ProcessAudio_NoVocabulary_InitialPromptIsNull test passed" << std::endl;
}