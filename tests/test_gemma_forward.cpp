#include "test_utils.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <memory>

const char* g_gemma_model_path = "../../weights/gemma3-270m-i8";

bool test_gemma_forward() {
    std::unique_ptr<cactus::engine::Model> model = cactus::engine::create_model(g_gemma_model_path);
    if (!model) {
        std::cerr << "FAILED: Could not create model" << std::endl;
        return false;
    }
    
    if (!model->init(g_gemma_model_path, 2048)) {
        std::cerr << "FAILED: Could not initialize model" << std::endl;
        return false;
    }
    
    auto* tokenizer = model->get_tokenizer();
    if (!tokenizer) {
        std::cerr << "FAILED: Could not get tokenizer" << std::endl;
        return false;
    }
    
    const char* prompt = R"(<start_of_turn>user
You are a helpful assistant.

Count to 10<end_of_turn>
<start_of_turn>model
)";
    
    std::vector<uint32_t> token_ids_raw = tokenizer->encode(prompt);
    std::vector<uint32_t> token_ids;
    token_ids.push_back(tokenizer->get_bos_token());
    token_ids.insert(token_ids.end(), token_ids_raw.begin(), token_ids_raw.end());
    
    if (token_ids.empty()) {
        std::cerr << "FAILED: Tokenization produced no tokens" << std::endl;
        return false;
    }
    
    std::vector<uint32_t> expected_token_ids = {
        2, 105, 2364, 107, 3048, 659, 496, 11045, 16326,
        236761, 108, 4377, 531, 236743, 236770, 236771, 106, 107,
        105, 4368, 107
    };
    
    if (token_ids.size() != expected_token_ids.size()) {
        std::cerr << "FAILED: Token count mismatch - got " << token_ids.size() 
                  << ", expected " << expected_token_ids.size() << "\n";
        std::cerr << "token_ids: ";
        for (auto id : token_ids) std::cerr << id << " ";
        std::cerr << "\nexpected_token_ids: ";
        for (auto id : expected_token_ids) std::cerr << id << " ";
        std::cerr << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < token_ids.size(); ++i) {
        uint32_t actual = token_ids[i];
        uint32_t expected = expected_token_ids[i];
        
        if (actual != expected) {
            std::cerr << "FAILED: Token mismatch at position " << i 
                      << " - got " << actual << ", expected " << expected << std::endl;
            return false;
        }
    }
    
    uint32_t next_token = model->generate(token_ids, 0.0f, 1, 1);
    std::vector<uint32_t> generated_tokens = {next_token};
    if (next_token != 19058) {
        std::cerr << "FAILED: Expected token 19058, but got token " 
                  << next_token << " (decoded: \"" <<  tokenizer->decode({next_token}) << "\")" << std::endl;
        return false;
    }
    
    for (int i = 0; i < 20; ++i) {
        if (next_token == tokenizer->get_eos_token() || tokenizer->decode({next_token}) == "<end_of_turn>") {
            break;
        }
        next_token = model->generate({next_token}, 0.6f, 0.95f, 20);
        generated_tokens.push_back(next_token);
    }
    
    if (generated_tokens.empty()) {
        std::cerr << "FAILED: Generation produced no tokens" << std::endl;
        return false;
    }
    
    std::string generated_text = tokenizer->decode(generated_tokens);
    if (generated_text.empty()) {
        std::cerr << "FAILED: Decoded text is empty" << std::endl;
        return false;
    } else {
        std::cout << "Generated text: " << generated_text << std::endl;
    }
    
    return true;
}

int main() {
    TestUtils::TestRunner runner("Gemma Forward Pass Tests");
    runner.run_test("gemma_forward", test_gemma_forward());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
