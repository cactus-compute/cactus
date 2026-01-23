#include <iostream>
#include <vector>
#include <cstring>
#include "../cactus/ffi/cactus_ffi.h"

// Note: This test requires linking against libcactus and enabling the FFI.
// It assumes a compatible ARM environment.

int main() {
    std::cout << "Starting Raw Image Buffer Test..." << std::endl;

    // 1. Initialize model (Mock path, requires actual model files to conform)
    // For this test we assume valid model files at "models/LFM2VL"
    const char* model_path = "models/LFM2VL";
    cactus_model_t model = cactus_init(model_path, nullptr);
    if (!model) {
        std::cerr << "Failed to init model from " << model_path << std::endl;
        std::cerr << "Last Error: " << cactus_get_last_error() << std::endl;
        return 1;
    }

    std::cout << "Model initialized." << std::endl;

    // 2. Prepare dummy image buffer (Red 2x2 image)
    // 2x2 pixels, 3 channels (RGB)
    size_t width = 2;
    size_t height = 2;
    size_t channels = 3;
    std::vector<unsigned char> buffer = {
        255, 0, 0,  255, 0, 0,
        255, 0, 0,  255, 0, 0
    };

    cactus_image_t img;
    img.id = "buffer://test_image";
    img.data = buffer.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    std::cout << "Prepared image buffer: " << width << "x" << height << "x" << channels << std::endl;

    // 3. Prepare JSON Request
    const char* messages = R"([
        {"role": "user", "content": [{"type": "text", "text": "Describe this image"}, {"type": "image", "image": "buffer://test_image"}]}
    ])";
    
    // Options
    const char* options = R"({
        "temperature": 0.7,
        "max_tokens": 50
    })";

    // 4. Call complete with images
    char response[2048];
    std::cout << "Calling cactus_complete_with_images..." << std::endl;
    
    int result = cactus_complete_with_images(
        model,
        messages,
        &img,
        1,          // image_count
        response,
        2048,
        options,
        nullptr,    // tools
        nullptr,    // callback
        nullptr     // user_data
    );

    if (result < 0) {
        std::cerr << "Error during completion: " << response << std::endl;
        cactus_destroy(model);
        return 1;
    } else {
        std::cout << "Completion successful." << std::endl;
        std::cout << "Response: " << response << std::endl;
    }

    // 5. Cleanup
    cactus_destroy(model);
    std::cout << "Test Finished." << std::endl;
    return 0;
}
