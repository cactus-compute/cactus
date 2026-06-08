# Quickstart

Install Cactus and run your first on-device AI completion.

## Installation

=== "React Native"

    --8<-- "react-native/README.md:install"

    ### Platform Integration

    --8<-- "react-native/README.md:integration"

=== "Flutter"

    --8<-- "flutter/README.md:install"

    ### Platform Integration

    --8<-- "flutter/README.md:integration"

=== "Kotlin"

    --8<-- "kotlin/README.md:install"

    ### Platform Integration

    --8<-- "kotlin/README.md:integration"

=== "Swift"

    --8<-- "swift/README.md:install"

    ### Platform Integration

    --8<-- "swift/README.md:integration"

=== "Python"

    --8<-- "python/README.md:install"

=== "Rust"

    --8<-- "rust/README.md:install"

=== "CLI"

    **Homebrew (macOS):**

    ```bash
    brew install cactus-compute/cactus/cactus
    ```

    **From Source (macOS):**

    ```bash
    brew install cmake
    git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup && cactus build --python
    ```

    **From Source (Linux):**

    ```bash
    sudo apt-get install python3 python3-venv python3-pip cmake build-essential libcurl4-openssl-dev
    git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup && cactus build --python
    ```

=== "C++"

    Include the Cactus header in your project:

    ```cpp
    #include <cactus_engine.h>
    ```

    See the [Cactus repository](https://github.com/cactus-compute/cactus) for CMake build instructions.

---

## Your First Completion

=== "React Native"

    --8<-- "react-native/README.md:example"

=== "Flutter"

    --8<-- "flutter/README.md:example"

=== "Kotlin"

    --8<-- "kotlin/README.md:example"

=== "Swift"

    --8<-- "swift/README.md:example"

=== "Python"

    --8<-- "python/README.md:example"

=== "Rust"

    ```rust
    use std::ffi::CString;
    use std::os::raw::c_char;

    mod cactus;

    fn main() {
        unsafe {
            let model_path = CString::new("path/to/weight/folder").unwrap();
            let model = cactus::cactus_init(model_path.as_ptr(), std::ptr::null(), false);

            let messages = CString::new(
                r#"[{"role": "user", "content": "What is the capital of France?"}]"#
            ).unwrap();

            let mut response = vec![0u8; 4096];
            cactus::cactus_complete(
                model, messages.as_ptr(),
                response.as_mut_ptr() as *mut c_char, response.len(),
                std::ptr::null(), std::ptr::null(),
                None, std::ptr::null_mut(),
                std::ptr::null(), 0,
            );

            println!("{}", String::from_utf8_lossy(&response));
            cactus::cactus_destroy(model);
        }
    }
    ```

=== "CLI"

    ```bash
    cactus run <model|path>
    ```

=== "C++"

    ```cpp
    #include <cactus_engine.h>

    cactus_model_t model = cactus_init(
        "path/to/weight/folder",
        "path/to/rag/documents",
        false
    );

    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ])";

    char response[4096];
    int result = cactus_complete(
        model, messages, response, sizeof(response),
        nullptr, nullptr, nullptr, nullptr,
        nullptr, 0
    );
    ```

---

## Next Steps

- **[Engine API](cactus_engine.md)** -- Full inference API reference
- **[Graph API](cactus_graph.md)** -- Zero-copy computation graph for custom models
- **[Fine-tuning & Deployment](finetuning.md)** -- Convert and deploy custom fine-tunes
- **[Choose Your Binding](choose-bindings.md)** -- Help picking the right binding for your project
