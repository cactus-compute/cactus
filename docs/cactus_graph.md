# Cactus Graph API Documentation

The Cactus Graph API provides a computational graph framework for building and executing tensor operations. It supports multiple precision types, broadcasting, and optimized execution for neural network inference.

## Table of Contents
- [Setup](#setup)
- [Core Concepts](#core-concepts)
- [Getting Started](#getting-started)
- [Tensor Operations](#tensor-operations)
- [Advanced Features](#advanced-features)
- [Complete Examples](#complete-examples)
- [Best Practices](#best-practices)

## Setup

Before using the Cactus Graph API, set up your development environment:

```bash
# Setup the environment and install dependencies
./setup

# Build the Cactus library
cactus build

# Run tests to verify everything works
cactus test
```

## Core Concepts

### Precision Types
The framework supports three precision types for tensors:

```cpp
enum class Precision {
    INT4,
    INT8,
    FP16
};
```

**Note:** INT4 tensors use packed storage (2 values per byte) and automatically unpack to INT8 for computation.

### Graph Construction
The `CactusGraph` class manages the computational graph:

```cpp
CactusGraph graph;
size_t input = graph.input({2, 3}, Precision::INT8);
size_t result = graph.add(input, another_input);
graph.execute();
void* output = graph.get_output(result);
```

### Test Fixtures
For testing, use the provided fixtures that handle memory management:

```cpp
TestUtils::Int8TestFixture fixture("My Test");
TestUtils::FloatTestFixture fixture("Float Test");
```

## Getting Started

### Basic Example
```cpp
#include "cactus/graph/graph.h"

CactusGraph graph;
size_t a = graph.input({4}, Precision::INT8);
size_t b = graph.input({4}, Precision::INT8);
size_t sum = graph.add(a, b);

std::vector<int8_t> data_a = {1, 2, 3, 4};
std::vector<int8_t> data_b = {5, 6, 7, 8};
graph.set_input(a, data_a.data(), Precision::INT8);
graph.set_input(b, data_b.data(), Precision::INT8);

graph.execute();
int8_t* result = static_cast<int8_t*>(graph.get_output(sum)); // [6, 8, 10, 12]
```

## Tensor Operations

### Basic Arithmetic

#### Element-wise Operations
```cpp
size_t add_result = graph.add(a, b);           // a + b
size_t sub_result = graph.subtract(a, b);      // a - b
size_t mul_result = graph.multiply(a, b);      // a * b
size_t div_result = graph.divide(a, b);        // a / b
```

#### Scalar Operations
```cpp
size_t scalar_add = graph.scalar_add(input, 5.0f);        // input + 5
size_t scalar_sub = graph.scalar_subtract(input, 2.0f);   // input - 2
size_t scalar_mul = graph.scalar_multiply(input, 3.0f);   // input * 3
size_t scalar_div = graph.scalar_divide(input, 2.0f);     // input / 2
```

#### Mathematical Functions
```cpp
size_t exp_result = graph.scalar_exp(input);    // e^input
size_t sqrt_result = graph.scalar_sqrt(input);  // √input
size_t cos_result = graph.scalar_cos(input);    // cos(input)
size_t sin_result = graph.scalar_sin(input);    // sin(input)
size_t log_result = graph.scalar_log(input);    // ln(input)
```

### Matrix Operations

#### Matrix Multiplication
```cpp
// Standard matmul: (2,3) x (3,4) = (2,4)
size_t a = graph.input({2, 3}, Precision::FP16);
size_t b = graph.input({3, 4}, Precision::FP16);
size_t result = graph.matmul(a, b);

// With pre-transposed right-hand side
size_t result = graph.matmul(a, b, true);
```

#### Transpose
```cpp
size_t transposed = graph.transpose(input); // (2,3) -> (3,2)
```

#### Reshape
```cpp
size_t reshaped = graph.reshape(input, {6, 1}); // (2,3) -> (6,1)
```

### Reduction Operations

```cpp
size_t sum_all = graph.sum(input, -1);   // -1 for all elements
size_t sum_axis0 = graph.sum(input, 0);
size_t mean_all = graph.mean(input, -1);
size_t var = graph.variance(input, axis);
size_t min_val = graph.min(input, axis);
size_t max_val = graph.max(input, axis);
```

### Neural Network Operations

#### Layer Normalization
```cpp
size_t weight = graph.input({hidden_size}, Precision::FP16);
size_t bias = graph.input({hidden_size}, Precision::FP16);
size_t normalized = graph.layernorm(input, weight, bias, 1e-5f);
```

#### RMS Normalization
```cpp
size_t weight = graph.input({hidden_size}, Precision::FP16);
size_t normalized = graph.rms_norm(input, weight, 1e-5f);
```

#### Softmax
```cpp
size_t softmax_result = graph.softmax(input, -1);
```

#### Attention Mechanism
```cpp
size_t attention_out = graph.attention(query, key, value, scale);
size_t attention_out = graph.attention(query, key, value, scale, position_offset);
size_t attention_out = graph.attention(query, key, value, scale, position_offset, window_size);
```

#### Rotary Position Embedding (RoPE)
```cpp
size_t rope_output = graph.rope(input, theta, position_offset);
```

#### Activation Functions
```cpp
size_t silu_out = graph.silu(input);
size_t gelu_out = graph.gelu(input);
```

### Indexing and Gathering

#### Gather Operation
```cpp
size_t embeddings = graph.input({vocab_size, embed_dim}, Precision::FP16);
size_t indices = graph.input({batch_size, seq_len}, Precision::INT8);
size_t gathered = graph.gather(embeddings, indices);
```

#### Embedding Lookup
```cpp
size_t embedded = graph.embedding(embedding_tensor, indices);
size_t embedded = graph.embedding("embeddings.bin", indices); // memory-mapped
```

#### Memory-Mapped Weights
```cpp
size_t mmap_embed = graph.mmap_embeddings("embeddings.bin");
size_t weights = graph.mmap_weights("model_weights.bin");
```

### Advanced Operations

#### Concatenation
```cpp
size_t concatenated = graph.concat(tensor1, tensor2, axis);
```

#### Slicing
```cpp
size_t sliced = graph.slice(input, axis, start, length);
```

#### Indexing
```cpp
size_t indexed = graph.index(input, index_value, dimension);
```

#### Top-K Selection
```cpp
size_t topk_values = graph.topk(input, k);
```

#### Sampling
```cpp
size_t sampled = graph.sample(logits, temperature, top_p, top_k);
```

## Advanced Features

### Broadcasting
The framework automatically handles broadcasting for compatible shapes:

```cpp
size_t tensor = graph.input({2, 3}, Precision::INT8);
size_t scalar = graph.input({1}, Precision::INT8);
size_t result = graph.add(tensor, scalar);  // {1} -> {2,3}

size_t a = graph.input({2, 3}, Precision::INT8);
size_t b = graph.input({2, 1}, Precision::INT8);
size_t result = graph.add(a, b);  // {2,1} -> {2,3}

size_t a = graph.input({2, 2, 3}, Precision::INT8);
size_t b = graph.input({2, 3}, Precision::INT8);
size_t result = graph.add(a, b);  // {2,3} -> {2,2,3}
```

### Precision Conversion
```cpp
size_t int8_tensor = graph.input({4}, Precision::INT8);
size_t fp16_tensor = graph.precision_cast(int8_tensor, Precision::FP16);
graph.set_quantization_scale(node_id, scale);
```

### Graph Persistence

#### Saving Nodes
```cpp
GraphFile::save_node(graph, node_id, "output.bin");
```

#### Loading Nodes
```cpp
CactusGraph new_graph;
auto loaded = GraphFile::load_into_graph(new_graph, "output.bin");
size_t node_id = loaded.node_id;
std::vector<size_t> shape = loaded.shape;
Precision precision = loaded.precision;
```

### Graph Management

#### Execution
```cpp
graph.execute();
graph.execute("profile_output.json"); // with profiling
```

#### Reset Operations
```cpp
graph.hard_reset(); // clear all nodes and buffers
graph.soft_reset(); // clear only buffers, keep graph structure
```

## Complete Examples

### Building a Simple Neural Network Layer
```cpp
CactusGraph graph;

size_t input = graph.input({2, 4}, Precision::FP16);
size_t weight = graph.input({4, 8}, Precision::FP16);
size_t bias = graph.input({8}, Precision::FP16);

size_t linear = graph.matmul(input, weight);
size_t with_bias = graph.add(linear, bias);
size_t activated = graph.gelu(with_bias);

size_t ln_weight = graph.input({8}, Precision::FP16);
size_t ln_bias = graph.input({8}, Precision::FP16);
size_t output = graph.layernorm(activated, ln_weight, ln_bias);
```

### Implementing Multi-Head Attention
```cpp
CactusGraph graph;

size_t hidden_dim = 512;
size_t num_heads = 8;
size_t head_dim = hidden_dim / num_heads;
size_t seq_len = 32;

size_t input = graph.input({1, seq_len, hidden_dim}, Precision::FP16);
size_t q_weight = graph.input({hidden_dim, hidden_dim}, Precision::FP16);
size_t k_weight = graph.input({hidden_dim, hidden_dim}, Precision::FP16);
size_t v_weight = graph.input({hidden_dim, hidden_dim}, Precision::FP16);

size_t query = graph.matmul(input, q_weight);
size_t key = graph.matmul(input, k_weight);
size_t value = graph.matmul(input, v_weight);

query = graph.reshape(query, {1, seq_len, num_heads, head_dim});
key = graph.reshape(key, {1, seq_len, num_heads, head_dim});
value = graph.reshape(value, {1, seq_len, num_heads, head_dim});

float scale = 1.0f / sqrt(head_dim);
size_t attention_out = graph.attention(query, key, value, scale);
```

### Working with Embeddings
```cpp
CactusGraph graph;

size_t vocab_size = 50000;
size_t embed_dim = 768;
size_t tokens = graph.input({2, 10}, Precision::INT8);

size_t embed_table = graph.input({vocab_size, embed_dim}, Precision::FP16);
size_t embeddings = graph.gather(embed_table, tokens);

// or memory-mapped for large models
size_t mmap_table = graph.mmap_embeddings("vocab_embeddings.bin");
size_t embeddings = graph.gather(mmap_table, tokens);

size_t pos_embed = graph.input({1, 10, embed_dim}, Precision::FP16);
size_t final_embed = graph.add(embeddings, pos_embed);
```

### Similarity Computation
```cpp
TestUtils::FloatTestFixture fixture("Similarity");

size_t text1 = fixture.create_input({1, 768}, Precision::FP16);
size_t text2 = fixture.create_input({1, 768}, Precision::FP16);

// L2 norms
size_t norm1 = fixture.graph().scalar_sqrt(
    fixture.graph().sum(fixture.graph().multiply(text1, text1), -1));
size_t norm2 = fixture.graph().scalar_sqrt(
    fixture.graph().sum(fixture.graph().multiply(text2, text2), -1));

// cosine similarity = dot(a,b) / (norm(a) * norm(b))
size_t dot_product = fixture.graph().sum(fixture.graph().multiply(text1, text2), -1);
size_t similarity = fixture.graph().divide(dot_product, fixture.graph().multiply(norm1, norm2));
```

## Best Practices

### Memory Management
1. **Use appropriate precision**: INT4/INT8 for memory efficiency, FP16 for accuracy
2. **Memory-map large tensors**: Use `mmap_embeddings()` for vocabulary tables
3. **Reset graphs**: Call `hard_reset()` when switching between different models
4. **External buffers**: Use `set_external_input()` to avoid copying large inputs

### Performance Optimization
1. **Batch operations**: Process multiple samples together
2. **Pre-transpose weights**: Use `pretransposed_rhs=true` for matmul when possible
3. **Fused operations**: The framework automatically fuses compatible operations
4. **Backend selection**: Use NPU backend for supported operations:
   ```cpp
   size_t result = graph.matmul(a, b, false, ComputeBackend::NPU);
   ```

### Graph Construction
1. **Build once, execute many**: Construct the graph once, run with different inputs
2. **Validate shapes**: Ensure tensor shapes are compatible before operations
3. **Handle broadcasts**: Be aware of automatic broadcasting rules
4. **Profile execution**: Use `execute("profile.json")` to identify bottlenecks

### Testing
1. **Use test fixtures**: Leverage provided fixtures for automatic cleanup
2. **Verify outputs**: Use `verify_output()` methods for tolerance-based comparison
3. **Test edge cases**: Include tests for broadcasting, empty tensors, and large inputs
4. **Check precision**: Test operations with different precision types

### Error Handling
```cpp
try {
    CactusGraph graph;
    // ... build and execute graph
} catch (const std::exception& e) {
    std::cerr << "Graph error: " << e.what() << std::endl;
}
```

## Common Patterns

### Sequential Processing
```cpp
CactusGraph graph;
size_t x = graph.input({batch, dim}, Precision::FP16);
x = graph.linear(x, weight1, bias1);
x = graph.gelu(x);
x = graph.layernorm(x, ln_weight1, ln_bias1);
x = graph.linear(x, weight2, bias2);
```

### Residual Connections
```cpp
size_t input = graph.input({batch, dim}, Precision::FP16);
size_t processed = graph.matmul(input, weight);
processed = graph.gelu(processed);
size_t output = graph.add(input, processed);
```

### Multi-Path Processing
```cpp
size_t input = graph.input({batch, dim}, Precision::FP16);
size_t path1 = graph.matmul(input, weight1);
path1 = graph.silu(path1);
size_t path2 = graph.matmul(input, weight2);
size_t output = graph.multiply(path1, path2);
```