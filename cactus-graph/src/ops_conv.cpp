#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include <cstring>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <string>

std::vector<__fp16> dequantize_int8_weights_to_fp16(
    const BufferDesc& W,
    size_t rows,
    size_t cols,
    const char* op_name) {
    const int8_t* W_int8 = W.data_as<int8_t>();
    std::vector<__fp16> W_fp16(rows * cols);

    if (W.group_size == 0 || W.activation_scales_data == nullptr) {
        for (size_t i = 0; i < W_fp16.size(); ++i) {
            W_fp16[i] = static_cast<__fp16>(W_int8[i]);
        }
        return W_fp16;
    }

    const size_t group_size = W.group_size;
    if ((cols % group_size) != 0) {
        throw std::runtime_error(std::string(op_name) + " grouped INT8 weight columns must be divisible by group size");
    }

    const __fp16* scales = reinterpret_cast<const __fp16*>(W.activation_scales_data);
    const size_t num_groups = cols / group_size;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            const size_t idx = row * cols + col;
            const size_t group_idx = col / group_size;
            const float scale = static_cast<float>(scales[row * num_groups + group_idx]);
            W_fp16[idx] = static_cast<__fp16>(W_int8[idx] * scale);
        }
    }
    return W_fp16;
}

void compute_conv1d_causal_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU causal convolution operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    auto& Y = node.output_buffer;

    if (X.shape.size() != 3) {
        throw std::runtime_error("Causal conv requires 3D input [batch, seq_len, in_channels]");
    }
    if (W.shape.size() != 3) {
        throw std::runtime_error("Weight must be 3D");
    }

    const size_t N     = X.shape[0];
    const size_t L     = X.shape[1];
    const size_t C_in  = X.shape[2];
    const size_t W0    = W.shape[0];
    const size_t W1    = W.shape[1];
    const size_t W2    = W.shape[2];
    const size_t dil   = node.params.dilation;
    if (dil < 1) throw std::runtime_error("dilation must be >= 1");

    size_t M = 1;
    size_t C_out = 0;
    const bool standard_layout = (W1 == 1);
    const bool transposed_layout = (W2 == 1);
    if ((!standard_layout && !transposed_layout) || (W0 % C_in != 0)) {
        throw std::runtime_error("Only depthwise causal convolution is supported currently");
    }
    const size_t K = standard_layout ? W2 : W1;
    M = W0 / C_in;
    C_out = C_in * M;

    Y.shape = { N, L, C_out };
    Y.precision = X.precision;

    auto transpose_depthwise_weights_fp16 = [&](const __fp16* src) {
        std::vector<__fp16> transposed(W0 * K);
        for (size_t oc = 0; oc < W0; ++oc) {
            for (size_t k = 0; k < K; ++k) {
                transposed[oc * K + k] = src[(oc * W1 + k) * W2];
            }
        }
        return transposed;
    };

    if (W.precision == Precision::INT8) {
        auto W_fp16 = dequantize_int8_weights_to_fp16(W, W0, K, "conv1d_causal");
        if (transposed_layout && !standard_layout) {
            auto fixed = transpose_depthwise_weights_fp16(W_fp16.data());
            cactus_conv1d_causal_depthwise_f16(
                X.data_as<__fp16>(), fixed.data(), Y.data_as<__fp16>(),
                N, L, C_in, K, dil);
        } else {
            cactus_conv1d_causal_depthwise_f16(
                X.data_as<__fp16>(), W_fp16.data(), Y.data_as<__fp16>(),
                N, L, C_in, K, dil);
        }
    } else if (W.precision == Precision::FP16) {
        if (transposed_layout && !standard_layout) {
            auto fixed = transpose_depthwise_weights_fp16(W.data_as<__fp16>());
            cactus_conv1d_causal_depthwise_f16(
                X.data_as<__fp16>(), fixed.data(), Y.data_as<__fp16>(),
                N, L, C_in, K, dil);
        } else {
            cactus_conv1d_causal_depthwise_f16(
                X.data_as<__fp16>(), W.data_as<__fp16>(), Y.data_as<__fp16>(),
                N, L, C_in, K, dil);
        }
    } else {
        throw std::runtime_error("Conv requires FP16 weights");
    }
}

void compute_conv1d_k3_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU causal convolution operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    auto& Y = node.output_buffer;

    if (X.shape.size() != 3)
        throw std::runtime_error("Conv requires 3D input [N, C_in, L]!");

    if (W.shape.size() != 3)
        throw std::runtime_error("Weight must be [C_out, C_in, 3]!");

    const size_t N    = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t L    = X.shape[2];

    const size_t C_out = W.shape[0];
    const size_t K     = W.shape[2];
    const size_t stride = node.params.stride;

    if (K != 3)
        throw std::runtime_error("Conv1d_k3 only supports K=3!");

    size_t L_out = ((L - 1) / stride) + 1;
    Y.shape     = { N, C_out, L_out };
    Y.precision = X.precision;

    if (X.precision != Precision::FP16) {
        throw std::runtime_error("Conv1d_k3 only supports FP16 activations");
    }

    if (W.precision == Precision::INT8) {
        auto W_fp16 = dequantize_int8_weights_to_fp16(W, C_out, C_in * K, "conv1d_k3");
        cactus_conv1d_f16_k3(
            X.data_as<__fp16>(),
            W_fp16.data(),
            Y.data_as<__fp16>(),
            N, L, C_in, C_out, stride
        );
    } else if (W.precision == Precision::FP16) {
        cactus_conv1d_f16_k3(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            Y.data_as<__fp16>(),
            N, L, C_in, C_out, stride
        );
    } else {
        throw std::runtime_error("Conv requires FP16 weights");
    }
}

void compute_conv1d_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                         const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }

    auto& Y = node.output_buffer;

    if (X.shape.size() != 3) {
        throw std::runtime_error("conv1d expects input [N, C_in, L]");
    }
    if (W.shape.size() != 3) {
        throw std::runtime_error("conv1d weight must be [C_out, C_in, K]");
    }

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t L = X.shape[2];
    const size_t C_out = W.shape[0];
    const size_t K = W.shape[2];
    const size_t stride = node.params.stride;

    if (W.shape[1] != C_in) {
        throw std::runtime_error("conv1d weight C_in mismatch");
    }

    if (X.precision != Precision::FP16) {
        throw std::runtime_error("Conv1d only supports FP16 activations");
    }

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->total_size != C_out) {
            throw std::runtime_error("conv1d bias size mismatch");
        }
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C_out);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C_out);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv1d bias only supports FP16/FP32");
        }
    }

    std::vector<__fp16> W_fp16;
    const __fp16* W_ptr = nullptr;
    if (W.precision == Precision::FP16) {
        W_ptr = W.data_as<__fp16>();
    } else if (W.precision == Precision::INT8) {
        W_fp16 = dequantize_int8_weights_to_fp16(W, C_out, C_in * K, "conv1d");
        W_ptr = W_fp16.data();
    } else {
        throw std::runtime_error("Conv1d only supports FP16/INT8 weights");
    }

    cactus_conv1d_f16(X.data_as<__fp16>(), W_ptr, bias_ptr,
                      Y.data_as<__fp16>(), N, L, C_in, C_out, K, stride);
}

void compute_conv1d_same_depthwise_k9_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                           const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU conv1d_same_depthwise_k9 operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }
    auto& Y = node.output_buffer;

    if (X.shape.size() != 3) {
        throw std::runtime_error("conv1d_same_depthwise_k9 expects input [N, L, C]");
    }
    if (X.precision != Precision::FP16) {
        throw std::runtime_error("conv1d_same_depthwise_k9 only supports FP16 activations");
    }

    const size_t N = X.shape[0];
    const size_t L = X.shape[1];
    const size_t C = X.shape[2];
    const size_t K = 9;

    if (W.shape.size() == 2) {
        if (W.shape[0] != C || W.shape[1] != K) {
            throw std::runtime_error("conv1d_same_depthwise_k9 weight must be [C, 9]");
        }
    } else if (W.shape.size() == 3) {
        if (W.shape[0] != C || W.shape[1] != 1 || W.shape[2] != K) {
            throw std::runtime_error("conv1d_same_depthwise_k9 weight must be [C, 1, 9]");
        }
    } else {
        throw std::runtime_error("conv1d_same_depthwise_k9 weight must be rank 2 or 3");
    }

    Y.shape = {N, L, C};
    Y.precision = Precision::FP16;

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->total_size != C) {
            throw std::runtime_error("conv1d_same_depthwise_k9 bias size mismatch");
        }
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv1d_same_depthwise_k9 bias only supports FP16/FP32");
        }
    }

    if (W.precision == Precision::FP16) {
        cactus_conv1d_same_depthwise_f16_k9(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, L, C
        );
        return;
    }

    if (W.precision == Precision::INT8) {
        auto W_fp16 = dequantize_int8_weights_to_fp16(W, C, K, "conv1d_same_depthwise_k9");
        cactus_conv1d_same_depthwise_f16_k9(
            X.data_as<__fp16>(),
            W_fp16.data(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, L, C
        );
        return;
    }

    throw std::runtime_error("Conv requires FP16/INT8 weights");
}

void compute_conv2d_k3s2p1_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU conv2d_k3s2p1 operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }
    auto& Y = node.output_buffer;

    if (X.shape.size() != 4) {
        throw std::runtime_error("conv2d_k3s2p1 expects input [N, C_in, H, W]");
    }
    if (X.precision != Precision::FP16) {
        throw std::runtime_error("conv2d_k3s2p1 only supports FP16 activations");
    }

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t H = X.shape[2];
    const size_t W_in = X.shape[3];
    const size_t C_out = Y.shape[1];

    if (H == 0 || W_in == 0) {
        throw std::runtime_error("conv2d_k3s2p1 input spatial dimensions must be > 0");
    }

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C_out);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C_out);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv2d_k3s2p1 bias only supports FP16/FP32");
        }
    }

    if (W.precision == Precision::FP16) {
        if (W.shape.size() != 4) {
            throw std::runtime_error("conv2d_k3s2p1 FP16 weight must be [C_out, C_in, 3, 3]");
        }
        cactus_conv2d_f16_k3s2p1_nchw(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, C_in, H, W_in, C_out
        );
        return;
    }

    if (W.precision == Precision::INT8) {
        if (W.shape.size() != 4) {
            throw std::runtime_error("conv2d_k3s2p1 INT8 weight must be [C_out, C_in, 3, 3]");
        }
        auto W_fp16 = dequantize_int8_weights_to_fp16(W, C_out, C_in * 3 * 3, "conv2d_k3s2p1");
        cactus_conv2d_f16_k3s2p1_nchw(
            X.data_as<__fp16>(),
            W_fp16.data(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, C_in, H, W_in, C_out
        );
        return;
    }

    throw std::runtime_error("Conv requires FP16/INT8 weights");
}

void compute_conv2d_depthwise_k3s2p1_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                          const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU conv2d_depthwise_k3s2p1 operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }
    auto& Y = node.output_buffer;

    if (X.shape.size() != 4) {
        throw std::runtime_error("conv2d_depthwise_k3s2p1 expects input [N, C, H, W]");
    }
    if (X.precision != Precision::FP16) {
        throw std::runtime_error("conv2d_depthwise_k3s2p1 only supports FP16 activations");
    }

    const size_t N = X.shape[0];
    const size_t C = X.shape[1];
    const size_t H = X.shape[2];
    const size_t W_in = X.shape[3];
    if (H == 0 || W_in == 0) {
        throw std::runtime_error("conv2d_depthwise_k3s2p1 input spatial dimensions must be > 0");
    }

    if (W.shape.size() == 3) {
        if (W.shape[0] != C || W.shape[1] != 3 || W.shape[2] != 3) {
            throw std::runtime_error("conv2d_depthwise_k3s2p1 weight must be [C, 3, 3]");
        }
    } else if (W.shape.size() == 4) {
        if (W.shape[0] != C || W.shape[1] != 1 || W.shape[2] != 3 || W.shape[3] != 3) {
            throw std::runtime_error("conv2d_depthwise_k3s2p1 weight must be [C, 1, 3, 3]");
        }
    } else {
        throw std::runtime_error("conv2d_depthwise_k3s2p1 weight must be rank 3 or 4");
    }

    const size_t H_out = (H - 1) / 2 + 1;
    const size_t W_out = (W_in - 1) / 2 + 1;
    Y.shape = {N, C, H_out, W_out};
    Y.precision = Precision::FP16;

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->total_size != C) {
            throw std::runtime_error("conv2d_depthwise_k3s2p1 bias size mismatch");
        }
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv2d_depthwise_k3s2p1 bias only supports FP16/FP32");
        }
    }

    if (W.precision == Precision::FP16) {
        cactus_conv2d_depthwise_f16_k3s2p1_nchw(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, C, H, W_in
        );
        return;
    }

    if (W.precision == Precision::INT8) {
        auto W_fp16 = dequantize_int8_weights_to_fp16(W, C, 3 * 3, "conv2d_depthwise_k3s2p1");
        cactus_conv2d_depthwise_f16_k3s2p1_nchw(
            X.data_as<__fp16>(),
            W_fp16.data(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, C, H, W_in
        );
        return;
    }

    throw std::runtime_error("Conv requires FP16/INT8 weights");
}

void compute_conv2d_pointwise_1x1_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                       const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU conv2d_pointwise_1x1 operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }
    auto& Y = node.output_buffer;

    if (X.shape.size() != 4) {
        throw std::runtime_error("conv2d_pointwise_1x1 expects input [N, C_in, H, W]");
    }
    if (X.precision != Precision::FP16) {
        throw std::runtime_error("conv2d_pointwise_1x1 only supports FP16 activations");
    }

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t H = X.shape[2];
    const size_t W_in = X.shape[3];
    if (H == 0 || W_in == 0) {
        throw std::runtime_error("conv2d_pointwise_1x1 input spatial dimensions must be > 0");
    }

    size_t C_out = 0;
    if (W.shape.size() == 2) {
        C_out = W.shape[0];
        if (W.shape[1] != C_in) {
            throw std::runtime_error("conv2d_pointwise_1x1 weight must be [C_out, C_in]");
        }
    } else if (W.shape.size() == 4) {
        C_out = W.shape[0];
        if (W.shape[1] != C_in || W.shape[2] != 1 || W.shape[3] != 1) {
            throw std::runtime_error("conv2d_pointwise_1x1 weight must be [C_out, C_in, 1, 1]");
        }
    } else {
        throw std::runtime_error("conv2d_pointwise_1x1 weight must be rank 2 or 4");
    }

    Y.shape = {N, C_out, H, W_in};
    Y.precision = Precision::FP16;

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->total_size != C_out) {
            throw std::runtime_error("conv2d_pointwise_1x1 bias size mismatch");
        }
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C_out);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C_out);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv2d_pointwise_1x1 bias only supports FP16/FP32");
        }
    }

    if (W.precision == Precision::FP16) {
        cactus_conv2d_pointwise_f16_1x1_nchw_gemm(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, C_in, H, W_in, C_out
        );
        return;
    }

    if (W.precision == Precision::INT8) {
        auto W_fp16 = dequantize_int8_weights_to_fp16(W, C_out, C_in, "conv2d_pointwise_1x1");
        cactus_conv2d_pointwise_f16_1x1_nchw_gemm(
            X.data_as<__fp16>(),
            W_fp16.data(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, C_in, H, W_in, C_out
        );
        return;
    }

    throw std::runtime_error("Conv requires FP16/INT8 weights");
}

void compute_conv1d_pointwise_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                   const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU conv1d_pointwise operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }
    auto& Y = node.output_buffer;

    if (X.shape.size() != 3) {
        throw std::runtime_error("conv1d_pointwise expects input [N, L, C_in]");
    }
    if (X.precision != Precision::FP16) {
        throw std::runtime_error("conv1d_pointwise only supports FP16 activations");
    }

    const size_t N = X.shape[0];
    const size_t L = X.shape[1];
    const size_t C_in = X.shape[2];

    size_t C_out = 0;
    if (W.shape.size() == 2) {
        C_out = W.shape[0];
        if (W.shape[1] != C_in) {
            throw std::runtime_error("conv1d_pointwise weight must be [C_out, C_in]");
        }
    } else if (W.shape.size() == 3) {
        C_out = W.shape[0];
        if (W.shape[1] != C_in || W.shape[2] != 1) {
            throw std::runtime_error("conv1d_pointwise weight must be [C_out, C_in, 1]");
        }
    } else {
        throw std::runtime_error("conv1d_pointwise weight must be rank 2 or 3");
    }

    Y.shape = {N, L, C_out};
    Y.precision = Precision::FP16;

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->total_size != C_out) {
            throw std::runtime_error("conv1d_pointwise bias size mismatch");
        }
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C_out);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C_out);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv1d_pointwise bias only supports FP16/FP32");
        }
    }

    if (W.precision == Precision::FP16) {
        cactus_conv1d_pointwise_f16_gemm(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, L, C_in, C_out
        );
        return;
    }

    if (W.precision == Precision::INT8) {
        auto W_fp16 = dequantize_int8_weights_to_fp16(W, C_out, C_in, "conv1d_pointwise");
        cactus_conv1d_pointwise_f16_gemm(
            X.data_as<__fp16>(),
            W_fp16.data(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, L, C_in, C_out
        );
        return;
    }

    throw std::runtime_error("Conv requires FP16/INT8 weights");
}

void compute_batchnorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                            const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.input_ids.size() != 5) {
        throw std::runtime_error("BatchNorm expects 5 inputs: input, weight, bias, running_mean, running_var");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const auto& B = get_input(node, 2, nodes, node_index_map);
    const auto& RM = get_input(node, 3, nodes, node_index_map);
    const auto& RV = get_input(node, 4, nodes, node_index_map);
    auto& Y = node.output_buffer;

    if (X.shape.empty()) {
        throw std::runtime_error("BatchNorm expects non-scalar input");
    }

    int axis = node.params.axis;
    if (axis < 0) axis += static_cast<int>(X.shape.size());
    if (axis < 0 || static_cast<size_t>(axis) >= X.shape.size()) {
        throw std::runtime_error("BatchNorm axis out of range");
    }

    const size_t C = X.shape[static_cast<size_t>(axis)];
    if (W.total_size != C || B.total_size != C || RM.total_size != C || RV.total_size != C) {
        throw std::runtime_error("BatchNorm parameter size mismatch");
    }

    auto load_1d_float = [C](const BufferDesc& buf, const char* name) -> std::vector<float> {
        if (buf.total_size != C) {
            throw std::runtime_error(std::string("BatchNorm parameter size mismatch for ") + name);
        }
        std::vector<float> out(C);
        if (buf.precision == Precision::FP16) {
            const __fp16* p = buf.data_as<__fp16>();
            for (size_t i = 0; i < C; ++i) out[i] = static_cast<float>(p[i]);
        } else if (buf.precision == Precision::FP32) {
            std::memcpy(out.data(), buf.data_as<float>(), C * sizeof(float));
        } else {
            throw std::runtime_error(std::string("BatchNorm parameter ") + name + " must be FP16 or FP32");
        }
        return out;
    };

    const std::vector<float> gamma = load_1d_float(W, "weight");
    const std::vector<float> beta = load_1d_float(B, "bias");
    const std::vector<float> mean = load_1d_float(RM, "running_mean");
    const std::vector<float> var = load_1d_float(RV, "running_var");

    size_t outer = 1;
    for (int i = 0; i < axis; ++i) {
        outer *= X.shape[static_cast<size_t>(i)];
    }
    size_t inner = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < X.shape.size(); ++i) {
        inner *= X.shape[i];
    }

    Y.shape = X.shape;
    Y.precision = X.precision;

    if (X.precision == Precision::FP16) {
        cactus_batchnorm_f16(
            X.data_as<__fp16>(),
            gamma.data(),
            beta.data(),
            mean.data(),
            var.data(),
            Y.data_as<__fp16>(),
            outer,
            C,
            inner,
            node.params.epsilon
        );
        return;
    }

    if (X.precision == Precision::FP32) {
        cactus_batchnorm_f32(
            X.data_as<float>(),
            gamma.data(),
            beta.data(),
            mean.data(),
            var.data(),
            Y.data_as<float>(),
            outer,
            C,
            inner,
            node.params.epsilon
        );
        return;
    }

    throw std::runtime_error("BatchNorm only supports FP16/FP32 activations");
}

void compute_stft_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                 const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    auto& Y = node.output_buffer;

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t L = X.shape[2];
    const size_t C_out = W.shape[0];
    const size_t K = W.shape[2];
    const size_t stride = node.params.stride;
    const size_t num_fft_bins = node.params.num_fft_bins;

    if (X.precision != Precision::FP16 || W.precision != Precision::FP16) {
        throw std::runtime_error("stft only supports FP16");
    }

    cactus_stft_f16(X.data_as<__fp16>(), W.data_as<__fp16>(),
                            Y.data_as<__fp16>(), N, L, C_in, C_out, K, stride, num_fft_bins);
}

void compute_conv1d_k7s3_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                         const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }

    auto& Y = node.output_buffer;

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t L = X.shape[2];
    
    if (W.shape.size() != 3) throw std::runtime_error("Weight must be 3D");
    const size_t C_in_W = W.shape[0];
    const size_t K = W.shape[1];
    const size_t C_out = W.shape[2];
    const size_t stride = node.params.stride;

    if (C_in != C_in_W) throw std::runtime_error("Channel mismatch in conv1d_k7s3");
    if (K != 7 || stride != 3) throw std::runtime_error("conv1d_k7s3 requires K=7, stride=3");

    if (X.precision != Precision::FP16 || W.precision != Precision::FP16) {
        throw std::runtime_error("Conv1d specialized only supports FP16");
    }
    
    size_t L_out = (L < 7) ? 0 : (L - 7) / 3 + 1;
    Y.shape = {N, C_out, L_out};
    Y.precision = Precision::FP16;

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->total_size != C_out) {
            throw std::runtime_error("conv1d_k7s3 bias size mismatch");
        }
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C_out);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C_out);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv1d_k7s3 bias only supports FP16/FP32");
        }
    }

    cactus_conv1d_f16_k7s3_oc8(
        X.data_as<__fp16>(), 
        W.data_as<__fp16>(), 
        bias_ptr,
        Y.data_as<__fp16>(), 
        N, L, C_in, C_out
    );
}

void compute_maxpool1d_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                            const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);

    size_t batch_size = input.shape[0];
    size_t channels = input.shape[1];
    size_t input_length = input.shape[2];
    size_t kernel_size = node.params.kernel_size;
    size_t stride = node.params.stride;

    cactus_maxpool1d_f16(
        input.data_as<__fp16>(),
        node.output_buffer.data_as<__fp16>(),
        batch_size, channels, input_length,
        kernel_size, stride);
}

void compute_conv2d_k3s1p1_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                 const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }
    auto& Y = node.output_buffer;

    if (X.shape.size() != 4) {
        throw std::runtime_error("conv2d_k3s1p1 expects input [N, C_in, H, W]");
    }
    if (X.precision != Precision::FP16) {
        throw std::runtime_error("conv2d_k3s1p1 only supports FP16 activations");
    }

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t H = X.shape[2];
    const size_t W_in = X.shape[3];
    const size_t C_out = Y.shape[1];

    if (H == 0 || W_in == 0) {
        throw std::runtime_error("conv2d_k3s1p1 input spatial dimensions must be > 0");
    }

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C_out);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C_out);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv2d_k3s1p1 bias only supports FP16/FP32");
        }
    }

    if (W.precision == Precision::FP16) {
        if (W.shape.size() != 4) {
            throw std::runtime_error("conv2d_k3s1p1 FP16 weight must be [C_out, C_in, 3, 3]");
        }
        cactus_conv2d_f16_k3s1p1_nchw(
            X.data_as<__fp16>(), W.data_as<__fp16>(), bias_ptr,
            Y.data_as<__fp16>(),
            N, C_in, H, W_in, C_out);
        return;
    }

    throw std::runtime_error("conv2d_k3s1p1 only supports FP16 weights");
}
