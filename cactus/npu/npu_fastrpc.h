#pragma once
#include "npu.h"
#include <memory>

namespace cactus {
namespace npu {

// NPUPrefill implementation backed by FastRPC + libggml-htp-vNN.so DSP skel.
// Replaces the QNN-based QNNDirectPrefill; no QAIRT SDK installation required.
// The DSP skel (libggml-htp-v73.so) is built from llama.cpp ggml-hexagon sources
// and must be registered as a Windows driver (pnputil /add-driver libggml-htp.inf).
class FastRPCPrefill : public NPUPrefill {
public:
    FastRPCPrefill();
    ~FastRPCPrefill() override;

    bool load(const std::string& model_path) override;
    bool is_available() const override;
    int  get_chunk_size() const override;
    int  get_hidden_dim() const override;
    int  get_num_layers() const override;
    int  get_num_kv_heads() const override;
    int  get_head_dim() const override;

    NPUPrefillDirectResult prefill_chunk_direct(
        const std::vector<__fp16>& embeddings,
        int position_offset = 0,
        const std::string& input_name = "x") override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace npu
} // namespace cactus
