#include "cactus.h"
#include "common.h"
#include "llama.h"
#include <vector>
#include <string>

namespace cactus {

int cactus_context::applyLoraAdapters(std::vector<common_adapter_lora_info> lora_adapters) {
    if (!ctx || !model) {
        LOG_ERROR("Context or model not initialized for applying LoRA adapters.");
        return -1;
    }

    for (auto &la : lora_adapters) {
        if (la.path.empty()) {
            LOG_WARNING("Skipping LoRA adapter with empty path.");
            continue;
        }
        la.ptr = llama_adapter_lora_init(model, la.path.c_str());
        if (la.ptr == nullptr) {
            LOG_ERROR("Failed to initialize LoRA adapter '%s'\n", la.path.c_str());
            return -1;
        }
        LOG_INFO("Initialized LoRA adapter: %s, Scale: %f", la.path.c_str(), la.scale);
    }

    this->lora = lora_adapters; 

    common_set_adapter_lora(ctx, this->lora);
    LOG_INFO("Applied %zu LoRA adapters.", this->lora.size());
    return 0;
}

void cactus_context::removeLoraAdapters() {
    if (!ctx) {
        LOG_ERROR("Context not initialized, cannot remove LoRA adapters.");
        return;
    }
    this->lora.clear();
    common_set_adapter_lora(ctx, this->lora);
    LOG_INFO("Removed all LoRA adapters.");
}

std::vector<common_adapter_lora_info> cactus_context::getLoadedLoraAdapters() {
    return this->lora;
}

} // namespace cactus