import torch

def patch_gemma4_audio_mask_for_export() -> None:
    import transformers.models.gemma4.modeling_gemma4 as gemma4_modeling

    #Gemma-specific: computes Gemma4 audio tower local bidirectional attention masks without vmap/proxy issues. X
    def exportable_bidirectional_mask(config, inputs_embeds, attention_mask=None, and_mask_function=None, **kwargs):
        batch_size, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device
        q_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
        kv_idx = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)

        left_window_size = getattr(config, "attention_context_left", seq_len) - 1
        right_window_size = getattr(config, "attention_context_right", 0)
        distance = q_idx - kv_idx
        left_mask = (distance >= 0) & (distance < left_window_size)
        right_mask = (distance < 0) & (-distance < right_window_size)
        mask = left_mask | right_mask

        if attention_mask is not None:
            mask = mask & attention_mask[:, None, None, :].bool()

        return mask.expand(batch_size, 1, seq_len, seq_len)

    gemma4_modeling.create_bidirectional_mask = exportable_bidirectional_mask


def patch_transformers_moe_grouped_mm_for_export() -> None:
    import transformers.integrations.moe as moe

    moe._can_use_grouped_mm = lambda input, weight, offs: False
