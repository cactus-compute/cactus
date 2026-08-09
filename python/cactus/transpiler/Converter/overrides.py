import torch
from typing import Any

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

def patch_clip_position_ids_for_export() -> None:
    import transformers.models.clip.modeling_clip as clip_modeling
    #CLIP registers position_ids as a non-persistent buffer, which torch.export lifts
    #into a weight placeholder no checkpoint can bind; recompute it in the graph instead.
    original_forward = clip_modeling.CLIPTextEmbeddings.forward
    def forward_with_computed_position_ids(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(length, device=device).unsqueeze(0)
        return original_forward(self, input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
    clip_modeling.CLIPTextEmbeddings.forward = forward_with_computed_position_ids

def patch_transformers_moe_grouped_mm_for_export() -> None:
    import transformers.integrations.moe as moe
    moe._can_use_grouped_mm = lambda input, weight, offs: False

def patch_lfm2_vl_image_features_for_export() -> None:
    import transformers.models.lfm2_vl.modeling_lfm2_vl as lfm2_vl_modeling
    import transformers.models.siglip2.modeling_siglip2 as siglip2_modeling
    import torch.nn.functional as F
    original_get_image_features = lfm2_vl_modeling.Lfm2VlModel.get_image_features
    original_embeddings_forward = siglip2_modeling.Siglip2VisionEmbeddings.forward
    def exportable_embeddings_forward(self, pixel_values, spatial_shapes):
        static_spatial_shapes = getattr(self, "_cactus_export_spatial_shapes", None)
        if not static_spatial_shapes:
            return original_embeddings_forward(self, pixel_values, spatial_shapes)
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size,
            self.position_embedding_size,
            -1,
        )
        resized_positional_embeddings = static_resize_siglip2_positional_embeddings(
            positional_embeddings,
            static_spatial_shapes,
            max_length=pixel_values.shape[1],
        )
        return patch_embeds + resized_positional_embeddings
    def static_resize_siglip2_positional_embeddings(positional_embeddings, static_spatial_shapes, max_length):
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype
        rows = []
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)
        for height, width in static_spatial_shapes:
            height = int(height)
            width = int(width)
            if height == int(positional_embeddings.shape[-2]) and width == int(positional_embeddings.shape[-1]):
                resized_embeddings = positional_embeddings.squeeze(0).permute(1, 2, 0).reshape(height * width, embed_dim)
            else:
                resized_embeddings = F.interpolate(
                    positional_embeddings,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)
            resized_embeddings = resized_embeddings.to(source_dtype)
            pad_length = max(0, int(max_length) - height * width)
            if pad_length > 0:
                padding = resized_embeddings[0].unsqueeze(0).expand(pad_length, embed_dim)
                resized_embeddings = torch.cat((resized_embeddings, padding), dim=0)
            rows.append(resized_embeddings[: int(max_length)])
        return torch.stack(rows, dim=0)
    def exportable_get_image_features(self, pixel_values, spatial_shapes, pixel_attention_mask, **kwargs):
        static_spatial_shapes = getattr(self, "_cactus_export_spatial_shapes", None)
        if not static_spatial_shapes:
            return original_get_image_features(self, pixel_values, spatial_shapes, pixel_attention_mask, **kwargs)
        kwargs = dict(kwargs)
        kwargs.pop("return_dict", None)
        image_outputs = self.vision_tower(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = image_outputs.last_hidden_state
        image_features = []
        for img_idx, (height, width) in enumerate(static_spatial_shapes):
            feature_length = int(height) * int(width)
            feature = last_hidden_state[img_idx, :feature_length, :].unsqueeze(0)
            feature = feature.reshape(1, int(height), int(width), -1)
            img_embedding = self.multi_modal_projector(feature)
            image_features.append(img_embedding.reshape(-1, img_embedding.size(-1)))
        image_outputs.pooler_output = image_features
        return image_outputs
    lfm2_vl_modeling.Lfm2VlModel.get_image_features = exportable_get_image_features
    siglip2_modeling.Siglip2VisionEmbeddings.forward = exportable_embeddings_forward

def prepare_model_input_hints_for_export(model: torch.nn.Module, kwargs: dict[str, Any]) -> None:
    spatial_shapes = kwargs.get("spatial_shapes")
    if not isinstance(spatial_shapes, torch.Tensor):
        return
    try:
        shapes = tuple(tuple(int(value) for value in row) for row in spatial_shapes.detach().cpu().tolist())
    except Exception:
        return
    for candidate in (model, getattr(model, "model", None)):
        if candidate is not None and hasattr(candidate, "get_image_features"):
            setattr(candidate, "_cactus_export_spatial_shapes", shapes)
        embeddings = getattr(getattr(getattr(candidate, "vision_tower", None), "vision_model", None), "embeddings", None)
        if embeddings is not None:
            setattr(embeddings, "_cactus_export_spatial_shapes", shapes)
