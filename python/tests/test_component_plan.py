from __future__ import annotations

from cactus.transpile.component_plan import infer_component_plan_from_config


def test_gemma4_vision_only_config_does_not_request_audio_component() -> None:
    config = {
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "vision_config": {"hidden_size": 1152},
        "text_config": {"num_hidden_layers": 30},
    }

    plan = infer_component_plan_from_config(config, model_id="google/gemma-4-26B-A4B")

    assert plan is not None
    assert plan.task == "multimodal_causal_lm_logits"
    assert plan.components == ("vision_encoder", "lm_encoder", "decoder")
    assert plan.needs_image
    assert not plan.needs_audio


def test_gemma4_image_audio_config_keeps_audio_component() -> None:
    config = {
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "vision_config": {"hidden_size": 1152},
        "audio_config": {"hidden_size": 1024},
        "text_config": {"num_hidden_layers": 30},
    }

    plan = infer_component_plan_from_config(config, model_id="google/gemma-4-E2B-it")

    assert plan is not None
    assert plan.components == ("vision_encoder", "audio_encoder", "lm_encoder", "decoder")
    assert plan.needs_audio


def test_needle_config_uses_component_pipeline_plan() -> None:
    config = {
        "model_type": "needle",
        "architectures": ["NeedleForCausalLM"],
        "num_encoder_layers": 12,
        "num_decoder_layers": 8,
    }

    plan = infer_component_plan_from_config(config, model_id="Cactus-Compute/needle")

    assert plan is not None
    assert plan.task == "causal_lm_logits"
    assert plan.components == ("source_encoder", "decoder_cross_kv", "decoder_step")
    assert plan.force_component_pipeline
