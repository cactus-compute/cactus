import unittest
from pathlib import Path

import torch

from cactus.cli import transpiler as cli_transpiler
from cactus.transpiler.Converter import diffusion
from cactus.transpiler.ModelProfiles import profiles
from cactus.transpiler.RuntimePlan import models as RPModels

MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"


class TestSdProfile(unittest.TestCase):
    def test_model_id_resolves_to_registered_profile_case_insensitively(self):
        profile = profiles.profile_for_model_id(MODEL_ID)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.model_profiles, "sd15_t2i")
        self.assertIs(profiles.profile_for_model_id(MODEL_ID.lower()), profile)

    def test_resolved_config_exports_one_mode_per_component(self):
        resolved = cli_transpiler.resolve_transpile_config(MODEL_ID)
        self.assertEqual(resolved.profile_source, "registered")
        self.assertEqual(resolved.inference_modes, ("text_encoder", "unet", "vae_decoder"))

    def test_registered_profile_rejects_generic_flags(self):
        with self.assertRaises(RuntimeError):
            cli_transpiler.resolve_transpile_config(MODEL_ID, generic_task="causal-lm")

    def test_component_sources_parse_kind_and_source(self):
        profile = profiles.profile_for_model_id(MODEL_ID)
        self.assertEqual(diffusion.component_source(profile, "unet"), ("sd_unet", "unet"))
        self.assertEqual(
            diffusion.component_source(profile, "vae_decoder"),
            ("taesd_decoder", "madebyollin/taesd"),
        )
        with self.assertRaises(ValueError):
            diffusion.component_source(profile, "decode_with_cache")

    def test_component_export_inputs_follow_the_configs(self):
        configs = {
            "unet/config.json": {
                "in_channels": 4, "sample_size": 64,
                "cross_attention_dim": 768, "time_cond_proj_dim": 256,
            },
            "text_encoder/config.json": {"max_position_embeddings": 77},
        }
        ids = diffusion.component_export_inputs("clip_text", configs)
        self.assertEqual(ids["input_ids"].shape, (1, 77))
        self.assertEqual(ids["input_ids"].dtype, torch.int64)

        unet = diffusion.component_export_inputs("sd_unet", configs)
        self.assertEqual(unet["sample"].shape, (1, 4, 64, 64))
        self.assertEqual(unet["timestep"].shape, (1,))
        self.assertEqual(unet["encoder_hidden_states"].shape, (1, 77, 768))
        self.assertEqual(unet["timestep_cond"].shape, (1, 256))
        self.assertTrue(all(t.dtype == torch.float16 for t in unet.values()))

        latent = diffusion.component_export_inputs("taesd_decoder", configs)
        self.assertEqual(latent["x"].shape, (1, 4, 64, 64))

        with self.assertRaises(ValueError):
            diffusion.component_export_inputs("vqgan", configs)

    def test_runtime_plan_declares_the_t2i_route_and_denoise_strategy(self):
        profile = profiles.profile_for_model_id(MODEL_ID)
        routes = RPModels.runtime_routes_from_model_profile(profile)
        by_name = {route.name: route for route in routes}
        self.assertIn("t2i_generate", by_name)
        edges = by_name["t2i_generate"].edges
        self.assertEqual([edge.output for edge in edges], ["unet", "vae_decoder"])
        self.assertEqual(edges[0].inputs, ("text_encoder",))

        metadata = RPModels.runtime_plan_metadata_from_model_profile(profile)
        self.assertEqual(metadata.get("runtime_execution_strategy"), "iterative_denoise")
        self.assertEqual(metadata.get("runtime_plan_name"), "sd15_text_to_image")

    def test_full_bundle_builds_from_the_cached_model(self):
        import tempfile

        try:
            import diffusers  # noqa: F401
        except Exception:
            self.skipTest("diffusers unavailable")
        try:
            from transformers import CLIPTextModel

            CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", local_files_only=True)
        except Exception as exc:
            self.skipTest(f"{MODEL_ID} not cached: {exc}")

        from cactus.cli.model import ensure_weights

        with tempfile.TemporaryDirectory() as tmp:
            weights_dir = ensure_weights(MODEL_ID, output_dir=Path(tmp) / "weights")
            self.assertTrue((weights_dir / "weights_manifest.json").exists())

            bundle = cli_transpiler.build_transpiled_bundle(
                MODEL_ID, weights_dir=weights_dir, output_dir=Path(tmp) / "bundle"
            )
            components = {path.stem for path in (bundle / "components").glob("*.cactus")}
            self.assertEqual(components, {"text_encoder", "unet", "vae_decoder"})
            self.assertTrue((bundle / "runtime_plan.json").exists())

            import json

            plan = json.loads((bundle / "runtime_plan.json").read_text())
            self.assertEqual(plan["metadata"].get("runtime_execution_strategy"), "iterative_denoise")


if __name__ == "__main__":
    unittest.main()
