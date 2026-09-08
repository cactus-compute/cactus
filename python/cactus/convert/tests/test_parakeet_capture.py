from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from cactus.transpile.hf_model import _run_parakeet_tdt_component_decode
from cactus.transpile.tdt_runtime import build_parakeet_tdt_component_specs


def test_capture_window_is_independent_of_validation_audio():
    features = torch.full((1, 2003, 128), 2.0)
    model = SimpleNamespace(
        config=SimpleNamespace(blank_id=0, predictor_num_layers=0),
        encoder=lambda features, mask: features[:, ::8, :4],
        initial_decoder_state=lambda **kwargs: (),
        decoder_step=object(),
    )
    with patch.dict("os.environ", {"CACTUS_TRANSPILER_AUDIO_BUCKETS": ""}):
        specs = build_parakeet_tdt_component_specs(model, named_tensors={"input_features": features})
    encoders = specs[:-1]
    assert [int(spec.metadata["audio_frames"]) for spec in encoders] == [100, 200, 300, 500, 800, 1300, 2100, 3000]
    assert encoders[-1].component == "audio_encoder"
    assert tuple(encoders[-1].example_inputs[0].shape) == (1, 3000, 128)
    assert torch.all(features == 2.0)
    assert not torch.all(encoders[-1].example_inputs[0] == 2.0)


def test_validation_preserves_real_features_and_masks_padding():
    features = torch.randn(1, 2003, 128)
    prepared = SimpleNamespace(names=("input_features",), tensors=(features,))
    encoder = SimpleNamespace(metadata={"audio_frames": "2100"})
    components = {
        "audio_encoder": SimpleNamespace(metadata={"audio_frames": "3000"}),
        "audio_encoder_2100": encoder,
        "decoder": object(),
    }
    model = SimpleNamespace(
        config=SimpleNamespace(subsampling_factor=8, predictor_num_layers=0),
        initial_decoder_state=lambda **kwargs: (),
        decode_token_ids=lambda tokens: "test",
    )

    def execute(selected, *, initial_store):
        assert selected == [encoder]
        padded = initial_store["input_features"]
        mask = initial_store["input_features_mask"]
        assert tuple(padded.shape) == (1, 2100, 128)
        assert torch.equal(padded[:, :2003], features)
        assert torch.all(padded[:, 2003:] == 0)
        assert torch.all(mask[:, :2003] == 1)
        assert torch.all(mask[:, 2003:] == 0)
        return {"encoder_hidden_states": np.zeros((1, 263, 4))}, {}

    with (
        patch("cactus.transpile.hf_model.execute_component_pipeline", side_effect=execute),
        patch("cactus.transpile.hf_model.greedy_decode_parakeet_tdt_token_ids", return_value=[]) as decode,
    ):
        _run_parakeet_tdt_component_decode(component_graphs=components, model=model, prepared=prepared)
    assert decode.call_args.kwargs["encoder_hidden_states"].shape == (1, 251, 4)
