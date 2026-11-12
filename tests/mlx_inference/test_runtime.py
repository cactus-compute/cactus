import json
import pytest

from cactus.mlx import runtime
from cactus.mlx.models import loader_for


def test_unknown_architecture_rejected_before_weight_download(tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'model_type': 'unsupported',
                                  'quantization_config': {'quant_method': 'cq'}}))
    monkeypatch.setattr(runtime, 'hf_hub_download', lambda repo, name: str(config))

    def unexpected_download(*args, **kwargs):
        pytest.fail('Weights should not download for an unsupported architecture')

    monkeypatch.setattr(runtime, 'snapshot_download', unexpected_download)
    with pytest.raises(ValueError, match='Unsupported CQ architecture'):
        runtime.resolve_model('another-owner/another-model')


def test_repository_selection_is_independent_of_architecture(tmp_path, monkeypatch):
    config = {'model_type': 'nemotron_h', 'quantization_config': {'quant_method': 'cq'}}
    for name in runtime.MODEL_FILES:
        (tmp_path / name).write_text(json.dumps(config) if name == 'config.json' else '')
    requested = []

    def fetch_config(repo, name):
        requested.append(repo)
        return str(tmp_path / name)

    def fetch_weights(repo, **kwargs):
        requested.append(repo)
        return str(tmp_path)

    monkeypatch.setattr(runtime, 'hf_hub_download', fetch_config)
    monkeypatch.setattr(runtime, 'snapshot_download', fetch_weights)
    for repo in ('owner/first-cq', 'different-owner/second-cq'):
        weights, directory = runtime.resolve_model(repo)
        assert weights == tmp_path / 'model.safetensors'
        assert directory == tmp_path
        assert requested[-2:] == [repo, repo]
    assert loader_for(config).__module__ == 'cactus.mlx.models.nemotron'


def test_dense_checkpoint_is_not_misread_as_cq():
    with pytest.raises(ValueError, match='requires a CQ checkpoint'):
        loader_for({'model_type': 'nemotron_h'})
