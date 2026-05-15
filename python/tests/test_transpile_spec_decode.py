from __future__ import annotations

import numpy as np
import pytest

from cactus.transpile.spec_decode import AssistantSpecDecodeAdapter
from cactus.transpile.spec_decode import TargetSpecDecodeAdapter
from cactus.transpile.spec_decode import validate_spec_decode_manifest


def _manifest() -> dict[str, object]:
    return {
        "spec_decode": {
            "version": 1,
            "method": "assistant_chain",
            "target": {
                "verifier_logits": "target_logits",
                "target_hidden_state": "target_hidden",
                "assistant_shared_state_tensors": ["shared_a"],
            },
            "assistant": {
                "current_token": "assistant_token",
                "previous_target_hidden": "assistant_prev_hidden",
                "target_shared_state_inputs": ["assistant_shared_a"],
                "position": "assistant_position",
                "logits_output": "assistant_logits_out",
                "next_hidden_output": "assistant_hidden_out",
            },
        }
    }


def test_valid_spec_decode_manifest_loads() -> None:
    validate_spec_decode_manifest(_manifest())


def test_missing_required_role_fails_clearly() -> None:
    manifest = _manifest()
    manifest["spec_decode"]["assistant"]["logits_output"] = ""
    with pytest.raises(ValueError, match="assistant.logits_output"):
        validate_spec_decode_manifest(manifest)


def test_unsupported_method_fails_clearly() -> None:
    manifest = _manifest()
    manifest["spec_decode"]["method"] = "tree"
    with pytest.raises(ValueError, match="unsupported spec_decode method"):
        validate_spec_decode_manifest(manifest)


def test_unsupported_version_fails_clearly() -> None:
    manifest = _manifest()
    manifest["spec_decode"]["version"] = 2
    with pytest.raises(ValueError, match="unsupported spec_decode version"):
        validate_spec_decode_manifest(manifest)


def test_target_adapter_emits_expected_role_names_and_shapes() -> None:
    logits = np.zeros((1, 3, 5), dtype=np.float32)
    hidden = np.zeros((1, 3, 7), dtype=np.float32)
    shared = np.zeros((1, 7), dtype=np.float32)
    roles = TargetSpecDecodeAdapter(_manifest()).emit_roles(
        {"target_logits": logits, "target_hidden": hidden, "shared_a": shared}
    )
    assert roles["verifier_logits"].shape == (1, 3, 5)
    assert roles["target_hidden_state"].shape == (1, 3, 7)
    assert roles["assistant_shared_state_tensors"][0].shape == (1, 7)


def test_assistant_adapter_is_manifest_driven_not_positional() -> None:
    adapter = AssistantSpecDecodeAdapter(_manifest())
    prepared = adapter.prepare_inputs(
        {
            "position": np.array([4]),
            "target_shared_state_inputs": (np.array([[1.0]], dtype=np.float32),),
            "previous_target_hidden": np.zeros((1, 7), dtype=np.float32),
            "current_token": np.array([2]),
        }
    )
    assert tuple(prepared) == (
        "assistant_token",
        "assistant_prev_hidden",
        "assistant_position",
        "assistant_shared_a",
    )

    roles = adapter.emit_roles(
        {
            "assistant_hidden_out": np.zeros((1, 7), dtype=np.float32),
            "assistant_logits_out": np.zeros((1, 5), dtype=np.float32),
        }
    )
    assert roles["assistant_logits"].shape == (1, 5)
    assert roles["next_assistant_hidden"].shape == (1, 7)
