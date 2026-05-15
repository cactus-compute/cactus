from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


REQUIRED_TARGET_ROLES = (
    "verifier_logits",
    "target_hidden_state",
    "assistant_shared_state_tensors",
)
REQUIRED_ASSISTANT_ROLES = (
    "current_token",
    "previous_target_hidden",
    "target_shared_state_inputs",
    "position",
    "logits_output",
    "next_hidden_output",
)


def validate_spec_decode_manifest(manifest: Mapping[str, Any]) -> None:
    spec = manifest.get("spec_decode")
    if not isinstance(spec, Mapping):
        raise ValueError("manifest is missing spec_decode")
    version = spec.get("version")
    if version != 1:
        raise ValueError(f"unsupported spec_decode version: {version!r}")
    method = spec.get("method")
    if method != "assistant_chain":
        raise ValueError(f"unsupported spec_decode method: {method!r}")

    target = spec.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("spec_decode manifest missing target roles")
    assistant = spec.get("assistant")
    if not isinstance(assistant, Mapping):
        raise ValueError("spec_decode manifest missing assistant roles")

    for role in REQUIRED_TARGET_ROLES:
        value = target.get(role)
        if not value:
            raise ValueError(f"spec_decode manifest missing required role: target.{role}")
    for role in REQUIRED_ASSISTANT_ROLES:
        value = assistant.get(role)
        if not value:
            raise ValueError(f"spec_decode manifest missing required role: assistant.{role}")


@dataclass(frozen=True)
class TargetSpecDecodeAdapter:
    manifest: Mapping[str, Any]

    def emit_roles(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        validate_spec_decode_manifest(self.manifest)
        target = self.manifest["spec_decode"]["target"]
        shared_names = tuple(target["assistant_shared_state_tensors"])
        roles = {
            "verifier_logits": _require_output(outputs, target["verifier_logits"]),
            "target_hidden_state": _require_output(outputs, target["target_hidden_state"]),
            "assistant_shared_state_tensors": tuple(_require_output(outputs, name) for name in shared_names),
        }
        return roles


@dataclass(frozen=True)
class AssistantSpecDecodeAdapter:
    manifest: Mapping[str, Any]

    def prepare_inputs(self, role_values: Mapping[str, Any]) -> dict[str, Any]:
        validate_spec_decode_manifest(self.manifest)
        assistant = self.manifest["spec_decode"]["assistant"]
        shared_names = tuple(assistant["target_shared_state_inputs"])
        prepared = {
            assistant["current_token"]: _require_output(role_values, "current_token"),
            assistant["previous_target_hidden"]: _require_output(role_values, "previous_target_hidden"),
            assistant["position"]: _require_output(role_values, "position"),
        }
        shared_values = tuple(_require_output(role_values, "target_shared_state_inputs"))
        if len(shared_values) != len(shared_names):
            raise ValueError(
                "target_shared_state_inputs count does not match assistant manifest roles"
            )
        prepared.update(dict(zip(shared_names, shared_values, strict=True)))
        return prepared

    def emit_roles(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        validate_spec_decode_manifest(self.manifest)
        assistant = self.manifest["spec_decode"]["assistant"]
        return {
            "assistant_logits": _require_output(outputs, assistant["logits_output"]),
            "next_assistant_hidden": _require_output(outputs, assistant["next_hidden_output"]),
        }


def _require_output(values: Mapping[str, Any], key: str) -> Any:
    if key not in values:
        raise KeyError(f"missing manifest-mapped value: {key}")
    return values[key]
