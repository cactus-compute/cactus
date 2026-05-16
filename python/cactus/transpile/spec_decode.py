from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


REQUIRED_TARGET_ROLES = (
    "verifier_logits",
    "target_hidden_state",
    "target_token_embedding",
)
REQUIRED_ASSISTANT_ROLES = (
    "current_token_embedding",
    "previous_target_hidden",
    "position",
    "logits_output",
    "next_hidden_output",
)
SHARED_KV_GROUPS = ("full_attention", "sliding_attention")
SHARED_KV_FIELDS = ("key", "value")


def validate_spec_decode_manifest(manifest: Mapping[str, Any]) -> None:
    spec = manifest.get("spec_decode")
    if not isinstance(spec, Mapping):
        raise ValueError("manifest is missing spec_decode")
    version = spec.get("version")
    if version != 1:
        raise ValueError(f"unsupported spec_decode version: {version!r}")
    method = spec.get("method")
    if method != "single_position_mtp":
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
    _validate_shared_kv_roles(target, "target")
    _validate_shared_kv_roles(assistant, "assistant")


@dataclass(frozen=True)
class TargetSpecDecodeAdapter:
    manifest: Mapping[str, Any]

    def emit_roles(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        validate_spec_decode_manifest(self.manifest)
        target = self.manifest["spec_decode"]["target"]
        roles = {
            "verifier_logits": _require_output(outputs, target["verifier_logits"]),
            "target_hidden_state": _require_output(outputs, target["target_hidden_state"]),
            "target_token_embedding": _require_output(outputs, target["target_token_embedding"]),
            "shared_kv": _emit_shared_kv(target, outputs),
        }
        return roles


@dataclass(frozen=True)
class AssistantSpecDecodeAdapter:
    manifest: Mapping[str, Any]

    def prepare_inputs(self, role_values: Mapping[str, Any]) -> dict[str, Any]:
        validate_spec_decode_manifest(self.manifest)
        assistant = self.manifest["spec_decode"]["assistant"]
        prepared = {
            assistant["current_token_embedding"]: _require_output(role_values, "current_token_embedding"),
            assistant["previous_target_hidden"]: _require_output(role_values, "previous_target_hidden"),
            assistant["position"]: _require_output(role_values, "position"),
        }
        shared_values = _require_output(role_values, "shared_kv")
        if not isinstance(shared_values, Mapping):
            raise ValueError("shared_kv values must be a mapping")
        prepared.update(_prepare_shared_kv_inputs(assistant, shared_values))
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


def _validate_shared_kv_roles(section: Mapping[str, Any], prefix: str) -> None:
    shared_kv = section.get("shared_kv")
    if not isinstance(shared_kv, Mapping):
        shared_kv = {}
    for group in SHARED_KV_GROUPS:
        group_roles = shared_kv.get(group)
        if not isinstance(group_roles, Mapping):
            group_roles = {}
        for field in SHARED_KV_FIELDS:
            if not group_roles.get(field):
                raise ValueError(
                    f"spec_decode manifest missing required role: {prefix}.shared_kv.{group}.{field}"
                )


def _emit_shared_kv(section: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    shared_kv = section["shared_kv"]
    return {
        group: {
            field: _require_output(values, shared_kv[group][field])
            for field in SHARED_KV_FIELDS
        }
        for group in SHARED_KV_GROUPS
    }


def _prepare_shared_kv_inputs(
    assistant: Mapping[str, Any],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    shared_kv = assistant["shared_kv"]
    prepared = {}
    for group in SHARED_KV_GROUPS:
        group_values = _require_output(values, group)
        if not isinstance(group_values, Mapping):
            raise ValueError(f"shared_kv.{group} values must be a mapping")
        for field in SHARED_KV_FIELDS:
            prepared[shared_kv[group][field]] = _require_output(group_values, field)
    return prepared
