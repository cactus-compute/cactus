from __future__ import annotations

import builtins
import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from cactus.transpile.model_profiles import GEMMA4_PROFILE
from cactus.transpile.model_profiles import ModelProfile
from cactus.transpile.model_profiles import profile_for_model_type


_TORCHVISION_COMPAT_LIBRARIES: list[object] = []


@dataclass
class PreparedInputs:
    names: tuple[str, ...]
    tensors: tuple[torch.Tensor, ...]
    metadata: dict[str, object]


def _transformers_supports_model_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _candidate_external_site_packages() -> list[Path]:
    candidates: list[Path] = []
    major_minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    pyenv_versions = Path.home() / ".pyenv" / "versions"
    if not pyenv_versions.exists():
        return candidates
    for version_dir in sorted(pyenv_versions.iterdir(), reverse=True):
        site_packages = version_dir / "lib" / major_minor / "site-packages"
        if site_packages.exists():
            candidates.append(site_packages)
    return candidates


def ensure_transformers_supports_profile(profile: ModelProfile | None) -> str | None:
    target_module = None if profile is None else profile.transformer_module
    if not target_module or _transformers_supports_model_module(target_module):
        return None

    for site_packages in _candidate_external_site_packages():
        candidate = site_packages / Path(*target_module.split("."))
        if not candidate.with_suffix(".py").exists() and not candidate.is_dir():
            continue
        sys.path.insert(0, str(site_packages))
        for module_name in list(sys.modules):
            root_name = module_name.split(".", 1)[0]
            if root_name in {"transformers", "huggingface_hub", "tokenizers"}:
                del sys.modules[module_name]
        if _transformers_supports_model_module(target_module):
            return str(site_packages)
        try:
            sys.path.remove(str(site_packages))
        except ValueError:
            pass
    return None


def ensure_transformers_supports_model_type(model_type: str) -> str | None:
    return ensure_transformers_supports_profile(profile_for_model_type(model_type))


def ensure_transformers_supports_gemma4() -> str | None:
    return ensure_transformers_supports_profile(GEMMA4_PROFILE)


def _patch_torchvision_missing_nms_op() -> str | None:
    """Keep mismatched torch/torchvision installs importable for processors."""

    try:
        import torchvision  # type: ignore  # noqa: F401
        return None
    except RuntimeError as exc:
        if "operator torchvision::nms does not exist" not in str(exc):
            return None
    except Exception:
        return None

    try:
        import torchvision.extension as tv_extension  # type: ignore

        if bool(getattr(tv_extension, "_HAS_OPS", False)):
            return None
    except Exception:
        pass

    try:
        library = torch.library.Library("torchvision", "DEF")
        library.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        _TORCHVISION_COMPAT_LIBRARIES.append(library)
        return "defined missing torchvision::nms operator for torchvision import compatibility"
    except Exception:
        return None


def patch_transformers_torchvision_probe() -> str | None:
    has_torchvision = importlib.util.find_spec("torchvision") is not None
    has_lzma = importlib.util.find_spec("_lzma") is not None

    if not has_torchvision:
        return None

    if has_lzma:
        nms_patch_note = _patch_torchvision_missing_nms_op()
        return nms_patch_note

    base_note: str | None = None
    try:
        import backports.lzma as backports_lzma  # type: ignore

        sys.modules.setdefault("lzma", backports_lzma)
        base_note = "using backports.lzma because this Python build is missing _lzma"
        nms_patch_note = _patch_torchvision_missing_nms_op()
        if nms_patch_note:
            return f"{base_note}; {nms_patch_note}"
        return base_note
    except Exception:
        pass

    class _InterpolationModeStub:
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"

    class _TorchvisionFunctionalStub:
        InterpolationMode = _InterpolationModeStub

        def __getattr__(self, name: str):
            raise RuntimeError(
                "torchvision functionality is unavailable because this Python build is missing _lzma; "
                f"attempted to access torchvision.transforms.functional.{name}"
            )

    builtins.F = _TorchvisionFunctionalStub()
    builtins.tvF = builtins.F

    import transformers.utils as tf_utils  # type: ignore
    import transformers.utils.import_utils as tf_import_utils  # type: ignore

    @lru_cache
    def _disabled() -> bool:
        return False

    tf_import_utils.is_torchvision_available = _disabled
    tf_import_utils.is_torchvision_v2_available = _disabled
    tf_utils.is_torchvision_available = _disabled
    tf_utils.is_torchvision_v2_available = _disabled
    return "disabled torchvision import checks because this Python build is missing _lzma"


def patch_torch_flex_attention_compat() -> str | None:
    try:
        import torch.nn.attention.flex_attention as flex_attention  # type: ignore
    except Exception:
        return None

    if hasattr(flex_attention, "AuxRequest"):
        return None

    class _AuxRequest:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    flex_attention.AuxRequest = _AuxRequest  # type: ignore[attr-defined]
    return "installed torch flex_attention AuxRequest compatibility stub"
