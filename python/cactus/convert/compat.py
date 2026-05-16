from __future__ import annotations


def patch_transformers_import_compat() -> list[str]:
    """Keep Transformers processor/model imports working in local dev envs.

    Some pyenv Python builds are missing the stdlib `_lzma` extension. Importing
    `AutoProcessor` can still reach torchvision datasets, which import `lzma`
    even when conversion only needs processor metadata. The transpiler already
    has the battle-tested patch, so reuse it before converter-side HF imports.
    """

    notes: list[str] = []
    try:
        from cactus.transpile.runtime_support import (
            ensure_transformers_supports_gemma4,
            patch_torch_flex_attention_compat,
            patch_transformers_torchvision_probe,
        )
    except Exception:
        return notes

    for patch in (
        patch_transformers_torchvision_probe,
        patch_torch_flex_attention_compat,
        ensure_transformers_supports_gemma4,
    ):
        try:
            note = patch()
        except Exception:
            note = None
        if note:
            notes.append(str(note))
    return notes
