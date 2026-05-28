from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CTX_SIZE = 16384


@dataclass(frozen=True)
class CachePolicy:
    ctx_size: int | None
    compact_to: int | None
    keep: int | None
    context_shift: bool
    keep_prompt: bool
    use_model_context: bool = False


def parse_optional_int(
    value: str | int | None,
    *,
    option_name: str,
    allow_auto: bool = False,
    allow_zero: bool = False,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "":
            return None
        if allow_auto and normalized == "auto":
            return None
        parsed = int(normalized)
    else:
        parsed = int(value)
    if parsed < 0 or (parsed == 0 and not allow_zero):
        quantity = "a non-negative integer" if allow_zero else "a positive integer"
        raise ValueError(f"{option_name} must be {quantity}" + (" or auto" if allow_auto else ""))
    return parsed


def normalize_cache_policy(
    *,
    ctx_size: str | int | None = None,
    cache_context_length: str | int | None = None,
    context_shift: bool | None = None,
    keep: str | int | None = None,
    keep_prompt: bool | None = None,
) -> CachePolicy:
    ctx_explicit = parse_optional_int(ctx_size, option_name="--ctx-size", allow_auto=True)
    cache_explicit = parse_optional_int(
        cache_context_length,
        option_name="--cache-context-length",
        allow_auto=True,
    )
    ctx_auto = isinstance(ctx_size, str) and ctx_size.strip().lower() == "auto"
    cache_auto = isinstance(cache_context_length, str) and cache_context_length.strip().lower() == "auto"
    if ctx_explicit is not None and cache_explicit is not None and ctx_explicit != cache_explicit:
        raise ValueError("--ctx-size and --cache-context-length disagree")
    if (ctx_explicit is not None and cache_auto) or (cache_explicit is not None and ctx_auto):
        raise ValueError("--ctx-size and --cache-context-length disagree")

    use_model_context = ctx_auto or cache_auto
    resolved_ctx = ctx_explicit if ctx_explicit is not None else cache_explicit
    if resolved_ctx is None and not use_model_context:
        resolved_ctx = DEFAULT_CTX_SIZE

    resolved_context_shift = True if context_shift is None else bool(context_shift)
    resolved_keep_prompt = True if keep_prompt is None else bool(keep_prompt)

    if use_model_context:
        return CachePolicy(
            ctx_size=None,
            compact_to=None,
            keep=None,
            context_shift=resolved_context_shift,
            keep_prompt=resolved_keep_prompt,
            use_model_context=True,
        )

    assert resolved_ctx is not None
    compact_to = max(1, resolved_ctx // 2)
    resolved_keep = (
        compact_to // 2
        if keep is None
        else parse_optional_int(keep, option_name="--keep", allow_auto=False, allow_zero=True)
    )
    assert resolved_keep is not None
    if resolved_keep >= compact_to:
        raise ValueError("--keep must be less than half of --ctx-size")

    return CachePolicy(
        ctx_size=resolved_ctx,
        compact_to=compact_to,
        keep=resolved_keep,
        context_shift=resolved_context_shift,
        keep_prompt=resolved_keep_prompt,
    )
