"""Cloud cost accounting, isolated here so pricing/estimation is easy to audit.

Primary path: the engine patch surfaces real cloud token usage as
`cloud_prompt_tokens` / `cloud_completion_tokens` (parsed from the cloud
endpoint's token_usage block). Fallback (unpatched engine): prompt tokens from
the local `prefill_tokens`, completion tokens estimated from response length.
"""
from __future__ import annotations

from typing import Any

PRICE_PER_M_INPUT = 0.30
PRICE_PER_M_OUTPUT = 2.50
CHARS_PER_TOKEN = 4.0


def resolve_cloud_tokens(resp: dict[str, Any]) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) the cloud was billed for."""
    pt = int(resp.get("cloud_prompt_tokens") or 0)
    ct = int(resp.get("cloud_completion_tokens") or 0)
    if pt or ct:
        return pt, ct
    pt = int(resp.get("prefill_tokens") or 0)
    text = resp.get("response") or ""
    ct = max(1, round(len(text) / CHARS_PER_TOKEN)) if text else 0
    return pt, ct


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * PRICE_PER_M_INPUT
            + completion_tokens * PRICE_PER_M_OUTPUT) / 1_000_000.0
