"""Sandbox registry. Each sandbox is a queryable mock environment (data pool +
tools that filter it). Lazy-imported so a broken sandbox doesn't block others."""
from __future__ import annotations

from .base import Sandbox, ToolSpec

_REGISTRY = {
    "email": ("email", "EmailSandbox"),
    "calendar": ("calendar", "CalendarSandbox"),
    "expense": ("expense", "ExpenseSandbox"),
    "incident": ("incident", "IncidentSandbox"),
}


def available() -> list[str]:
    return list(_REGISTRY)


def get(name: str) -> Sandbox:
    if name not in _REGISTRY:
        raise KeyError(f"unknown sandbox '{name}'. available: {available()}")
    module_name, cls_name = _REGISTRY[name]
    module = __import__(f"{__name__}.{module_name}", fromlist=[cls_name])
    return getattr(module, cls_name)()


__all__ = ["Sandbox", "ToolSpec", "available", "get"]
