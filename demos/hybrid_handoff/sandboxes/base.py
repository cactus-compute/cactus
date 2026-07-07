"""Base class for mock agent sandboxes.

A Sandbox is a small *queryable mock environment*: a data pool loaded from
JSON/markdown plus tool functions that actually filter/search that pool by the
model's arguments. There is no canned trajectory -- the agent free-runs over the
data, so an arbitrary --query works.

To add a sandbox: subclass Sandbox, fill `tools` (a list of ToolSpec), set
`system_prompt` + `default_query`, and load your data in `__init__`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSpec:
    """One tool: its OpenAI function schema + the Python callable that runs it."""
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Sandbox:
    name: str = "base"
    system_prompt: str = ""
    default_query: str = ""

    def __init__(self) -> None:
        self.tools: list[ToolSpec] = []
        self._by_name: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self.tools.append(spec)
        self._by_name[spec.name] = spec

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools]

    def dispatch(self, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
        """Run a tool by name with parsed args. Always returns a dict (never raises)
        so the agent can recover from a bad call."""
        spec = self._by_name.get(name)
        if spec is None:
            return {"error": f"unknown tool '{name}'. available: {list(self._by_name)}"}
        args = args or {}
        if not isinstance(args, dict):
            return {"error": f"arguments for '{name}' must be an object, got {type(args).__name__}"}
        try:
            result = spec.fn(**args)
        except TypeError as exc:
            return {"error": f"bad arguments for '{name}': {exc}"}
        except Exception as exc:
            return {"error": f"tool '{name}' failed: {exc}"}
        return result if isinstance(result, dict) else {"result": result}

    def _data_path(self, filename: str) -> str:
        return os.path.join(os.path.dirname(self._module_file()), "data", filename)

    def _module_file(self) -> str:
        import sys
        return sys.modules[self.__class__.__module__].__file__

    def load_json(self, filename: str) -> Any:
        with open(self._data_path(filename), encoding="utf-8") as fh:
            return json.load(fh)

    def load_text(self, filename: str) -> str:
        with open(self._data_path(filename), encoding="utf-8") as fh:
            return fh.read()
