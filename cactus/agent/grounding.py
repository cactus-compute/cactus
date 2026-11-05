from __future__ import annotations

import datetime
import decimal
import re

from .tools import _is_pydantic_model, pydantic_schema


class ExtractionValidationError(ValueError):
    """Extracted values are not grounded in the input."""


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict") and _is_pydantic_model(type(value)):
        return value.dict()
    return str(value)


def _schema_parameters(schema):
    raw = pydantic_schema(schema) if _is_pydantic_model(schema) else schema
    return raw.get("parameters", raw) if isinstance(raw, dict) else {}


def _resolve_ref(node, root):
    seen = set()
    while isinstance(node, dict) and "$ref" in node:
        if node["$ref"] in seen:
            break
        seen.add(node["$ref"])
        target = root
        for part in node["$ref"].removeprefix("#/").split("/"):
            target = target.get(part.replace("~1", "/").replace("~0", "~"), {})
        if target is node:
            break
        node = target
    return node


def _source_years(text):
    months = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
              r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
              r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
    patterns = [


        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{months}[\s,]+(\d{{1,4}})(?![0-9A-Za-z]|:\d|\s*[ap]\.?m\b)",
        rf"\b{months}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s+(\d{{1,4}})(?![0-9A-Za-z]|:\d|\s*[ap]\.?m\b)",
        rf"\b{months}[\s,]+(\d{{3,4}})(?![0-9A-Za-z])",
        r"\byear\s+(\d{1,4})(?![0-9A-Za-z])",


        r"(?<![0-9])(\d{4})(?=[-/]\d{1,2}[-/]\d{1,2}(?![0-9]))",
    ]
    lowered = text.lower()
    return {int(match.group(1)) for pattern in patterns
            for match in re.finditer(pattern, lowered)}


_ISO_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}")


_DATE_KEY = re.compile(r"(?<![A-Za-z])date:")


def _with_date_fact(system: str) -> str:
    """Prefix the local date fact unless the caller already supplied one."""
    if _DATE_KEY.search(system) or _ISO_STAMP.search(system):
        return system
    now = datetime.datetime.now()
    fact = now.strftime("date: %Y-%m-%d %a %H:%M")
    if not system.strip():
        return fact
    if system.lstrip().startswith("{"):
        return system
    return fact + "; " + system


_RELATIVE_CUE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|next|this|coming|now|in \d+ (?:days?|weeks?|months?|years?)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)


def _relative_cue(text) -> bool:
    return bool(text) and _RELATIVE_CUE.search(text) is not None


def _licensed_years(seen_years, system=None, relative=True):
    """Years a date argument may carry: the ones written in the input, plus the
    system date's year when the input reasons relatively (tomorrow, next week)."""
    years = set(seen_years)
    if years and system and relative:
        years |= _source_years(system)
    return years


def _walk_grounding(schema, arguments, years):
    root = _schema_parameters(schema)
    checked, failures = set(), set()

    def walk(value, node, path):
        node = _resolve_ref(node, root)
        variants = node.get("anyOf") or node.get("oneOf") or []
        concrete = [v for v in variants if _resolve_ref(v, root).get("type") != "null"]
        if len(concrete) == 1:
            node = _resolve_ref(concrete[0], root)
        fmt = node.get("format")
        if fmt in ("date", "date-time") and isinstance(value, str):
            match = re.match(r"^(\d{4})-", value)
            if match and years:
                checked.add(path)
                if int(match.group(1)) not in years:
                    failures.add(path)
            return
        if isinstance(value, dict):
            properties = node.get("properties", {})
            for key, item in value.items():
                if key in properties:
                    walk(item, properties[key], f"{path}.{key}" if path else key)
        elif isinstance(value, list) and "items" in node:
            for index, item in enumerate(value):
                walk(item, node["items"], f"{path}[{index}]")

    walk(arguments, root, "")
    return checked, failures


def _temporal_grounding(text, schema, arguments, system=None):
    years = _licensed_years(_source_years(text), system, _relative_cue(text))
    return _walk_grounding(schema, arguments, years)


_NUMBER_TOKEN = re.compile(
    r"(?<![\w.,])(?:[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)(?!\d)")


def _numeric_paths(arguments, path=""):
    """Yield ``(path, value)`` for every numeric leaf, arguments-relative.

    Paths match the form both call sites derive after stripping the engine's
    tool prefix, so a flagged ``Tool.path`` resolves to a helper path.
    """
    if isinstance(arguments, bool):
        return
    if isinstance(arguments, (int, float)):
        yield path, arguments
    elif isinstance(arguments, dict):
        for key, item in arguments.items():
            yield from _numeric_paths(item, f"{path}.{key}" if path else str(key))
    elif isinstance(arguments, list):
        for index, item in enumerate(arguments):
            yield from _numeric_paths(item, f"{path}[{index}]")


_DATE_FACT = re.compile(r"date:\s*\d{4}-\d{2}-\d{2}(?:\s+[A-Za-z]{3})?(?:\s+\d{2}:\d{2})?|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?")


def _without_date_facts(text):
    return _DATE_FACT.sub(" ", text or "")


def _source_numbers(*sources):
    sources = tuple(_without_date_facts(source) for source in sources)
    """Numeric values written in the source texts, separators normalized."""
    text = "\n".join(source for source in sources if source)
    return {decimal.Decimal(match.group(0).replace(",", ""))
            for match in _NUMBER_TOKEN.finditer(text)}


def _grounded_number_paths(arguments, *sources):
    numbers = _source_numbers(*sources)
    if not numbers:
        return set()
    grounded = set()
    for path, value in _numeric_paths(arguments):
        try:
            if decimal.Decimal(str(value)) in numbers:
                grounded.add(path)
        except (decimal.InvalidOperation, ValueError):
            continue
    return grounded


def _ungrounded_paths(response):
    validation = response.get("validation") or {}
    grouped = {}
    for name in validation.get("ungrounded") or []:
        tool, _, path = str(name).partition(".")
        grouped.setdefault(tool, set()).add(path or tool)
    return grouped


def _annotate_ungrounded(response, tool_schemas, seen_years, system=None, text=None):
    calls = response.get("function_calls") or []
    years = _licensed_years(seen_years, system, _relative_cue(text) if text is not None else True)
    if not calls or not years:
        return
    schemas = {entry.get("name"): entry for entry in tool_schemas}
    found = []
    for call in calls:
        schema = schemas.get(call.get("name"))
        if schema is None:
            continue
        _, failures = _walk_grounding(schema, call.get("arguments") or {}, years)
        found.extend(f"{call['name']}.{path}" for path in sorted(failures))
    if not found:
        return
    validation = response.setdefault("validation", {}) or {}
    response["validation"] = validation
    existing = list(validation.get("ungrounded") or [])
    validation["ungrounded"] = existing + [
        name for name in found if name not in existing]


def _validate_extraction(text, schema, arguments, response, system=None):
    checked, failures = _temporal_grounding(text, schema, arguments, system)
    validation = response.get("validation") or {}
    flagged = validation.get("ungrounded") or []
    grounded = _grounded_number_paths(arguments, text, system) if flagged else set()
    for name in flagged:
        path = name.split(".", 1)[-1]
        if path in grounded:
            continue
        if path not in checked or path in failures:
            failures.add(path)
    if validation.get("negation"):
        failures.add("negated request")
    if failures:
        detail = ", ".join(sorted(failures))
        raise ExtractionValidationError(
            f"extraction returned values not grounded in the input: {detail}")

