"""Diagnose Cactus tool-calling format vs official chat_template.jinja for gemma-4-E2B-it."""
import json
import difflib
import re
from transformers import AutoTokenizer

MODEL_PATH = "/workspace/model/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)

# A representative tool-calling scenario
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Paris?"},
]
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "description": "Temperature units", "enum": ["C", "F"]},
            },
            "required": ["city"],
        },
    },
}]

# === Ground truth: official chat template ===
official = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)

# === Cactus simulation (mirror C++ format_gemma4_style + gemma::format_tools) ===
def cactus_escape(s):
    return "<escape>" + s + "<escape>"

def cactus_format_parameters(properties: dict, required: list):
    standard = {"description","type","properties","required","nullable"}
    result = ""
    first = True
    # C++ uses std::map -> sorted by key. Match dictsort.
    for key in sorted(properties.keys()):
        if key in standard:
            continue
        if not first:
            result += ","
        first = False
        prop = properties[key]
        result += f"{key}:{{"
        parts = []
        if "description" in prop:
            parts.append("description:" + cactus_escape(prop["description"]))
        type_val = prop.get("type", "")
        if type_val.upper() == "STRING":
            if "enum" in prop:
                # format_argument with escape_keys=True for strings -> escape() each
                items = "[" + ",".join(cactus_escape(x) for x in prop["enum"]) + "]"
                parts.append("enum:" + items)
        elif type_val.upper() == "ARRAY":
            if "items" in prop and isinstance(prop["items"], dict):
                pass  # complex; skip for our test case
        elif type_val.upper() == "OBJECT":
            if "properties" in prop:
                nested_req = prop.get("required", [])
                parts.append("properties:{" + cactus_format_parameters(prop["properties"], nested_req) + "}")
            if "required" in prop:
                req_items = ",".join(cactus_escape(x) for x in prop["required"])
                if req_items:
                    parts.append(f"required:[{req_items}]")
        if type_val:
            parts.append("type:" + cactus_escape(type_val.upper()))
        result += ",".join(parts)
        result += "}"
    return result

def cactus_format_function_declaration(name, description, params):
    result = "declaration:" + name + "{"
    result += "description:" + cactus_escape(description)
    if params:
        result += ",parameters:{"
        if "properties" in params:
            result += "properties:{" + cactus_format_parameters(params["properties"], params.get("required", [])) + "}"
        if "required" in params:
            req = ",".join(cactus_escape(x) for x in params["required"])
            if req:
                result += ",required:[" + req + "]"
        if "type" in params:
            result += ",type:" + cactus_escape(params["type"].upper())
        result += "}"
    result += "}"
    return result

def cactus_format_tools(tools, use_pipe_tags=True):
    if not tools:
        return ""
    decl_start = "<|tool>" if use_pipe_tags else "<start_function_declaration>"
    decl_end   = "<tool|>" if use_pipe_tags else "<end_function_declaration>"
    out = ""
    for t in tools:
        f = t["function"]
        out += decl_start
        out += cactus_format_function_declaration(f["name"], f["description"], f.get("parameters", {}))
        out += decl_end
    return out

def cactus_format_chat(messages, add_gen, tools_json, enable_thinking=False):
    result = "<bos>"
    sys_content = ""
    first_msg = 0
    if messages and messages[0]["role"] in ("system", "developer"):
        sys_content = messages[0]["content"]  # NOT trimmed in C++
        first_msg = 1
    if enable_thinking or sys_content or tools_json:
        result += "<|turn>system\n"
        if enable_thinking:
            result += "<|think|>"  # C++ does NOT add \n
        result += sys_content
        result += tools_json
        result += "<turn|>\n"
    for m in messages[first_msg:]:
        role = "model" if m["role"] == "assistant" else m["role"]
        result += f"<|turn>{role}\n"
        if role == "model":
            result += m["content"]
        else:
            result += m["content"]
        result += "<turn|>\n"
    if add_gen:
        result += "<|turn>model\n"
    return result

cactus_tools = cactus_format_tools(tools, True)
cactus = cactus_format_chat(messages, True, cactus_tools, False)

print("=" * 80)
print("OFFICIAL prompt:")
print("=" * 80)
print(repr(official))
print()
print("=" * 80)
print("CACTUS prompt:")
print("=" * 80)
print(repr(cactus))
print()

print("=" * 80)
print("UNIFIED DIFF (line-based, after splitting on <|tool> boundaries):")
print("=" * 80)
def split_visible(s):
    # split into lines+special-token chunks for readability
    parts = re.split(r'(<\|?[a-zA-Z_"]+\|?>)', s)
    return [p for p in parts if p]
o_parts = split_visible(official)
c_parts = split_visible(cactus)
for line in difflib.unified_diff(o_parts, c_parts, lineterm="", n=3):
    print(line)
print()

# === Token-level diff ===
print("=" * 80)
print("TOKEN ID DIFF:")
print("=" * 80)
o_ids = tok.encode(official, add_special_tokens=False)
c_ids = tok.encode(cactus, add_special_tokens=False)
print(f"Official: {len(o_ids)} tokens; Cactus: {len(c_ids)} tokens")
o_str = [tok.decode([i]) for i in o_ids]
c_str = [tok.decode([i]) for i in c_ids]
for line in difflib.unified_diff(
    [f"{i:>6}  {repr(s)}" for i,s in zip(o_ids, o_str)],
    [f"{i:>6}  {repr(s)}" for i,s in zip(c_ids, c_str)],
    lineterm="", n=2,
):
    print(line)

# === Step 2: Special-token tokenization sanity ===
print()
print("=" * 80)
print("STEP 2: Special-token tokenization (each should be exactly 1 token)")
print("=" * 80)
for s in ["<bos>", "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
          "<|tool_response>", "<tool_response|>", "<|turn>", "<turn|>",
          "<|think|>", "<|channel>", "<channel|>", '<|"|>', "<escape>"]:
    ids = tok.encode(s, add_special_tokens=False)
    flag = "  OK" if len(ids) == 1 else "  *** MULTI-TOKEN ***"
    print(f"  {s!r:30s} -> {ids}  {flag}")
