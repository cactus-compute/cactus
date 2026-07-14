import json
from pathlib import Path

from .common import GREEN, RED, apply_runtime_env, launch_binary, print_color

NEEDLE_DEFAULT_TOOLS = [
    {"type": "function", "function": {
        "name": "set_timer",
        "description": "Set a timer for the specified duration or end time.",
        "parameters": {"type": "object", "properties": {
            "time_human": {"type": "string", "description": "The duration or target end time in human readable format e.g. '1 hour and 30 minutes', '45 minutes', 'at 13:30'."},
        }, "required": ["time_human"]},
    }},
    {"type": "function", "function": {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "description": "The recipient's email address."},
            "subject": {"type": "string", "description": "The email subject line."},
            "body": {"type": "string", "description": "The email body text."},
        }, "required": ["to", "subject", "body"]},
    }},
    {"type": "function", "function": {
        "name": "create_note",
        "description": "Create a new note with the given text.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "The text of the note."},
            "title": {"type": "string", "description": "Optional title for the note."},
        }, "required": ["text"]},
    }},
]


def resolve_tools(spec):
    """Validate OpenAI function-calling tools from inline JSON or a file path; return compact JSON."""
    if not spec.lstrip().startswith(("[", "{")):
        spec = Path(spec).expanduser().read_text()
    tools = json.loads(spec)
    if not isinstance(tools, list):
        raise ValueError("tools must be a JSON array")
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        params = (fn.get("parameters") or {}) if isinstance(fn, dict) else {}
        if not (isinstance(fn, dict) and tool.get("type") == "function" and fn.get("name")
                and isinstance(params, dict)
                and (not params or (params.get("type") == "object"
                                    and isinstance(params.get("properties"), dict)))):
            raise ValueError(
                'each tool must be {"type": "function", "function": {"name": ..., "description": ..., '
                '"parameters": {"type": "object", "properties": {...}, "required": [...]}}}')
    return json.dumps(tools, separators=(",", ":"), ensure_ascii=False)


def bundle_model_type(bundle_dir):
    from .list import _read_config
    config = Path(bundle_dir) / "config.txt"
    return _read_config(config).get("model_type") if config.is_file() else None


def cmd_run(args) -> int:
    apply_runtime_env(args)

    if args.image:
        args.image = str(Path(args.image).expanduser())
        if not Path(args.image).is_file():
            print_color(RED, f"Image not found: {args.image}")
            return 1
    if args.audio:
        args.audio = str(Path(args.audio).expanduser())
        if not Path(args.audio).is_file():
            print_color(RED, f"Audio not found: {args.audio}")
            return 1
    if args.result_json:
        args.result_json = str(Path(args.result_json).expanduser())
    if args.tools:
        try:
            args.tools = resolve_tools(args.tools)
        except (ValueError, OSError) as exc:
            print_color(RED, f"Invalid --tools: {exc}")
            return 1

    from .model import TranspileOptions, prepare_bundle

    bundle_dir = prepare_bundle(args, transpile=TranspileOptions(
        image_files=[args.image] if args.image else None,
        audio_file=args.audio,
    ))
    if bundle_dir is None:
        return 1

    if args.tools is None and bundle_model_type(bundle_dir) == "needle":
        args.tools = json.dumps(NEEDLE_DEFAULT_TOOLS, separators=(",", ":"))
        print("Using the demo toolset; pass --tools <file> to use your own.")

    cmd = [str(bundle_dir)]
    for flag, value in (
        ("--system", args.system),
        ("--prompt", args.prompt),
        ("--tools", args.tools),
        ("--image", args.image),
        ("--audio", args.audio),
        ("--input-ids", args.input_ids),
        ("--input-ids-file", args.input_ids_file),
        ("--max-new-tokens", args.max_new_tokens),
        ("--result-json", args.result_json),
        ("--confidence-threshold", args.confidence_threshold),
        ("--cloud-timeout-ms", args.cloud_timeout_ms),
        ("--backend", args.backend),
    ):
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.thinking:
        cmd.append("--thinking")
    if args.no_cloud_handoff:
        cmd.append("--no-cloud-handoff")

    print_color(GREEN, f"Running: {bundle_dir}")
    print()
    return launch_binary("run", *cmd)
