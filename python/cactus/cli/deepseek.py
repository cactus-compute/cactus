import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from .common import BLUE, GREEN, RED, YELLOW, PROJECT_ROOT, print_color


DEFAULT_DEEPSEEK_TOKENIZER = "deepseek-ai/DeepSeek-V4-Flash"


def looks_like_deepseek_bundle(path: Path) -> bool:
    if not (
        path.is_dir()
        and (path / "weights_manifest.json").exists()
        and ((path / "config.json").exists() or (path / "config.txt").exists())
        and not (path / "components" / "manifest.json").exists()
    ):
        return False
    try:
        cfg_path = path / "config.json" if (path / "config.json").exists() else path / "config.txt"
        text = cfg_path.read_text(errors="ignore").lower()
        if "deepseek_v4" in text or "deepseek-v4" in text or "deepseek_v4_flash" in text:
            return True
        manifest = json.loads((path / "weights_manifest.json").read_text())
        blob = json.dumps(manifest)[:200000].lower()
        return "hc_head" in blob and "attn_hc" in blob and "self_attn.compressor" in blob
    except Exception:
        return False


def _find_deepseek_stream_binary() -> Path | None:
    env_bin = os.getenv("CACTUS_DEEPSEEK_STREAM_BIN", "").strip()
    candidates = []
    if env_bin:
        candidates.append(Path(env_bin).expanduser())
    candidates.extend([
        Path(__file__).resolve().parent.parent / "bin" / "deepseek_stream",
        PROJECT_ROOT / "cactus-engine" / "build" / "deepseek_stream",
        PROJECT_ROOT / "build" / "deepseek_stream",
        Path("/tmp/cactus-engine-build/deepseek_stream"),
    ])
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _ensure_deepseek_stream_binary() -> Path:
    binary = _find_deepseek_stream_binary()
    if binary:
        return binary
    build_dir = Path("/tmp/cactus-engine-build")
    subprocess.run(["cmake", "-S", str(PROJECT_ROOT / "cactus-engine"), "-B", str(build_dir)], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "deepseek_stream", "-j", "8"], check=True)
    binary = build_dir / "deepseek_stream"
    if not binary.exists():
        raise RuntimeError("failed to build deepseek_stream")
    return binary


def _load_tokenizer(model_dir: Path, tokenizer_source: str | None):
    try:
        from transformers import AutoTokenizer
        from transformers.utils import logging as transformers_logging
    except Exception as exc:
        raise RuntimeError("transformers is required for DeepSeek tokenization") from exc
    transformers_logging.set_verbosity_error()
    source = tokenizer_source or (str(model_dir) if (model_dir / "tokenizer_config.json").exists() else DEFAULT_DEEPSEEK_TOKENIZER)
    return AutoTokenizer.from_pretrained(source, trust_remote_code=True)


def _token_id(tokenizer, text: str) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(text)
    if isinstance(token_id, int) and token_id >= 0:
        return token_id
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def _build_prompt_ids(tokenizer, messages) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(ids, Mapping):
            ids = ids.get("input_ids", [])
        return [int(x) for x in ids]
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=True)]


def _parse_kv_line(line: str) -> tuple[str, dict[str, str]]:
    parts = line.strip().split()
    fields = {}
    for item in parts[1:]:
        if "=" in item:
            k, v = item.split("=", 1)
            fields[k] = v
    return (parts[0] if parts else ""), fields


def _stderr_worker(stream, verbose: bool, out: queue.Queue[str]):
    for line in iter(stream.readline, ""):
        if verbose:
            sys.stderr.write(line)
            sys.stderr.flush()
        out.put(line)


def _run_once(binary: Path, model_dir: Path, tokenizer, messages, args) -> str:
    prompt_ids = _build_prompt_ids(tokenizer, messages)
    max_new = int(getattr(args, "max_new_tokens", None) or 128)
    context = int(getattr(args, "context", None) or max(2048, len(prompt_ids) + max_new + 16))
    temperature = float(getattr(args, "temperature", 0.0))
    top_p = float(getattr(args, "top_p", 1.0))
    top_k = int(getattr(args, "top_k", 1))

    stop_ids = set()
    for token in ("<｜end▁of▁sentence｜>", "<|im_end|>", "</s>"):
        token_id = _token_id(tokenizer, token)
        if token_id is not None:
            stop_ids.add(token_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        stop_ids.add(int(eos_id))

    cmd = [
        str(binary),
        "--model", str(model_dir),
        "--ids", ",".join(str(i) for i in prompt_ids),
        "--max-new-tokens", str(max_new),
        "--context", str(context),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--stop-ids", ",".join(str(i) for i in sorted(stop_ids)),
    ]
    if getattr(args, "kimi_warmup_moe_experts", False):
        cmd.append("--warmup-moe-experts")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    stderr_lines: queue.Queue[str] = queue.Queue()
    threading.Thread(
        target=_stderr_worker,
        args=(proc.stderr, bool(getattr(args, "verbose", False)), stderr_lines),
        daemon=True,
    ).start()

    pieces: list[str] = []
    token_count = 0
    ready = False
    stats = {}
    started = time.monotonic()
    assert proc.stdout is not None
    for line in proc.stdout:
        event, fields = _parse_kv_line(line)
        if event == "READY":
            ready = True
            print_color(BLUE, f"[loaded in {float(fields.get('init_ms', '0')) / 1000.0:.2f}s | {fields.get('mode', 'native')}]")
            print("Assistant: ", end="", flush=True)
        elif event == "WARMUP":
            print_color(BLUE, f"[moe expert prefetch in {float(fields.get('moe_expert_prefetch_ms', '0')) / 1000.0:.2f}s]")
            print("Assistant: ", end="", flush=True)
        elif event == "TOKEN":
            token_id = int(fields["id"])
            text = tokenizer.decode([token_id], skip_special_tokens=True)
            if text:
                print(text, end="", flush=True)
                pieces.append(text)
            token_count += 1
        elif event == "DONE":
            stats = fields

    rc = proc.wait()
    if rc != 0:
        buffered = []
        while not stderr_lines.empty():
            buffered.append(stderr_lines.get_nowait())
        if not bool(getattr(args, "verbose", False)) and buffered:
            sys.stderr.write("".join(buffered[-30:]))
        raise RuntimeError(f"deepseek_stream failed with exit code {rc}")
    if not ready:
        raise RuntimeError("deepseek_stream exited before reporting READY")

    elapsed = time.monotonic() - started
    print()
    print_color(
        GREEN,
        f"[{token_count} tokens | TTFT {float(stats.get('ttft_ms', '0')) / 1000.0:.2f}s | "
        f"total {float(stats.get('total_ms', '0')) / 1000.0:.2f}s | "
        f"decode {float(stats.get('decode_tps', '0')):.2f} tok/s | wall {elapsed:.2f}s | "
        f"stopped={stats.get('stopped', '0') == '1'}]",
    )
    return "".join(pieces)


def cmd_run_deepseek(args) -> int:
    model_dir = Path(args.model_id).expanduser().resolve()
    if not looks_like_deepseek_bundle(model_dir):
        print_color(RED, f"Not a DeepSeek V4 weight bundle: {model_dir}")
        return 1
    try:
        binary = _ensure_deepseek_stream_binary()
        tokenizer = _load_tokenizer(model_dir, getattr(args, "tokenizer", None))
    except Exception as exc:
        print_color(RED, f"DeepSeek setup failed: {exc}")
        return 1

    print_color(GREEN, f"Starting DeepSeek V4 native stream with model: {model_dir}")
    print_color(YELLOW, "Current DeepSeek runner uses full-prefix recompute per generated token; decode cache optimization is separate.")
    print()

    messages = []
    if getattr(args, "system", None):
        messages.append({"role": "system", "content": args.system})
    initial = getattr(args, "prompt", None)
    if initial:
        messages.append({"role": "user", "content": initial})
        try:
            assistant = _run_once(binary, model_dir, tokenizer, messages, args)
        except KeyboardInterrupt:
            print()
            return 130
        except Exception as exc:
            print_color(RED, f"DeepSeek run failed: {exc}")
            return 1
        messages.append({"role": "assistant", "content": assistant})
        return 0

    print("Commands: /reset, /exit")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/exit", "exit", "quit"}:
            break
        if user == "/reset":
            messages = [{"role": "system", "content": args.system}] if getattr(args, "system", None) else []
            print("Conversation reset.")
            continue
        messages.append({"role": "user", "content": user})
        try:
            assistant = _run_once(binary, model_dir, tokenizer, messages, args)
        except KeyboardInterrupt:
            print()
            return 130
        except Exception as exc:
            print_color(RED, f"DeepSeek run failed: {exc}")
            messages.pop()
            return 1
        messages.append({"role": "assistant", "content": assistant})
        print()
    return 0
