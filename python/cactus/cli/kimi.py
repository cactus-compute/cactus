import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from .common import BLUE, GREEN, RED, YELLOW, PROJECT_ROOT, print_color


DEFAULT_KIMI_TOKENIZER = "moonshotai/Kimi-K2.6"


def looks_like_kimi_bundle(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "weights_manifest.json").exists()
        and ((path / "config.json").exists() or (path / "config.txt").exists())
        and not (path / "components" / "manifest.json").exists()
    )


def _find_kimi_stream_binary() -> Path | None:
    env_bin = os.getenv("CACTUS_KIMI_STREAM_BIN", "").strip()
    candidates = []
    if env_bin:
        candidates.append(Path(env_bin).expanduser())
    candidates.extend([
        Path(__file__).resolve().parent.parent / "bin" / "kimi_stream",
        PROJECT_ROOT / "cactus-engine" / "build" / "kimi_stream",
        PROJECT_ROOT / "build" / "kimi_stream",
        Path("/tmp/cactus-engine-build/kimi_stream"),
    ])
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _ensure_kimi_stream_binary() -> Path:
    binary = _find_kimi_stream_binary()
    if binary:
        return binary
    if not (PROJECT_ROOT / "cactus-engine" / "CMakeLists.txt").exists():
        raise RuntimeError("kimi_stream binary not found; set CACTUS_KIMI_STREAM_BIN to a built runner")

    build_dir = Path("/tmp/cactus-engine-build")
    subprocess.run(
        ["cmake", "-S", str(PROJECT_ROOT / "cactus-engine"), "-B", str(build_dir)],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "kimi_stream", "-j", "8"], check=True)
    binary = build_dir / "kimi_stream"
    if not binary.exists():
        raise RuntimeError("failed to build kimi_stream")
    return binary


def _load_tokenizer(model_dir: Path, tokenizer_source: str | None):
    try:
        from transformers import AutoTokenizer
        from transformers.utils import logging as transformers_logging
    except Exception as exc:
        raise RuntimeError("transformers is required for Kimi tokenization") from exc
    transformers_logging.set_verbosity_error()

    if tokenizer_source:
        source = tokenizer_source
    elif (model_dir / "tiktoken.model").exists():
        source = str(model_dir)
    else:
        source = DEFAULT_KIMI_TOKENIZER
        cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--moonshotai--Kimi-K2.6" / "snapshots"
        if cache_root.exists():
            snapshots = sorted(
                (p for p in cache_root.iterdir() if (p / "tiktoken.model").exists()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                source = str(snapshots[0])
    return AutoTokenizer.from_pretrained(source, trust_remote_code=True)


def _token_id(tokenizer, text: str) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(text)
    if isinstance(token_id, int) and token_id >= 0:
        return token_id
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def _build_prompt_ids(tokenizer, messages, thinking: bool) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(ids, Mapping):
        ids = ids.get("input_ids", [])
    ids = [int(x) for x in ids]
    if not thinking:
        close_think = _token_id(tokenizer, "</think>")
        if close_think is not None and (not ids or ids[-1] != close_think):
            ids.append(close_think)
    return ids


def _parse_kv_line(line: str) -> tuple[str, dict[str, str]]:
    parts = line.strip().split()
    if not parts:
        return "", {}
    fields = {}
    for item in parts[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    return parts[0], fields


def _stderr_worker(stream, verbose: bool, out: queue.Queue[str]):
    for line in iter(stream.readline, ""):
        if verbose:
            sys.stderr.write(line)
            sys.stderr.flush()
        out.put(line)


def _run_once(binary: Path, model_dir: Path, tokenizer, messages, args) -> str:
    prompt_ids = _build_prompt_ids(tokenizer, messages, getattr(args, "thinking", False))
    max_new = int(getattr(args, "max_new_tokens", None) or 128)
    context = int(getattr(args, "context", None) or max(2048, len(prompt_ids) + max_new + 16))
    temperature = float(getattr(args, "temperature", 0.0))
    top_p = float(getattr(args, "top_p", 1.0))
    top_k = int(getattr(args, "top_k", 1))

    stop_ids = set()
    for token in ("<|im_end|>", "[EOS]"):
        token_id = _token_id(tokenizer, token)
        if token_id is not None:
            stop_ids.add(token_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        stop_ids.add(int(eos_id))

    env = os.environ.copy()
    if not getattr(args, "kimi_fast_kv", False):
        env["CACTUS_KV_CACHE_FP16"] = "1"
    else:
        env.pop("CACTUS_KV_CACHE_FP16", None)
    if getattr(args, "kimi_moe_prefetch", False):
        env["CACTUS_MOE_PREFETCH"] = "1"
    else:
        env.pop("CACTUS_MOE_PREFETCH", None)

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

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    stderr_lines: queue.Queue[str] = queue.Queue()
    thread = threading.Thread(
        target=_stderr_worker,
        args=(proc.stderr, bool(getattr(args, "verbose", False)), stderr_lines),
        daemon=True,
    )
    thread.start()

    pieces: list[str] = []
    token_count = 0
    ready = False
    started = time.monotonic()
    stats = {}
    assert proc.stdout is not None
    for line in proc.stdout:
        event, fields = _parse_kv_line(line)
        if event == "READY":
            ready = True
            init_ms = float(fields.get("init_ms", "0"))
            print_color(BLUE, f"[loaded in {init_ms / 1000.0:.2f}s]")
            if not getattr(args, "kimi_warmup_moe_experts", False):
                print("Assistant: ", end="", flush=True)
        elif event == "WARMUP":
            warmup_ms = float(fields.get("moe_expert_prefetch_ms", "0"))
            print_color(BLUE, f"[moe expert prefetch in {warmup_ms / 1000.0:.2f}s]")
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
            sys.stderr.write("".join(buffered[-20:]))
        raise RuntimeError(f"kimi_stream failed with exit code {rc}")

    if not ready:
        raise RuntimeError("kimi_stream exited before reporting READY")

    elapsed = time.monotonic() - started
    ttft_ms = float(stats.get("ttft_ms", "0"))
    total_ms = float(stats.get("total_ms", "0"))
    decode_tps = float(stats.get("decode_tps", "0"))
    stopped = stats.get("stopped", "0") == "1"
    print()
    print_color(
        GREEN,
        f"[{token_count} tokens | TTFT {ttft_ms / 1000.0:.2f}s | total {total_ms / 1000.0:.2f}s | "
        f"decode {decode_tps:.2f} tok/s | wall {elapsed:.2f}s | stopped={stopped}]",
    )
    return "".join(pieces)


def cmd_run_kimi(args) -> int:
    model_dir = Path(args.model_id).expanduser().resolve()
    if not looks_like_kimi_bundle(model_dir):
        print_color(RED, f"Not a Kimi weight bundle: {model_dir}")
        return 1

    try:
        binary = _ensure_kimi_stream_binary()
        tokenizer = _load_tokenizer(model_dir, getattr(args, "tokenizer", None))
    except Exception as exc:
        print_color(RED, f"Kimi setup failed: {exc}")
        return 1

    print_color(GREEN, f"Starting Kimi stream with model: {model_dir}")
    if not getattr(args, "kimi_fast_kv", False):
        print_color(YELLOW, "Using FP16 KV cache for correctness parity. Pass --kimi-fast-kv for faster approximate KV.")
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
            print_color(RED, f"Kimi run failed: {exc}")
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
            print_color(RED, f"Kimi run failed: {exc}")
            messages.pop()
            return 1
        messages.append({"role": "assistant", "content": assistant})
        print()
    return 0
