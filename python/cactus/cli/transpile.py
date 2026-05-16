from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .common import (
    DEFAULT_MODEL_ID,
    PROJECT_ROOT,
    GREEN,
    RED,
    YELLOW,
    print_color,
)
from .download import get_weights_dir


MODEL_ID_ALIASES = {
    "gemma4": DEFAULT_MODEL_ID,
    "gemma4-e2b": DEFAULT_MODEL_ID,
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "parakeet-tdt": "nvidia/parakeet-tdt-0.6b-v3",
    "whisper": "openai/whisper-small",
    "qwen": "Qwen/Qwen3-1.7B",
    "lfm": "LiquidAI/LFM2-VL-450M",
}


def resolve_model_id_alias(model_id: str) -> str:
    normalized = (model_id or "").strip()
    return MODEL_ID_ALIASES.get(normalized.lower(), normalized)


def _python_runtime_library_path() -> Path:
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return PROJECT_ROOT / "cactus" / "build" / f"libcactus{suffix}"


def _static_cactus_library_path() -> Path:
    return PROJECT_ROOT / "cactus" / "build" / "libcactus.a"


def _build_static_cactus_library() -> Path:
    build_script = PROJECT_ROOT / "cactus" / "build.sh"
    if not build_script.exists():
        raise RuntimeError(f"The Cactus build script is missing: {build_script}")

    build = subprocess.run([str(build_script)], cwd=PROJECT_ROOT / "cactus")
    if build.returncode != 0:
        raise RuntimeError("Failed to build the Cactus static runtime")

    static_library_path = _static_cactus_library_path()
    if not static_library_path.exists():
        raise RuntimeError(
            "The Cactus build completed, but the static library was not produced.\n"
            f"Expected: {static_library_path}"
        )
    return static_library_path


def _public_cactus_api_symbols(static_library_path: Path) -> list[str]:
    if platform.system() == "Darwin":
        command = ["nm", "-gU", str(static_library_path)]
    else:
        command = ["nm", "-g", "--defined-only", str(static_library_path)]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to inspect the Cactus static runtime symbols.\n"
            f"Command: {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )

    symbols: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        symbol = parts[-1].strip()
        normalized = symbol[1:] if symbol.startswith("_") else symbol
        if normalized.startswith("cactus_") and symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        raise RuntimeError(f"Could not find any public cactus_* symbols in {static_library_path}")
    return symbols


def _link_python_runtime_library(*, static_library_path: Path, library_path: Path) -> None:
    build_dir = library_path.parent
    build_dir.mkdir(parents=True, exist_ok=True)
    if library_path.exists():
        library_path.unlink()

    exported_symbols = _public_cactus_api_symbols(static_library_path)
    if platform.system() == "Darwin":
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            raise RuntimeError("Failed to find a C++ compiler for linking libcactus.dylib")
        command = [
            compiler,
            "-dynamiclib",
            "-o",
            str(library_path),
            *[f"-Wl,-u,{symbol}" for symbol in exported_symbols],
            str(static_library_path),
            "-Wl,-install_name,@rpath/libcactus.dylib",
            "-lcurl",
            "-framework",
            "Accelerate",
            "-framework",
            "CoreML",
            "-framework",
            "Foundation",
            "-framework",
            "Security",
            "-framework",
            "SystemConfiguration",
            "-framework",
            "CFNetwork",
        ]
    else:
        compiler = shutil.which("g++") or shutil.which("c++")
        if not compiler:
            raise RuntimeError("Failed to find a C++ compiler for linking libcactus.so")
        command = [
            compiler,
            "-shared",
            "-o",
            str(library_path),
            *[f"-Wl,--undefined={symbol}" for symbol in exported_symbols],
            str(static_library_path),
            "-lcurl",
            "-pthread",
            "-ldl",
            "-lm",
        ]

    result = subprocess.run(command, cwd=build_dir)
    if result.returncode != 0 or not library_path.exists():
        raise RuntimeError(f"Failed to link the Cactus shared runtime: {library_path}")


def _ensure_python_runtime_library() -> Path:
    library_path = _python_runtime_library_path()
    static_library_path = _static_cactus_library_path()
    if (
        library_path.exists()
        and static_library_path.exists()
        and library_path.stat().st_mtime >= static_library_path.stat().st_mtime
    ):
        return library_path

    print_color(YELLOW, "Preparing Cactus shared runtime for transpiler...")
    if not static_library_path.exists():
        static_library_path = _build_static_cactus_library()
    _link_python_runtime_library(static_library_path=static_library_path, library_path=library_path)
    return library_path


def _weights_dir_looks_transpile_ready(weights_dir: Path) -> bool:
    root = Path(weights_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return False
    if (root / "weights_manifest.json").exists():
        return True
    if any(root.glob("*.cq[1-4].weights")):
        return True
    # Legacy converted Cactus weight folders may not have a manifest yet, but
    # they should still carry runtime config plus Cactus tensor files.
    return (root / "config.txt").exists() and any(root.glob("*.weights"))


def _extra_args_has_option(extra_args: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in extra_args)


def _resolve_transpiled_manifest(path_value: str | os.PathLike[str] | None) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value).expanduser().resolve()
    if not candidate.exists():
        return None
    if candidate.is_file() and candidate.name == "manifest.json":
        return candidate
    for manifest in (
        candidate / "components" / "manifest.json",
        candidate / "manifest.json",
    ):
        if manifest.exists():
            return manifest
    return None


def _prepend_python_path(env: dict[str, str]) -> None:
    python_root = str(PROJECT_ROOT / "python")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = python_root if not existing else f"{python_root}{os.pathsep}{existing}"


def cmd_transpile(args) -> int:
    """Invoke the saved-model transpiler entrypoint."""
    model_id = resolve_model_id_alias(args.model_id)
    extra_args = list(getattr(args, "extra_args", []) or [])
    allow_unconverted = bool(getattr(args, "allow_unconverted_weights", False))

    command = [sys.executable, "-m", "cactus.transpile.hf_model", "--model-id", model_id]
    if _extra_args_has_option(extra_args, "--weights-dir"):
        pass
    else:
        default_weights_dir = get_weights_dir(model_id)
        if _weights_dir_looks_transpile_ready(default_weights_dir):
            command.extend(["--weights-dir", str(default_weights_dir)])
        elif not allow_unconverted:
            print_color(
                RED,
                "Error: transpilation requires converted Cactus CQ weights.",
            )
            print_color(
                YELLOW,
                "Run conversion first, then retry:\n"
                f"  cactus convert {model_id} {default_weights_dir} --bits 4\n"
                f"  cactus transpile {model_id} --weights-dir {default_weights_dir}",
            )
            return 1

    if allow_unconverted:
        command.append("--allow-unconverted-weights")
    if not getattr(args, "execute_after_transpile", False) and "--skip-execute" not in extra_args:
        command.append("--skip-execute")
    command.extend(extra_args)

    try:
        transpile_lib = _ensure_python_runtime_library()
    except RuntimeError as exc:
        print_color(RED, f"Error: {exc}")
        return 1

    env = os.environ.copy()
    env["CACTUS_LIB_PATH"] = str(transpile_lib)
    _prepend_python_path(env)
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return result.returncode


def cmd_run_transpiled(args) -> int:
    """Run a saved transpiled component bundle."""
    try:
        transpile_lib = _ensure_python_runtime_library()
    except RuntimeError as exc:
        print_color(RED, f"Error: {exc}")
        return 1

    os.environ["CACTUS_LIB_PATH"] = str(transpile_lib)
    python_root = str(PROJECT_ROOT / "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)

    from cactus.transpile.component_bundle_runtime import run_transpiled_bundle

    if getattr(args, "_transpiled_from_run", False):
        return _run_transpiled_interactive_chat(args, run_transpiled_bundle)

    bundle_dir = getattr(args, "bundle_dir", None) or getattr(args, "model_id", None)
    result = _run_transpiled_once(args, run_transpiled_bundle, bundle_dir=bundle_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if getattr(args, "result_json", None):
        result_path = Path(args.result_json).expanduser().resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print_color(GREEN, f"Saved result to {result_path}")
    return 0


def _run_transpiled_once(args, run_transpiled_bundle, *, bundle_dir: str | os.PathLike[str] | None) -> dict[str, object]:
    image_values: list[str] = []
    seen_images: set[str] = set()

    def add_image(value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        normalized = str(Path(value.strip()).expanduser().resolve())
        if normalized in seen_images:
            return
        seen_images.add(normalized)
        image_values.append(normalized)

    for attr_name in ("image_file", "image_files"):
        value = getattr(args, attr_name, None)
        if isinstance(value, str) and value.strip():
            add_image(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                add_image(str(item))
    image_arg = getattr(args, "image", None)
    if isinstance(image_arg, str) and image_arg.strip():
        add_image(image_arg)

    return run_transpiled_bundle(
        bundle_dir,
        audio_file=getattr(args, "audio_file", None) or getattr(args, "audio", None),
        image_files=tuple(image_values),
        prompt=getattr(args, "prompt", None),
        input_ids=getattr(args, "input_ids", None),
        weights_dir=getattr(args, "weights_dir", None),
        system_prompt=getattr(args, "system", None),
        enable_thinking=bool(getattr(args, "thinking", False)),
        max_new_tokens=getattr(args, "max_new_tokens", None),
        stop_sequences=tuple(getattr(args, "stop_sequence", []) or ()),
    )


def _run_transpiled_interactive_chat(args, run_transpiled_bundle) -> int:
    bundle_dir = getattr(args, "bundle_dir", None) or getattr(args, "model_id", None)
    print(f"Loading model from {bundle_dir}...")
    try:
        from cactus.transpile.component_bundle_runtime import _default_weights_dir_for_manifest
        from cactus.transpile.component_bundle_runtime import load_component_bundle_manifest
        from cactus.transpile.component_bundle_runtime import load_saved_component_graphs
        from cactus.transpile.component_bundle_runtime import runtime_include_components_for_manifest

        _, manifest = load_component_bundle_manifest(bundle_dir)
        resolved_weights_dir = _default_weights_dir_for_manifest(
            manifest,
            explicit=getattr(args, "weights_dir", None),
        )
        include_components = runtime_include_components_for_manifest(
            family=str(manifest.get("family", "") or ""),
            task=str(manifest.get("task", "") or ""),
            manifest=manifest,
        )
        if (
            str(manifest.get("family", "") or "").strip().lower() == "gemma4"
            and str(manifest.get("task", "") or "") == "multimodal_causal_lm_logits"
            and not getattr(args, "image", None)
            and not (getattr(args, "image_file", None) or [])
            and not (getattr(args, "audio_file", None) or getattr(args, "audio", None))
        ):
            manifest_components = {
                str(component_entry.get("component", "")).strip()
                for component_entry in manifest.get("components", [])
                if isinstance(component_entry, dict)
            }
            if {"lm_encoder_step", "decoder_step"}.issubset(manifest_components):
                include_components = {"lm_encoder_step", "decoder_step"}
        load_saved_component_graphs(
            bundle_dir,
            weights_dir=resolved_weights_dir,
            include_components=include_components,
        )
    except Exception as exc:
        print(f"Warning: deferred graph load until first turn ({exc})", file=sys.stderr)
    print("Model loaded.")
    print("Commands: /image <path> [prompt], /audio <path> [prompt], /clear, reset, exit\n")

    history: list[tuple[str, str]] = []
    current_image = getattr(args, "image", None)
    image_files = list(getattr(args, "image_file", []) or [])
    if current_image:
        image_files.append(str(current_image))
    current_image = image_files[-1] if image_files else None
    current_audio = getattr(args, "audio_file", None) or getattr(args, "audio", None)
    initial_prompt = getattr(args, "prompt", None)
    auto_send = bool(initial_prompt or current_image or current_audio)

    while True:
        if auto_send:
            auto_send = False
            user_input = initial_prompt or "Describe the attached input."
            print(f"You: {user_input}")
        else:
            try:
                user_input = input("You: ")
            except EOFError:
                break

        user_input = user_input.rstrip(" \t")
        if not user_input:
            continue
        if user_input in {"exit", "quit"}:
            break
        if user_input == "reset":
            history.clear()
            current_image = None
            current_audio = None
            print("Conversation reset.")
            continue
        if user_input == "/clear":
            current_image = None
            current_audio = None
            print("Attachments cleared.")
            continue

        parsed = _parse_transpiled_attachment_command(user_input, "/image ")
        if parsed is not None:
            path_value, remainder = parsed
            if not Path(path_value).expanduser().exists():
                print(f"File not found: {path_value}", file=sys.stderr)
                continue
            current_image = str(Path(path_value).expanduser().resolve())
            if not remainder:
                print(f"Image attached: {current_image}")
                continue
            user_input = remainder

        parsed = _parse_transpiled_attachment_command(user_input, "/audio ")
        if parsed is not None:
            path_value, remainder = parsed
            if not Path(path_value).expanduser().exists():
                print(f"File not found: {path_value}", file=sys.stderr)
                continue
            current_audio = str(Path(path_value).expanduser().resolve())
            if not remainder:
                print(f"Audio attached: {current_audio}")
                continue
            user_input = remainder

        if current_image:
            print(f"[image: {current_image}]")
        if current_audio:
            print(f"[audio: {current_audio}]")
        print("Assistant: ", end="", flush=True)

        # Text-only turns use the full chat history through the cached decode path.
        # Multimodal graphs are statically shaped, so keep media turns focused on
        # the current attachment instead of replaying a long text transcript.
        prompt_history = _history_for_transpiled_turn(
            history,
            has_media=bool(current_image or current_audio),
        )
        prompt = _build_transpiled_interactive_prompt(
            system_prompt=getattr(args, "system", None),
            history=prompt_history,
            user_input=user_input,
        )
        turn_args = _copy_namespace_for_transpiled_turn(
            args,
            prompt=prompt,
            image=current_image,
            audio=current_audio,
        )

        started = time.perf_counter()
        try:
            result = _run_transpiled_once(turn_args, run_transpiled_bundle, bundle_dir=bundle_dir)
        except Exception as exc:
            print()
            print(f"Error: {exc}", file=sys.stderr)
            continue
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)

        response_text = _transpiled_response_text(result)
        if response_text:
            print(response_text)
        else:
            print(json.dumps(result, indent=2, sort_keys=True))

        _print_transpiled_interactive_stats(result, elapsed_ms=elapsed_ms)
        print()

        history.append(("user", user_input))
        if response_text:
            history.append(("assistant", response_text))
        current_image = None
        current_audio = None

    print("Goodbye.")
    return 0


def _copy_namespace_for_transpiled_turn(args, *, prompt: str, image: str | None, audio: str | None):
    import argparse

    values = vars(args).copy()
    values["prompt"] = prompt
    values["image"] = image
    values["image_file"] = [image] if image else []
    values["audio"] = audio
    values["audio_file"] = audio
    return argparse.Namespace(**values)


def _parse_transpiled_attachment_command(text: str, prefix: str) -> tuple[str, str] | None:
    if not text.startswith(prefix):
        return None
    rest = text[len(prefix):].strip()
    if not rest:
        return "", ""
    split = rest.find(" ")
    if split < 0:
        return rest, ""
    return rest[:split], rest[split + 1 :].strip()


def _history_for_transpiled_turn(
    history: list[tuple[str, str]],
    *,
    has_media: bool,
) -> list[tuple[str, str]]:
    return [] if has_media else history


def _build_transpiled_interactive_prompt(
    *,
    system_prompt: str | None,
    history: list[tuple[str, str]],
    user_input: str,
) -> str:
    if not system_prompt and not history:
        return user_input

    turns: list[str] = []
    if system_prompt:
        turns.append(f"System: {system_prompt}")
    for role, content in history:
        label = "User" if role == "user" else "Assistant"
        turns.append(f"{label}: {content}")
    turns.append(f"User: {user_input}")
    turns.append("Assistant:")
    return "\n".join(turns)


def _transpiled_response_text(result: dict[str, object]) -> str:
    for key in ("response", "transcript"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    next_token = result.get("next_token")
    if isinstance(next_token, str) and next_token:
        return next_token
    return ""


def _print_transpiled_interactive_stats(result: dict[str, object], *, elapsed_ms: float) -> None:
    token_count = int(
        result.get("decode_tokens", 0)
        or result.get("decoder_steps", 0)
        or len(result.get("generated_token_ids", []) or [])
        or len(result.get("token_ids", []) or [])
    )
    if token_count <= 0:
        token_count = 1 if _transpiled_response_text(result) else 0
    ttft_s = float(result.get("time_to_first_token_ms", 0.0) or 0.0) / 1000.0
    total_s = float(result.get("total_ms", elapsed_ms) or elapsed_ms) / 1000.0
    tps = float(result.get("decode_tps", 0.0) or 0.0)
    if tps <= 0.0 and total_s > 0.0:
        tps = token_count / total_s
    print(
        f"[{token_count} tokens | latency: {ttft_s:.3f}s | "
        f"total: {total_s:.3f}s | {tps:.1f} tok/s]"
    )


def _print_transpiled_run_result(result: dict[str, object]) -> None:
    response_text = _transpiled_response_text(result)
    if response_text:
        print(response_text)
        return

    print(json.dumps(result, indent=2, sort_keys=True))
