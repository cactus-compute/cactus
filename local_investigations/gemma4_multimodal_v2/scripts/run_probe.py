from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
INVESTIGATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_ROOT = REPO_ROOT.parent / "cactus" / "weights"
ASSET_DIR = REPO_ROOT / "cactus-engine" / "tests" / "assets"
TMP_DIR = REPO_ROOT / "local_integration_tests" / ".tmp"


def run(cmd: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CACTUS_NO_CLOUD_TELE"] = "1"
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def command_record(case: str, command: list[str], completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    parsed: Any = None
    if completed.stdout.strip().startswith("{"):
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError:
            parsed = None
    return {
        "case": case,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "parsed_stdout": parsed,
    }


def ensure_runner() -> Path:
    build_dir = TMP_DIR / "build"
    runner = build_dir / "integration_runner"
    sources = [
        REPO_ROOT / "local_integration_tests" / "native_runner.cpp",
        REPO_ROOT / "local_integration_tests" / "CMakeLists.txt",
        REPO_ROOT / "cactus" / "build" / "libcactus.a",
    ]
    if runner.exists() and all(runner.stat().st_mtime >= path.stat().st_mtime for path in sources if path.exists()):
        return runner
    cmake = run(
        [
            "cmake",
            "-S",
            str(REPO_ROOT / "local_integration_tests"),
            "-B",
            str(build_dir),
            f"-DCACTUS_REPO_ROOT={REPO_ROOT}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        timeout=120,
    )
    if cmake.returncode != 0:
        raise RuntimeError(cmake.stderr or cmake.stdout)
    build = run(["cmake", "--build", str(build_dir), "--target", "integration_runner", "-j", str(os.cpu_count() or 4)], timeout=120)
    if build.returncode != 0:
        raise RuntimeError(build.stderr or build.stdout)
    if not runner.exists():
        raise RuntimeError(f"integration runner was not built: {runner}")
    return runner


def ensure_question_wav() -> Path:
    wav = TMP_DIR / "what_is_in_this_image.wav"
    if wav.exists():
        return wav
    mp3 = REPO_ROOT.parent / "cactus" / "what_is_in_this_image.mp3"
    if not mp3.exists():
        raise RuntimeError(f"missing paired mp3: {mp3}")
    wav.parent.mkdir(parents=True, exist_ok=True)
    completed = run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(mp3),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(wav),
        ],
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return wav


def cactus_complete(model: Path, prompt: str, image: Path | None = None, audio: Path | None = None, max_tokens: int = 60) -> dict[str, Any]:
    runner = ensure_runner()
    command: list[str] = [
        str(runner),
        "complete",
        "--model",
        str(model),
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
    ]
    if image is not None:
        command.extend(["--image", str(image)])
    if audio is not None:
        command.extend(["--audio", str(audio)])
    return command_record("cactus_complete", command, run(command, timeout=360))


def native_json(command: list[str], *, timeout: int = 360) -> dict[str, Any]:
    completed = run(command, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"native command did not return JSON: {command}\n{completed.stdout[:1000]}") from exc


def parakeet_transcribe(model: Path, audio: Path, pcm: bool = False) -> dict[str, Any]:
    runner = ensure_runner()
    command = [str(runner), "transcribe-pcm" if pcm else "transcribe", "--model", str(model), "--audio", str(audio)]
    return command_record("parakeet_transcribe_pcm" if pcm else "parakeet_transcribe", command, run(command, timeout=240))


def hf_gemma_image(model_dir: Path, image: Path, prompt: str) -> dict[str, Any]:
    script = f"""
import json
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText

model_dir = Path({str(model_dir)!r})
image_path = Path({str(image)!r})
prompt = {prompt!r}
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(model_dir, local_files_only=True, dtype=torch.float16, device_map='auto')
messages = [{{'role': 'user', 'content': [{{'type': 'image', 'image': str(image_path)}}, {{'type': 'text', 'text': prompt}}]}}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
inputs = {{k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}}
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
text = processor.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
print(json.dumps({{'response': text, 'input_tokens': int(inputs['input_ids'].shape[-1])}}))
"""
    command = [sys.executable, "-c", script]
    return command_record("hf_gemma_image", command, run(command, timeout=360))


def hf_gemma_native_image(model_dir: Path, image: Path, prompt: str) -> dict[str, Any]:
    script = f"""
import json
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from cactus.transpile.multimodal_runtime import prepare_gemma4_multimodal_inputs

model_dir = Path({str(model_dir)!r})
image_path = Path({str(image)!r})
prompt = {prompt!r}
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
prepared = prepare_gemma4_multimodal_inputs(
    processor,
    prompt=prompt,
    image_files=(str(image_path),),
    audio_file=None,
    torch_dtype=torch.float16,
    use_gemma4_chat_template=True,
)
model = AutoModelForImageTextToText.from_pretrained(model_dir, local_files_only=True, dtype=torch.float16, device_map='auto')
inputs = {{}}
summaries = {{}}
for name, tensor in zip(prepared.names, prepared.tensors):
    key = 'image_position_ids' if name == 'pixel_position_ids' else name
    inputs[key] = tensor.to(model.device)
    summaries[name] = {{
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'min': float(tensor.float().min().item()) if tensor.numel() else 0.0,
        'max': float(tensor.float().max().item()) if tensor.numel() else 0.0,
        'mean': float(tensor.float().mean().item()) if tensor.numel() else 0.0,
    }}
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
text = processor.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
input_ids = prepared.tensors[prepared.names.index('input_ids')]
print(json.dumps({{
    'response': text,
    'input_tokens': int(input_ids.shape[-1]),
    'input_shapes': prepared.metadata.get('input_shapes'),
    'processor_prompt_prefix': str(prepared.metadata.get('processor_prompt'))[:240],
    'tensor_summaries': summaries,
}}))
"""
    command = [sys.executable, "-c", script]
    return command_record("hf_gemma_native_image", command, run(command, timeout=360))


def hf_gemma_e002_inputs(model_dir: Path, image: Path, prompt: str) -> dict[str, Any]:
    script = f"""
import hashlib
import json
import torch
from pathlib import Path
from transformers import AutoProcessor
from cactus.transpile.multimodal_runtime import prepare_gemma4_multimodal_inputs

model_dir = Path({str(model_dir)!r})
image_path = Path({str(image)!r})
prompt = {prompt!r}
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
prepared = prepare_gemma4_multimodal_inputs(
    processor,
    prompt=prompt,
    image_files=(str(image_path),),
    audio_file=None,
    torch_dtype=torch.float32,
    use_gemma4_chat_template=True,
)
by_name = {{name: tensor.detach().cpu() for name, tensor in zip(prepared.names, prepared.tensors)}}
input_ids = by_name['input_ids'].reshape(-1).to(torch.int64)
pixel_values = by_name['pixel_values'].to(torch.float32).contiguous()
pixel_position_ids = by_name['pixel_position_ids'].reshape(-1).to(torch.int64).contiguous()
tokenizer = getattr(processor, 'tokenizer', processor)
image_token = getattr(processor, 'image_token', None)
image_token_id = tokenizer.convert_tokens_to_ids(image_token) if image_token is not None and hasattr(tokenizer, 'convert_tokens_to_ids') else -1
ids = input_ids.tolist()
image_positions = [i for i, tok in enumerate(ids) if tok == image_token_id]
processor_prompt = str(prepared.metadata.get('processor_prompt'))
assistant_generation_start = len(ids)
if processor_prompt.endswith('<|turn>model\\n'):
    no_generation_prompt = processor_prompt[:-len('<|turn>model\\n')]
    no_gen = tokenizer(no_generation_prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].reshape(-1)
    assistant_generation_start = int(no_gen.numel())
valid_positions = int(((by_name['pixel_position_ids'] != -1).any(dim=-1)).sum().item())
pixel_np = pixel_values.numpy()
pos_np = by_name['pixel_position_ids'].to(torch.int64).contiguous().numpy()
payload = {{
    'token_count': len(ids),
    'assistant_generation_start': assistant_generation_start,
    'image_token_id': int(image_token_id),
    'image_token_start': int(image_positions[0]) if image_positions else -1,
    'image_token_count': len(image_positions),
    'first_20': ids[:20],
    'last_20': ids[-20:],
    'input_ids': ids,
    'prompt_prefix': processor_prompt[:320],
    'pixel_values_shape': list(pixel_values.shape),
    'pixel_position_ids_shape': list(by_name['pixel_position_ids'].shape),
    'pixel_values_min': float(pixel_values.min().item()),
    'pixel_values_max': float(pixel_values.max().item()),
    'pixel_values_mean': float(pixel_values.mean().item()),
    'pixel_values_sha256': hashlib.sha256(pixel_np.tobytes()).hexdigest(),
    'pixel_position_ids_sha256': hashlib.sha256(pos_np.tobytes()).hexdigest(),
    'valid_position_count': valid_positions,
    'first_pixel_values': pixel_values.reshape(-1)[:32].tolist(),
    'first_position_rows_flat': pixel_position_ids[:32].tolist(),
    'pixel_values': pixel_values.reshape(-1).tolist(),
    'pixel_position_ids': pixel_position_ids.tolist(),
}}
print(json.dumps(payload))
"""
    command = [sys.executable, "-c", script]
    completed = run(command, timeout=360)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return json.loads(completed.stdout)


def e002_cactus_tokens(model: Path, image: Path, prompt: str) -> dict[str, Any]:
    runner = ensure_runner()
    command = [
        str(runner),
        "e002-tokens",
        "--model",
        str(model),
        "--prompt",
        prompt,
        "--image",
        str(image),
    ]
    return native_json(command, timeout=360)


def e002_cactus_preprocess(model: Path, image: Path) -> dict[str, Any]:
    runner = ensure_runner()
    command = [
        str(runner),
        "e002-preprocess",
        "--model",
        str(model),
        "--image",
        str(image),
    ]
    return native_json(command, timeout=360)


def e003_cactus_preprocess_variants(model: Path, image: Path) -> dict[str, Any]:
    runner = ensure_runner()
    command = [
        str(runner),
        "e003-preprocess-variants",
        "--model",
        str(model),
        "--image",
        str(image),
    ]
    return native_json(command, timeout=360)


def e002_cactus_vision(model: Path, image: Path) -> dict[str, Any]:
    runner = ensure_runner()
    command = [
        str(runner),
        "e002-vision",
        "--model",
        str(model),
        "--image",
        str(image),
        "--prompt",
        "What animal is in this image? Reply in one complete short sentence.",
    ]
    return native_json(command, timeout=360)


def hf_gemma_vision_features(model_dir: Path, image: Path, prompt: str) -> dict[str, Any]:
    script = f"""
import json
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from cactus.transpile.multimodal_runtime import prepare_gemma4_multimodal_inputs

model_dir = Path({str(model_dir)!r})
image_path = Path({str(image)!r})
prompt = {prompt!r}
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
prepared = prepare_gemma4_multimodal_inputs(
    processor,
    prompt=prompt,
    image_files=(str(image_path),),
    audio_file=None,
    torch_dtype=torch.float16,
    use_gemma4_chat_template=True,
)
by_name = {{name: tensor for name, tensor in zip(prepared.names, prepared.tensors)}}
model = AutoModelForImageTextToText.from_pretrained(model_dir, local_files_only=True, dtype=torch.float16, device_map='cpu')
pixel_values = by_name['pixel_values'].to(dtype=torch.float16)
position_ids = by_name['pixel_position_ids'].to(dtype=torch.long)
with torch.no_grad():
    outputs = model.get_image_features(pixel_values=pixel_values, image_position_ids=position_ids)
candidates = {{}}
for name, tensor in (
    ('last_hidden_state', getattr(outputs, 'last_hidden_state', None)),
    ('pooler_output', getattr(outputs, 'pooler_output', None)),
):
    if tensor is None:
        continue
    flat = tensor.detach().float().cpu().reshape(-1)
    candidates[name] = {{
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'count': int(flat.numel()),
        'min': float(flat.min().item()) if flat.numel() else 0.0,
        'max': float(flat.max().item()) if flat.numel() else 0.0,
        'mean': float(flat.mean().item()) if flat.numel() else 0.0,
        'l2': float(torch.linalg.vector_norm(flat).item()) if flat.numel() else 0.0,
        'first_values': flat[:32].tolist(),
        'values': flat.tolist(),
    }}
print(json.dumps({{'features': candidates}}))
"""
    completed = run([sys.executable, "-c", script], timeout=420)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return json.loads(completed.stdout)


def first_mismatch(left: list[Any], right: list[Any]) -> int | None:
    for idx, (lval, rval) in enumerate(zip(left, right)):
        if lval != rval:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def sha256_float_list(values: list[float]) -> str:
    import array

    arr = array.array("f", values)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def run_e002_boundary(out_dir: Path, weights_root: Path) -> None:
    image = ASSET_DIR / "test_monkey.png"
    prompt = "What animal is in this image? Reply in one complete short sentence."
    fresh_gemma = weights_root / "gemma-4-e2b-it-fresh-mm"
    hf_snapshot = Path("/Users/noahcylich/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db")

    hf = hf_gemma_e002_inputs(hf_snapshot, image, prompt)
    cactus_tokens = e002_cactus_tokens(fresh_gemma, image, prompt)
    write_json(out_dir / "hf_native_inputs.json", hf)
    write_json(out_dir / "cactus_token_probe.json", cactus_tokens)

    token_mismatch = first_mismatch(hf["input_ids"], cactus_tokens["input_ids"])
    token_compare = {
        "gate": "prompt_and_token_spans",
        "status": "pass" if token_mismatch is None else "fail",
        "input_ids_match": token_mismatch is None,
        "first_mismatch_index": token_mismatch,
        "hf": {key: hf[key] for key in ("token_count", "assistant_generation_start", "image_token_id", "image_token_start", "image_token_count", "first_20", "last_20", "prompt_prefix")},
        "cactus": {key: cactus_tokens[key] for key in ("token_count", "assistant_generation_start", "image_token_id", "image_token_start", "image_token_count", "first_20", "last_20", "prompt_prefix")},
    }
    if token_mismatch is not None:
        lo = max(0, token_mismatch - 8)
        hi = token_mismatch + 9
        token_compare["mismatch_window"] = {
            "hf": hf["input_ids"][lo:hi],
            "cactus": cactus_tokens["input_ids"][lo:hi],
            "start": lo,
        }
    write_json(out_dir / "token_spans.json", token_compare)
    if token_mismatch is not None:
        return

    cactus_preprocess = e002_cactus_preprocess(fresh_gemma, image)
    write_json(out_dir / "cactus_preprocess.json", cactus_preprocess)
    hf_values = hf["pixel_values"]
    cactus_values = cactus_preprocess["pixel_values"]
    hf_positions = hf["pixel_position_ids"]
    cactus_positions = cactus_preprocess["pixel_position_ids"]
    if len(hf_values) != len(cactus_values):
        max_abs_diff = None
        mean_abs_diff = None
        first_pixel_mismatch = min(len(hf_values), len(cactus_values))
    else:
        diffs = [abs(float(a) - float(b)) for a, b in zip(hf_values, cactus_values)]
        max_abs_diff = max(diffs) if diffs else 0.0
        mean_abs_diff = sum(diffs) / len(diffs) if diffs else 0.0
        first_pixel_mismatch = next((i for i, diff in enumerate(diffs) if diff > 1e-5), None)
    position_mismatch = first_mismatch(hf_positions, cactus_positions)
    preprocess_pass = (
        hf["pixel_values_shape"] == cactus_preprocess["pixel_values_shape"]
        and hf["pixel_position_ids_shape"] == cactus_preprocess["pixel_position_ids_shape"]
        and position_mismatch is None
        and max_abs_diff is not None
        and max_abs_diff <= 1e-5
    )
    compare = {
        "gate": "image_preprocessing_tensors",
        "status": "pass" if preprocess_pass else "fail",
        "pixel_values_shape_match": hf["pixel_values_shape"] == cactus_preprocess["pixel_values_shape"],
        "pixel_position_ids_shape_match": hf["pixel_position_ids_shape"] == cactus_preprocess["pixel_position_ids_shape"],
        "pixel_position_ids_match": position_mismatch is None,
        "first_position_mismatch_index": position_mismatch,
        "pixel_values_max_abs_diff": max_abs_diff,
        "pixel_values_mean_abs_diff": mean_abs_diff,
        "first_pixel_value_mismatch_index_at_1e-5": first_pixel_mismatch,
        "hf_summary": {key: hf[key] for key in ("pixel_values_shape", "pixel_position_ids_shape", "pixel_values_min", "pixel_values_max", "pixel_values_mean", "pixel_values_sha256", "pixel_position_ids_sha256", "valid_position_count", "first_pixel_values", "first_position_rows_flat")},
        "cactus_summary": {key: cactus_preprocess[key] for key in ("pixel_values_shape", "pixel_position_ids_shape", "pixel_values_min", "pixel_values_max", "pixel_values_mean", "pixel_values_hash", "pixel_position_ids_hash", "valid_position_count", "first_pixel_values", "first_position_rows_flat")},
    }
    if first_pixel_mismatch is not None:
        lo = max(0, first_pixel_mismatch - 8)
        hi = first_pixel_mismatch + 9
        compare["pixel_mismatch_window"] = {
            "start": lo,
            "hf": hf_values[lo:hi],
            "cactus": cactus_values[lo:hi],
        }
    if position_mismatch is not None:
        lo = max(0, position_mismatch - 8)
        hi = position_mismatch + 9
        compare["position_mismatch_window"] = {
            "start": lo,
            "hf": hf_positions[lo:hi],
            "cactus": cactus_positions[lo:hi],
        }
    write_json(out_dir / "image_preprocess_compare.json", compare)


def run_e003_preprocess_isolation(out_dir: Path, weights_root: Path) -> None:
    image = ASSET_DIR / "test_monkey.png"
    prompt = "What animal is in this image? Reply in one complete short sentence."
    fresh_gemma = weights_root / "gemma-4-e2b-it-fresh-mm"
    hf_snapshot = Path("/Users/noahcylich/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db")

    hf = hf_gemma_e002_inputs(hf_snapshot, image, prompt)
    variants = e003_cactus_preprocess_variants(fresh_gemma, image)
    write_json(out_dir / "hf_native_inputs.json", hf)
    write_json(out_dir / "cactus_resize_variants.json", variants)

    hf_values = hf["pixel_values"]
    comparisons: dict[str, Any] = {}
    best_name: str | None = None
    best_max: float | None = None
    for name, variant in variants["variants"].items():
        values = variant["pixel_values"]
        if len(values) != len(hf_values):
            comparison = {
                "status": "fail",
                "length_match": False,
                "hf_length": len(hf_values),
                "variant_length": len(values),
            }
        else:
            diffs = [abs(float(a) - float(b)) for a, b in zip(hf_values, values)]
            max_abs = max(diffs) if diffs else 0.0
            mean_abs = sum(diffs) / len(diffs) if diffs else 0.0
            first_mismatch = next((i for i, diff in enumerate(diffs) if diff > 1e-5), None)
            comparison = {
                "status": "pass" if max_abs <= 1e-5 else "fail",
                "length_match": True,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "first_mismatch_index_at_1e-5": first_mismatch,
                "variant_summary": {
                    key: variant[key]
                    for key in (
                        "resized_min",
                        "resized_max",
                        "first_resized_values",
                        "pixel_values_min",
                        "pixel_values_max",
                        "pixel_values_mean",
                        "first_pixel_values",
                    )
                },
            }
            if first_mismatch is not None:
                lo = max(0, first_mismatch - 8)
                hi = first_mismatch + 9
                comparison["mismatch_window"] = {
                    "start": lo,
                    "hf": hf_values[lo:hi],
                    "variant": values[lo:hi],
                }
            if best_max is None or max_abs < best_max:
                best_name = name
                best_max = max_abs
        comparisons[name] = comparison

    write_json(
        out_dir / "preprocess_operation_compare.json",
        {
            "gate": "e003_preprocess_operation_isolation",
            "source_width": variants["source_width"],
            "source_height": variants["source_height"],
            "target_width": variants["target_width"],
            "target_height": variants["target_height"],
            "patch_size": variants["patch_size"],
            "pooling_kernel_size": variants["pooling_kernel_size"],
            "max_patches": variants["max_patches"],
            "patch_dim": variants["patch_dim"],
            "rescale_factor": variants["rescale_factor"],
            "hf_summary": {
                key: hf[key]
                for key in (
                    "pixel_values_shape",
                    "pixel_values_min",
                    "pixel_values_max",
                    "pixel_values_mean",
                    "first_pixel_values",
                )
            },
            "best_variant": best_name,
            "best_variant_max_abs_diff": best_max,
            "comparisons": comparisons,
        },
    )


def compare_float_vectors(hf_values: list[float], cactus_values: list[float]) -> dict[str, Any]:
    if len(hf_values) != len(cactus_values):
        return {
            "status": "fail",
            "length_match": False,
            "hf_length": len(hf_values),
            "cactus_length": len(cactus_values),
        }
    cactus_nonfinite = sum(1 for value in cactus_values if value is None)
    hf_nonfinite = sum(1 for value in hf_values if value is None)
    if cactus_nonfinite or hf_nonfinite:
        return {
            "status": "fail",
            "length_match": True,
            "hf_nonfinite_count": hf_nonfinite,
            "cactus_nonfinite_count": cactus_nonfinite,
            "first_nonfinite_index": next(
                (i for i, (a, b) in enumerate(zip(hf_values, cactus_values)) if a is None or b is None),
                None,
            ),
        }
    diffs = [abs(float(a) - float(b)) for a, b in zip(hf_values, cactus_values)]
    dot = sum(float(a) * float(b) for a, b in zip(hf_values, cactus_values))
    hf_norm = sum(float(a) * float(a) for a in hf_values) ** 0.5
    cactus_norm = sum(float(b) * float(b) for b in cactus_values) ** 0.5
    max_abs = max(diffs) if diffs else 0.0
    mean_abs = sum(diffs) / len(diffs) if diffs else 0.0
    first_mismatch = next((i for i, diff in enumerate(diffs) if diff > 1e-3), None)
    result: dict[str, Any] = {
        "status": "pass" if max_abs <= 1e-2 and mean_abs <= 1e-3 else "fail",
        "length_match": True,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "cosine_similarity": dot / (hf_norm * cactus_norm) if hf_norm and cactus_norm else 0.0,
        "first_mismatch_index_at_1e-3": first_mismatch,
    }
    if first_mismatch is not None:
        lo = max(0, first_mismatch - 8)
        hi = first_mismatch + 9
        result["mismatch_window"] = {
            "start": lo,
            "hf": hf_values[lo:hi],
            "cactus": cactus_values[lo:hi],
        }
    return result


def run_e002_vision_boundary(out_dir: Path, weights_root: Path) -> None:
    image = ASSET_DIR / "test_monkey.png"
    prompt = "What animal is in this image? Reply in one complete short sentence."
    fresh_gemma = weights_root / "gemma-4-e2b-it-fresh-mm"
    hf_snapshot = Path("/Users/noahcylich/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db")

    hf = hf_gemma_vision_features(hf_snapshot, image, prompt)
    cactus = e002_cactus_vision(fresh_gemma, image)
    write_json(out_dir / "hf_vision_features.json", hf)
    write_json(out_dir / "cactus_vision_features.json", cactus)

    comparisons: dict[str, Any] = {}
    for cactus_name, cactus_feature in cactus["features"].items():
        comparisons[cactus_name] = {}
        for hf_name, hf_feature in hf["features"].items():
            comparison = compare_float_vectors(hf_feature["values"], cactus_feature["values"])
            comparison["hf_shape"] = hf_feature["shape"]
            comparison["cactus_shape"] = cactus_feature["shape"]
            comparison["hf_summary"] = {key: hf_feature[key] for key in ("dtype", "count", "min", "max", "mean", "l2", "first_values")}
            comparison["cactus_summary"] = {key: cactus_feature[key] for key in ("precision", "count", "finite_count", "nonfinite_count", "min", "max", "mean", "l2", "first_values")}
            comparisons[cactus_name][hf_name] = comparison

    write_json(
        out_dir / "vision_encoder_compare.json",
        {
            "gate": "vision_encoder_output",
            "comparisons": comparisons,
        },
    )


def manifest_summary(path: Path) -> dict[str, Any]:
    manifest = path / "components" / "manifest.json"
    config = path / "config.txt"
    result: dict[str, Any] = {"path": str(path), "manifest_exists": manifest.exists(), "config_exists": config.exists()}
    if manifest.exists():
        data = json.loads(manifest.read_text())
        result.update(
            {
                "family": data.get("family"),
                "task": data.get("task"),
                "model_source": data.get("model_source"),
                "component_order": data.get("component_order"),
                "components": [
                    {
                        "component": component.get("component"),
                        "logical_inputs": component.get("logical_inputs"),
                        "logical_outputs": component.get("logical_outputs"),
                    }
                    for component in data.get("components", [])
                ],
            }
        )
    return result


def run_case(case: str, out_dir: Path, weights_root: Path) -> None:
    image = ASSET_DIR / "test_monkey.png"
    prompt = "What animal is in this image? Reply in one complete short sentence."
    fresh_gemma = weights_root / "gemma-4-e2b-it-fresh-mm"
    qwen = weights_root / "qwen3-vl-2b-instruct-reconvert"
    parakeet = weights_root / "parakeet-tdt-0.6b-v3-transpiled"
    hf_snapshot = Path("/Users/noahcylich/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db")

    if case == "inventory":
        write_json(out_dir / "inventory.json", {"gemma_fresh": manifest_summary(fresh_gemma)})
        return
    if case == "hf-gemma-image":
        write_json(out_dir / "hf_gemma_image.json", hf_gemma_image(hf_snapshot, image, prompt))
        return
    if case == "hf-gemma-native-image":
        write_json(out_dir / "hf_gemma_native_image.json", hf_gemma_native_image(hf_snapshot, image, prompt))
        return
    if case == "cactus-gemma-image":
        write_json(out_dir / "cactus_gemma_image.json", cactus_complete(fresh_gemma, prompt, image=image))
        return
    if case == "cactus-gemma-audio":
        wav = ensure_question_wav()
        audio_prompt = "Listen to the audio and answer briefly."
        write_json(out_dir / "cactus_gemma_audio.json", cactus_complete(fresh_gemma, audio_prompt, audio=wav))
        return
    if case == "cactus-gemma-mixed":
        wav = ensure_question_wav()
        mixed_prompt = "Answer the spoken question using the provided image. Reply in one short sentence."
        write_json(out_dir / "cactus_gemma_mixed.json", cactus_complete(fresh_gemma, mixed_prompt, image=image, audio=wav))
        return
    if case == "cactus-qwen-image":
        write_json(out_dir / "cactus_qwen_image.json", cactus_complete(qwen, prompt, image=image, max_tokens=40))
        return
    if case == "parakeet-audio":
        wav = ensure_question_wav()
        write_json(out_dir / "parakeet_audio.json", parakeet_transcribe(parakeet, wav))
        write_json(out_dir / "parakeet_audio_pcm.json", parakeet_transcribe(parakeet, wav, pcm=True))
        return
    if case == "all-baseline":
        for subcase in ("inventory", "hf-gemma-image", "hf-gemma-native-image", "cactus-gemma-image", "cactus-qwen-image", "parakeet-audio"):
            run_case(subcase, out_dir, weights_root)
        return
    if case == "e002-boundary":
        run_e002_boundary(out_dir, weights_root)
        return
    if case == "e003-preprocess-isolation":
        run_e003_preprocess_isolation(out_dir, weights_root)
        return
    if case == "e002-vision-boundary":
        run_e002_vision_boundary(out_dir, weights_root)
        return
    raise ValueError(f"unknown case: {case}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--weights-root", default=str(DEFAULT_WEIGHTS_ROOT))
    args = parser.parse_args()

    out_dir = INVESTIGATION_ROOT / "artifacts" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "run_metadata.json",
        {
            "case": args.case,
            "experiment": args.experiment,
            "repo_root": str(REPO_ROOT),
            "weights_root": str(Path(args.weights_root).resolve()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    run_case(args.case, out_dir, Path(args.weights_root).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
