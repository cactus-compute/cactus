from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import assert_completion, assert_transcript, model_path, run_runner


def long_prompt() -> str:
    sentence = "Summarize this integration test context in one concise sentence. "
    return sentence * 40


@pytest.mark.smoke
def test_gemma4_e2b_text_image_and_audio(runner: Path, weights_root: Path, assets: dict[str, Path]) -> None:
    text_model = model_path(
        weights_root,
        "gemma-4-e2b-it-sharedkv",
        "gemma-4-e2b-it-sharedkv-c128",
        "gemma-4-e2b-it",
        native_config=True,
    )
    media_model = model_path(
        weights_root,
        "gemma-4-e2b-it-alt-fixed-rowtuned-mm",
        "gemma-4-e2b-it",
        native_config=True,
    )
    cases = [
        [text_model, "--prompt", "Reply with one short sentence about local inference."],
        [media_model, "--prompt", "Describe this image in one short sentence.", "--image", assets["image_monkey"]],
        [media_model, "--prompt", "Transcribe or summarize this audio in one short sentence.", "--audio", assets["audio"]],
    ]
    for case in cases:
        model, *args = case
        result = run_runner(runner, "complete", "--model", model, "--max-tokens", 24, *args, timeout=240)
        assert_completion(result)


@pytest.mark.smoke
def test_gemma4_e2b_combined_image_audio(
    runner: Path,
    weights_root: Path,
    assets: dict[str, Path],
) -> None:
    model = model_path(
        weights_root,
        "gemma-4-e2b-it-alt-fixed-rowtuned-mm",
        "gemma-4-e2b-it",
        native_config=True,
    )
    result = run_runner(
        runner,
        "complete",
        "--model",
        model,
        "--prompt",
        "Describe the image and summarize the audio in one short sentence.",
        "--image",
        assets["image_thing"],
        "--audio",
        assets["audio"],
        "--max-tokens",
        24,
        timeout=240,
    )
    assert_completion(result)


@pytest.mark.smoke
def test_gemma4_e2b_multiturn_text_reuse(runner: Path, weights_root: Path) -> None:
    model = model_path(
        weights_root,
        "gemma-4-e2b-it-sharedkv",
        "gemma-4-e2b-it-sharedkv-c128",
        "gemma-4-e2b-it",
        native_config=True,
    )
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Remember the name Henry."},
        {"role": "assistant", "content": "I will remember the name Henry."},
        {"role": "user", "content": "What name should you remember?"},
    ]
    result = run_runner(
        runner,
        "complete",
        "--model",
        model,
        "--messages-json",
        json.dumps(messages),
        "--max-tokens",
        24,
        timeout=240,
    )
    assert_completion(result)


@pytest.mark.full
def test_gemma4_e4b_all_modalities_when_available(runner: Path, weights_root: Path, assets: dict[str, Path]) -> None:
    model = model_path(weights_root, "gemma-4-e4b-it", required=False, native_config=True)
    result = run_runner(
        runner,
        "complete",
        "--model",
        model,
        "--prompt",
        "Describe the image and audio briefly.",
        "--image",
        assets["image_monkey"],
        "--audio",
        assets["audio"],
        "--max-tokens",
        24,
        timeout=360,
    )
    assert_completion(result)


@pytest.mark.smoke
def test_qwen3_vl_text_and_chunked_prefill(runner: Path, weights_root: Path) -> None:
    model = model_path(
        weights_root,
        "qwen3-vl-2b-instruct-reconvert",
        "qwen3-vl-2b-instruct-reconvert-c256",
        "qwen3-vl-2b-instruct-reconvert-chunks",
        "qwen3-vl-2b-instruct",
        "qwen3-vl-2b",
        native_config=True,
    )
    for prompt in ("Reply with one short sentence about cactus.", long_prompt()):
        result = run_runner(runner, "complete", "--model", model, "--prompt", prompt, "--max-tokens", 16, timeout=300)
        assert_completion(result)


@pytest.mark.smoke
def test_qwen3_vl_image(runner: Path, weights_root: Path, assets: dict[str, Path]) -> None:
    model = model_path(
        weights_root,
        "qwen3-vl-2b-instruct-reconvert",
        "qwen3-vl-2b-instruct-reconvert-c256",
        "qwen3-vl-2b-instruct-reconvert-c512",
        native_config=True,
    )
    result = run_runner(
        runner,
        "complete",
        "--model",
        model,
        "--prompt",
        "Describe this image in one short sentence.",
        "--image",
        assets["image_monkey"],
        "--max-tokens",
        16,
        timeout=300,
    )
    assert_completion(result)


@pytest.mark.smoke
def test_lfm2_vl_text_image_and_chunked_prefill(runner: Path, weights_root: Path, assets: dict[str, Path]) -> None:
    model = model_path(
        weights_root,
        "lfm2.5-vl-1.6b-retranspile-4096",
        "lfm2.5-vl-1.6b-rechunk-c256",
        "lfm2.5-vl-1.6b",
        "lfm2.5-vl-1.6b-rechunk-c128",
        "lfm2-vl-450m",
        native_config=True,
    )
    for prompt, image in (
        ("Reply with one short sentence about local model testing.", None),
        ("Describe this image in one short sentence.", assets["image_thing"]),
        (long_prompt(), None),
    ):
        args: list[object] = ["complete", "--model", model, "--prompt", prompt, "--max-tokens", 16]
        if image is not None:
            args.extend(["--image", image])
        result = run_runner(runner, *args, timeout=300)
        assert_completion(result)


@pytest.mark.smoke
def test_parakeet_tdt_file_and_pcm_transcription(runner: Path, weights_root: Path, assets: dict[str, Path]) -> None:
    model = model_path(
        weights_root,
        "parakeet-tdt-0.6b-v3-transpiled",
        "parakeet-tdt-0.6b-v3",
        native_config=True,
    )
    file_result = run_runner(runner, "transcribe", "--model", model, "--audio", assets["audio"], timeout=240)
    pcm_result = run_runner(runner, "transcribe-pcm", "--model", model, "--audio", assets["audio"], timeout=240)
    assert_transcript(file_result)
    assert_transcript(pcm_result)


@pytest.mark.full
def test_whisper_small_file_and_pcm_transcription_when_available(
    runner: Path,
    weights_root: Path,
    assets: dict[str, Path],
) -> None:
    model = model_path(weights_root, "whisper-small", required=False, native_config=True)
    file_result = run_runner(runner, "transcribe", "--model", model, "--audio", assets["audio"], timeout=300)
    pcm_result = run_runner(runner, "transcribe-pcm", "--model", model, "--audio", assets["audio"], timeout=300)
    assert_transcript(file_result)
    assert_transcript(pcm_result)
