from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = REPO_ROOT / "local_integration_tests" / ".tmp"
ASSET_DIR = REPO_ROOT / "cactus-engine" / "tests" / "assets"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "smoke: fast local integration coverage")
    config.addinivalue_line("markers", "full: slower coverage for every available key model")


def _weights_root() -> Path:
    explicit = os.environ.get("CACTUS_INTEGRATION_WEIGHTS")
    candidates = [explicit] if explicit else []
    candidates.extend([str(REPO_ROOT.parent / "cactus" / "weights"), str(REPO_ROOT / "weights")])
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return Path(candidates[0]).expanduser().resolve()


@pytest.fixture(scope="session")
def weights_root() -> Path:
    root = _weights_root()
    if not root.exists():
        pytest.fail(f"weights root does not exist: {root}")
    return root


@pytest.fixture(scope="session")
def assets() -> dict[str, Path]:
    result = {
        "audio": ASSET_DIR / "test.wav",
        "image_monkey": ASSET_DIR / "test_monkey.png",
        "image_thing": ASSET_DIR / "test_thing.png",
    }
    missing = [str(path) for path in result.values() if not path.exists()]
    if missing:
        pytest.fail("missing integration fixtures: " + ", ".join(missing))
    return result


@pytest.fixture(scope="session")
def runner() -> Path:
    env = os.environ.copy()
    env["CACTUS_NO_CLOUD_TELE"] = "1"
    subprocess.run([str(REPO_ROOT / "cactus" / "build.sh")], cwd=REPO_ROOT, env=env, check=True)

    build_dir = TMP_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(REPO_ROOT / "local_integration_tests"),
            "-B",
            str(build_dir),
            f"-DCACTUS_REPO_ROOT={REPO_ROOT}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "integration_runner", "-j", str(os.cpu_count() or 4)],
        cwd=REPO_ROOT,
        check=True,
    )
    binary = build_dir / "integration_runner"
    if not binary.exists():
        pytest.fail(f"integration runner was not built: {binary}")
    return binary


def _is_transpiled_bundle(path: Path) -> bool:
    return (path / "components" / "manifest.json").exists() or (path / "manifest.json").exists()


def model_path(
    weights_root: Path,
    *candidates: str,
    required: bool = True,
    transpiled: bool = True,
    native_config: bool = False,
) -> Path | None:
    for name in candidates:
        path = weights_root / name
        if not path.exists():
            continue
        if transpiled and not _is_transpiled_bundle(path):
            continue
        if native_config and not (path / "config.txt").exists():
            continue
        if path.exists():
            return path
    if required:
        suffixes = []
        if transpiled:
            suffixes.append("components/manifest.json")
        if native_config:
            suffixes.append("config.txt")
        suffix = " with " + " and ".join(suffixes) if suffixes else ""
        pytest.fail("missing required model artifact" + suffix + "; tried: " + ", ".join(candidates))
    pytest.skip("optional model artifact not available; tried: " + ", ".join(candidates))
    return None


def run_runner(runner: Path, *args: object, timeout: int = 180) -> dict[str, Any]:
    env = os.environ.copy()
    env["CACTUS_NO_CLOUD_TELE"] = "1"
    env.pop("CACTUS_KV_CACHE_FP16", None)
    completed = subprocess.run(
        [str(runner), *(str(arg) for arg in args)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"integration runner failed with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"runner did not return JSON: {exc}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")


def assert_completion(result: dict[str, Any]) -> None:
    response = str(result.get("response") or result.get("text") or "").strip()
    token_count = int(result.get("completion_tokens") or result.get("tokens_predicted") or result.get("decode_tokens") or 0)
    assert response or token_count > 0, result


def assert_transcript(result: dict[str, Any], fragments: tuple[str, ...] = ("hello", "testing", "voice")) -> None:
    transcript = str(result.get("response") or result.get("transcript") or "").strip().lower()
    assert len(transcript) > 5, result
    assert any(fragment in transcript for fragment in fragments), transcript
