"""The optional inference interface must not load MLX for base-package users."""
import subprocess
import sys

import pytest
from cactus.mlx import _environment


def test_public_import_and_cli_help_do_not_load_optional_backends():
    subprocess.run([sys.executable, "-c",
                    "import cactus; import cactus.mlx; import sys; "
                    "assert 'mlx.core' not in sys.modules; "
                    "assert 'mlx_lm' not in sys.modules; "
                    "assert 'jax' not in sys.modules"], check=True)
    result = subprocess.run([sys.executable, "-m", "cactus.mlx.cli", "--help"],
                            capture_output=True, text=True, check=True)
    assert "generate" in result.stdout and "server" in result.stdout


def test_unsupported_platform_fails_before_loading_a_model(monkeypatch):
    monkeypatch.setattr(_environment.sys, "platform", "linux")
    monkeypatch.setattr(_environment.sys, "version_info", (3, 12))
    with pytest.raises(RuntimeError, match="Apple silicon"):
        _environment.require_runtime()


def test_missing_optional_dependencies_explain_the_install_command(monkeypatch):
    monkeypatch.setattr(_environment.sys, "platform", "darwin")
    monkeypatch.setattr(_environment.sys, "version_info", (3, 12))
    monkeypatch.setattr(_environment.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(_environment.platform, "mac_ver", lambda: ("26.0", (), ""))
    monkeypatch.setattr(_environment.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match=r"cactus-compute\[mlx\]"):
        _environment.require_runtime()
