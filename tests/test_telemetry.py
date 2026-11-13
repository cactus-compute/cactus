import pytest

from cactus import _telemetry


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("CACTUS_TELEMETRY", "NEEDLE_TELEMETRY", "DO_NOT_TRACK", "CI"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_enabled_by_default(clean_env):
    assert _telemetry._enabled()


@pytest.mark.parametrize("name,value", [
    ("CACTUS_TELEMETRY", "0"),
    ("NEEDLE_TELEMETRY", "0"),
    ("DO_NOT_TRACK", "1"),
    ("CI", "true"),
])
def test_opt_outs_disable_telemetry(clean_env, name, value):
    clean_env.setenv(name, value)
    assert not _telemetry._enabled()
