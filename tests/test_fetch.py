from cactus.agent import runtime
import sys

import pytest

def test_lib_path_env_override(tmp_path, monkeypatch):
    import cactus

    fake = tmp_path / "libcactus.dylib"
    fake.write_bytes(b"x")
    monkeypatch.setenv("CACTUS_LIB_PATH", str(fake))
    assert runtime._library_path() == str(fake)


def test_legacy_generation_is_rejected_before_loading(monkeypatch):
    import cactus
    monkeypatch.setenv("CACTUS_LIB_PATH", "/tmp/legacy.so")
    with pytest.raises(ValueError, match="legacy branch"):
        runtime._library_path(2)


def test_download_target_kinds():
    import pytest
    from cactus.cli.download import _download_target

    assert _download_target("macos-arm64") == ("platform", "macos-arm64")
    assert _download_target("needle3") == ("base", 3)
    assert _download_target("mirage") == ("base", "mirage")
    with pytest.raises(SystemExit):
        _download_target("needle2.cact")
    assert _download_target("needle3.safetensors") == ("checkpoint", "needle3.safetensors")
    assert _download_target("acme/tuned/model.cact") == ("hub", "acme/tuned/model.cact")
    with pytest.raises(SystemExit, match="unknown download"):
        _download_target("needle4")


def test_fetch_weights_copies_the_base_archive_and_reuses_it(tmp_path, monkeypatch):
    monkeypatch.delenv("CACTUS_NEEDLE_WEIGHTS", raising=False)
    from cactus.agent import fetch

    archive = tmp_path / "needle3.cact"
    archive.write_bytes(b"weights")
    import hashlib
    monkeypatch.setitem(fetch.MODEL_SHA256, 3, hashlib.sha256(b"weights").hexdigest())
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs["filename"])
        return str(archive)

    monkeypatch.setattr(fetch, "_register_download", lambda generation: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    dest = tmp_path / "cache"
    out = fetch.fetch_weights(3, str(dest))
    assert out == str(dest / "needle3.cact") and open(out, "rb").read() == b"weights"
    assert calls == ["needle3.cact"]
    assert fetch.fetch_weights(3, str(dest)) == out and calls == ["needle3.cact"]


def test_fetch_checkpoint_prefers_the_checkpoints_folder(tmp_path, monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError
    from cactus.agent import fetch

    source = tmp_path / "needle3.safetensors"
    source.write_bytes(b"ckpt")

    def fake_download(**kwargs):
        if kwargs["filename"] != "checkpoints/needle3.safetensors":
            raise EntryNotFoundError("missing")
        return str(source)

    monkeypatch.setattr(fetch, "_register_download", lambda generation: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    out = fetch.fetch_checkpoint("needle3.safetensors", str(tmp_path / "dl" / "checkpoints"))
    assert out.endswith("checkpoints/needle3.safetensors") and open(out, "rb").read() == b"ckpt"


@pytest.mark.parametrize("model", [3, "whistle", "mirage"])
def test_cached_model_load_refreshes_only_its_config(tmp_path, monkeypatch, model):
    import hashlib
    from cactus.agent import fetch

    monkeypatch.delenv("CACTUS_" + ("NEEDLE" if model == 3 else model.upper()) + "_WEIGHTS", raising=False)
    weights = tmp_path / fetch.BASE_WEIGHTS[model]
    weights.write_bytes(b"cached model")
    monkeypatch.setitem(fetch.MODEL_SHA256, model, hashlib.sha256(weights.read_bytes()).hexdigest())
    requests = []

    def download(**kwargs):
        requests.append(kwargs)
        return str(tmp_path / "config.json")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    for _ in range(2):
        assert fetch.fetch_weights(model, tmp_path) == str(weights)
    assert requests == [{
        "repo_id": fetch.MODEL_REPOS[model],
        "filename": "config.json",
        "revision": fetch.MODEL_REVISIONS[model],
        "force_download": True,
    }] * 2

    def offline(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", offline)
    assert fetch.fetch_weights(model, tmp_path) == str(weights)


def test_weights_spec_parsing():
    from cactus.cli.download import _weights_spec

    assert _weights_spec("acme/tuned/model.cact") == ("acme/tuned", "model.cact")
    assert _weights_spec("acme/tuned") == ("acme/tuned", None)
    assert _weights_spec("acme/tuned/sub/dir/m.cact") == ("acme/tuned", "sub/dir/m.cact")


def test_lib_name_for_tags():
    from cactus.agent.fetch import _lib_name_for

    assert _lib_name_for("macosx_11_0_arm64") == "libcactus.dylib"
    assert _lib_name_for("win_amd64") == "libcactus.dll"
    assert _lib_name_for("manylinux2014_aarch64") == "libcactus.so"
    assert _lib_name_for("musllinux_1_2_x86_64") == "libcactus.so"


def test_component_platform_is_downloadable():
    from cactus.agent.fetch import PLATFORMS

    assert "wasm-component" in PLATFORMS


def test_release_gate_checks_pinned_revision_and_all_wheel_hashes(monkeypatch):
    from types import SimpleNamespace
    from cactus.agent import fetch
    paths = [fetch.engine_wheel(fetch.ENGINE_VERSION, tag) for tag in fetch.WHEEL_TAGS]
    files = {path[len(fetch.repo_prefix()):]: {"size": 5, "sha256": "a" * 64} for path in paths}
    monkeypatch.setattr(fetch, "_manifest", lambda: {"files": files})
    entries = {path: SimpleNamespace(path=path, size=5, lfs=SimpleNamespace(sha256="a" * 64)) for path in paths}
    def get_paths(self, repo, wanted, revision):
        assert repo == fetch.ENGINE_REPO and revision == fetch.ENGINE_REVISION
        assert wanted == paths
        return list(entries.values())
    monkeypatch.setattr("huggingface_hub.HfApi.get_paths_info", get_paths)
    assert fetch.unpublished_engine_wheels() == []
    del entries[paths[-1]]
    entries[paths[0]].lfs.sha256 = "b" * 64
    assert fetch.unpublished_engine_wheels() == [fetch.ENGINE_REPO + "/" + paths[0], fetch.ENGINE_REPO + "/" + paths[-1]]


def test_unpublished_manifest_blocks_a_release(monkeypatch):
    from cactus.agent import fetch
    monkeypatch.setattr(fetch, "ENGINE_REVISION", "UNPUBLISHED")
    with pytest.raises(RuntimeError, match="no verified engine release pin"):
        fetch.unpublished_engine_wheels()


def test_release_gate_runs_before_the_publish_step():
    from pathlib import Path

    workflow = Path(__file__).parents[1] / ".github/workflows/release.yaml"
    text = workflow.read_text()
    assert text.index("unpublished_engine_wheels") < text.index("pypa/gh-action-pypi-publish")


def test_fetch_library_creates_destination(tmp_path, monkeypatch):
    import zipfile
    from cactus.agent import fetch

    wheel = tmp_path / "engine.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("cactus/libcactus.so", b"engine")
    monkeypatch.setattr(fetch, "_register_download", lambda generation: None)
    monkeypatch.setattr(fetch, "_artifact", lambda filename: str(wheel))
    out = fetch.fetch_library(fetch.ENGINE_VERSION, tmp_path / "new", tag="manylinux2014_x86_64")

    assert (tmp_path / "new" / "libcactus.so").read_bytes() == b"engine"
    assert out == str(tmp_path / "new" / "libcactus.so")


def test_engine_gate_finds_the_cache_the_runtime_loads_from(tmp_path, monkeypatch):
    import cactus
    from cactus.agent import fetch
    from conftest import _engine_available

    package = tmp_path / "pkg"
    package.mkdir()
    monkeypatch.setattr(cactus, "__file__", str(package / "__init__.py"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("CACTUS_LIB_PATH", raising=False)
    monkeypatch.delenv("CACTUS_LIB_PATH", raising=False)

    assert not _engine_available()

    from pathlib import Path
    cache = Path(fetch.engine_cache_dir())
    cache.mkdir(parents=True)
    (cache / fetch._lib_name()).write_bytes(b"")

    assert _engine_available()


def test_engine_gate_honours_the_library_override(tmp_path, monkeypatch):
    import cactus
    from cactus.agent import fetch
    from conftest import _engine_available

    package = tmp_path / "pkg"
    package.mkdir()
    monkeypatch.setattr(cactus, "__file__", str(package / "__init__.py"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    engine = tmp_path / fetch._lib_name()
    engine.write_bytes(b"")
    monkeypatch.setenv("CACTUS_LIB_PATH", str(engine))
    assert _engine_available()

    monkeypatch.setenv("CACTUS_LIB_PATH", str(tmp_path / "gone"))
    assert not _engine_available()


def test_musl_detected_when_libc_ver_claims_glibc(monkeypatch, tmp_path):
    import platform
    from cactus.agent import fetch

    monkeypatch.setattr(platform, "libc_ver", lambda *a, **k: ("glibc", "2.9"))
    monkeypatch.setattr(sys, "platform", "linux")
    maps = tmp_path / "maps"
    maps.write_bytes(b"7f0000000000-7f0000001000 r-xp /lib/ld-musl-x86_64.so.1\n")
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: real_open(maps, *a, **k) if p == "/proc/self/maps"
                        else real_open(p, *a, **k))
    assert fetch._is_musl() is True


def test_glibc_detected_from_maps(monkeypatch, tmp_path):
    import platform
    from cactus.agent import fetch

    monkeypatch.setattr(platform, "libc_ver", lambda *a, **k: ("", ""))
    monkeypatch.setattr(sys, "platform", "linux")
    maps = tmp_path / "maps"
    maps.write_bytes(b"7f0000000000-7f0000001000 r-xp /usr/lib/x86_64-linux-gnu/libc.so.6\n")
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: real_open(maps, *a, **k) if p == "/proc/self/maps"
                        else real_open(p, *a, **k))
    assert fetch._is_musl() is False


def test_platform_tag_names_only_the_architectures_the_engine_is_built_for(monkeypatch):
    import platform
    from cactus.agent import fetch

    monkeypatch.setattr(fetch, "_is_musl", lambda: False)
    monkeypatch.setattr(fetch, "_is_android", lambda: False)
    for system, machine, tag in (("linux", "x86_64", "manylinux2014_x86_64"),
                                 ("linux", "aarch64", "manylinux2014_aarch64"),
                                 ("win32", "AMD64", "win_amd64"),
                                 ("win32", "ARM64", "win_arm64"),
                                 ("darwin", "arm64", "macosx_11_0_arm64")):
        monkeypatch.setattr(sys, "platform", system)
        monkeypatch.setattr(platform, "machine", lambda machine=machine: machine)
        assert fetch._platform_tag() == tag

    # Every other machine was handed the x86_64 build, which cannot load there.
    for system, machine in (("linux", "armv7l"), ("linux", "riscv64"), ("linux", "i686"),
                            ("win32", "x86")):
        monkeypatch.setattr(sys, "platform", system)
        monkeypatch.setattr(platform, "machine", lambda machine=machine: machine)
        with pytest.raises(RuntimeError, match=f"on {machine};.*CACTUS_LIB_PATH"):
            fetch._platform_tag()


def test_other_libc_tag_swaps_families(monkeypatch):
    from cactus.agent import fetch

    monkeypatch.setattr(fetch, "_platform_tag", lambda: "manylinux2014_aarch64")
    assert fetch.other_libc_tag() == "musllinux_1_2_aarch64"
    monkeypatch.setattr(fetch, "_platform_tag", lambda: "musllinux_1_2_x86_64")
    assert fetch.other_libc_tag() == "manylinux2014_x86_64"
    monkeypatch.setattr(fetch, "_platform_tag", lambda: "macosx_11_0_arm64")
    assert fetch.other_libc_tag() is None


def test_engine_load_falls_back_to_the_other_libc(monkeypatch, tmp_path):
    import ctypes
    import cactus
    from cactus.agent import fetch

    good = tmp_path / "musl" / "libcactus.so"
    good.parent.mkdir()
    good.write_bytes(b"x")
    monkeypatch.setattr(runtime, "_library_path", lambda generation=2: str(tmp_path / "libcactus.so"))
    monkeypatch.setattr(fetch, "other_libc_tag", lambda: "musllinux_1_2_x86_64")
    monkeypatch.setattr(fetch, "fetch_library",
                        lambda version=None, dest_dir=None, tag=None, generation=2: str(good))
    loaded = {}

    def fake_cdll(path):
        if path.endswith("musl/libcactus.so"):
            loaded["path"] = path
            return "handle"
        raise OSError("Error relocating libcactus.so: strtoll_l: symbol not found")

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    with pytest.warns(UserWarning, match="using the musllinux"):
        assert runtime._load_cdll(3) == "handle"
    assert loaded["path"].endswith("musl/libcactus.so")


def test_android_is_refused_instead_of_being_handed_a_glibc_wheel(monkeypatch):
    """Termux reports Linux/aarch64, so it used to download a wheel Bionic cannot load."""
    import platform
    from cactus.agent import fetch

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(fetch, "_is_musl", lambda: False)

    monkeypatch.setattr(fetch, "_is_android", lambda: False)
    assert fetch._platform_tag() == "manylinux2014_aarch64"

    monkeypatch.setattr(fetch, "_is_android", lambda: True)
    with pytest.raises(RuntimeError, match="Android .Bionic.*libm.so.6.*CACTUS_LIB_PATH"):
        fetch._platform_tag()


def test_android_is_detected_from_the_api_level_or_the_runtime_env(monkeypatch):
    from cactus.agent import fetch

    monkeypatch.setattr(fetch, "_loaded_libc", lambda: None)
    monkeypatch.delenv("ANDROID_ROOT", raising=False)
    monkeypatch.delenv("ANDROID_DATA", raising=False)
    monkeypatch.delattr(sys, "getandroidapilevel", raising=False)
    assert fetch._is_android() is False

    monkeypatch.setenv("ANDROID_ROOT", "/system")
    assert fetch._is_android() is False, "one marker alone is not Android"
    monkeypatch.setenv("ANDROID_DATA", "/data")
    assert fetch._is_android() is True

    monkeypatch.delenv("ANDROID_ROOT")
    monkeypatch.delenv("ANDROID_DATA")
    monkeypatch.setattr(sys, "getandroidapilevel", lambda: 34, raising=False)
    assert fetch._is_android() is True


def test_android_markers_inside_a_glibc_or_musl_userspace_are_linux(monkeypatch, tmp_path):
    from cactus.agent import fetch

    monkeypatch.delattr(sys, "getandroidapilevel", raising=False)
    monkeypatch.setenv("ANDROID_ROOT", "/system")
    monkeypatch.setenv("ANDROID_DATA", "/data")
    maps = tmp_path / "maps"
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: real_open(maps, *a, **k) if p == "/proc/self/maps"
                        else real_open(p, *a, **k))
    rootfs = b"/data/data/com.termux/files/usr/var/lib/proot-distro/installed-rootfs"
    for loaded, android in ((rootfs + b"/debian/usr/lib/aarch64-linux-gnu/libc.so.6", False),
                            (rootfs + b"/alpine/lib/ld-musl-aarch64.so.1", False),
                            (b"/apex/com.android.runtime/lib64/bionic/libc.so", True)):
        maps.write_bytes(b"7f0000000000-7f0000001000 r-xp 00000000 fd:00 1 " + loaded + b"\n")
        assert fetch._is_android() is android, loaded


def test_building_the_published_archive_needs_no_training_stack(tmp_path, monkeypatch):
    import cactus._telemetry
    from cactus.agent import fetch
    from cactus.cli import main

    base = tmp_path / "needle3.cact"
    base.write_bytes(b"archive")

    def fake_platform(name, out_dir, generation=3, dest=None):
        (tmp_path / "pi").mkdir()
        (tmp_path / "pi" / "cactus").write_bytes(b"engine")
        return [str(tmp_path / "pi" / "cactus")]

    monkeypatch.setattr(cactus._telemetry, "track", lambda *a, **k: None)
    monkeypatch.setattr(fetch, "fetch_weights", lambda generation=2, dest_dir=None, force=False: str(base))
    monkeypatch.setattr(fetch, "download_platform", fake_platform)
    for name in ("numpy", "jax"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(sys, "argv", ["cactus", "build", "--platform", "linux-arm64", "--out", str(tmp_path / "pi")])
    main()
    assert (tmp_path / "pi" / "needle3.cact").read_bytes() == b"archive"


def test_tampered_manifest_is_rejected(tmp_path, monkeypatch):
    from cactus.agent import fetch
    path = tmp_path / 'manifest.json'
    path.write_text('{"api":"cactus","version":"4.0.0"}')
    monkeypatch.setattr(fetch, 'ENGINE_REVISION', 'a' * 40)
    monkeypatch.setattr(fetch, 'MANIFEST_SHA256', '0' * 64)
    monkeypatch.setattr('huggingface_hub.hf_hub_download', lambda **kwargs: str(path))
    with pytest.raises(RuntimeError, match='manifest checksum mismatch'):
        fetch._manifest()


def test_corrupt_artifact_cannot_replace_existing_library(tmp_path, monkeypatch):
    import hashlib
    from cactus.agent import fetch
    path = tmp_path / 'download.whl'
    path.write_bytes(b'changed')
    target = tmp_path / 'installed' / 'libcactus.so'
    target.parent.mkdir()
    target.write_bytes(b'working engine')
    filename = f'python/cactus_runtime-{fetch.ENGINE_VERSION}-py3-none-manylinux2014_x86_64.whl'
    monkeypatch.setattr(fetch, '_manifest', lambda: {'files': {filename: {'size': 7, 'sha256': hashlib.sha256(b'correct').hexdigest()}}})
    monkeypatch.setattr('huggingface_hub.hf_hub_download', lambda **kwargs: str(path))
    with pytest.raises(RuntimeError, match='artifact checksum mismatch'):
        fetch.fetch_library(dest_dir=target.parent, tag='manylinux2014_x86_64')
    assert target.read_bytes() == b'working engine'


def test_corrupt_cached_model_is_repaired_from_the_pinned_revision(tmp_path, monkeypatch):
    import hashlib
    from cactus.agent import fetch
    monkeypatch.delenv('CACTUS_NEEDLE_WEIGHTS', raising=False)
    source = tmp_path / 'source.cact'
    source.write_bytes(b'correct weights')
    target = tmp_path / 'cache' / 'needle3.cact'
    target.parent.mkdir()
    target.write_bytes(b'corrupt weights')
    monkeypatch.setitem(fetch.MODEL_SHA256, 3, hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(fetch, '_register_download', lambda generation: None)
    def download(**kwargs):
        assert kwargs['revision'] == fetch.MODEL_REVISIONS[3]
        return str(source)
    monkeypatch.setattr('huggingface_hub.hf_hub_download', download)
    fetch.fetch_weights(dest_dir=target.parent)
    assert target.read_bytes() == source.read_bytes()
