"""Pinned engine artifacts and independently versioned model weights."""
import hashlib
import json
import os
import platform
import sysconfig
import sys
import tempfile
import zipfile

WHISTLE = "whistle"
MIRAGE = "mirage"
ENGINE_REPO = "Cactus-Compute/cactus"
ENGINE_VERSION = "4.1.0"
ENGINE_REVISION = "4e822c9036be2eb0e47ef556df32a818bb1572ef"
MANIFEST_SHA256 = "32e182f6f811acde1109084ee5f5c9a606098b95cc8774f27ece8bc1a506de3b"
MODEL_REPOS = {3: "Cactus-Compute/needle3", WHISTLE: "Cactus-Compute/whistle", MIRAGE: "Cactus-Compute/mirage"}
MODEL_REVISIONS = {
    3: "d6c7d6a2e9655505e0f8e23339356dd1705900eb",
    WHISTLE: "741118687e1f008f06d1351a402a731274789790",
    MIRAGE: "76c1e9e53381aca8fc3843c636216c96ff220122",
}
MODEL_SHA256 = {3: 'c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38', 'whistle': 'b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb', 'mirage': '4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237'}
BASE_WEIGHTS = {3: "needle3.cact", WHISTLE: "whistle.cact", MIRAGE: "mirage.cact"}
CHECKPOINT_PREFIX = "checkpoints"
WHEEL_TAGS = ("macosx_11_0_arm64", "macosx_11_0_x86_64", "manylinux2014_aarch64", "manylinux2014_x86_64",
              "musllinux_1_2_aarch64", "musllinux_1_2_x86_64", "win_amd64", "win_arm64")
PLATFORMS = ("macos-arm64", "linux-x86_64", "linux-arm64", "linux-armv7", "linux-riscv64", "linux-mipsel",
             "windows-x86_64", "windows-arm64", "android-arm64", "android-armv7", "android-riscv64",
             "ios-arm64", "ios-sim-arm64", "tvos-arm64", "watchos-arm64", "wasm", "wasm-component")


def _engine(generation=3):
    if str(generation) == "3":
        return 3
    if generation in (WHISTLE, MIRAGE):
        return generation
    raise ValueError(f"unsupported model: {generation}; Needle 2 requires cactus-compute<4 from the legacy branch")


def _lib_name_for(tag):
    return "libcactus." + ("dylib" if tag.startswith("macosx") else "dll" if tag.startswith("win") else "so")


def _lib_name():
    return "libcactus." + ("dylib" if sys.platform == "darwin" else "dll" if sys.platform == "win32" else "so")


def lib_name(generation=3, tag=None):
    _engine(generation)
    return _lib_name_for(tag) if tag else _lib_name()


def lib_path_env(generation=3):
    _engine(generation)
    return "CACTUS_LIB_PATH"


def _loaded_libc():
    try:
        with open("/proc/self/maps", "rb") as maps:
            blob = maps.read()
    except OSError:
        return None
    if b"musl" in blob:
        return "musl"
    if b"/libc.so.6" in blob or b"/ld-linux" in blob:
        return "glibc"
    return None


def _is_musl():
    """True on a musl system (Alpine and friends).

    platform.libc_ver() scans the interpreter binary and reports glibc on some
    musl builds, so the loader and the filesystem are asked first.
    """
    if sys.platform != "linux":
        return False
    libc = _loaded_libc()
    if libc:
        return libc == "musl"
    import glob
    if glob.glob("/lib/ld-musl-*.so.1"):
        return True
    if glob.glob("/lib*/libc.so.6") or glob.glob("/lib/*-linux-gnu/libc.so.6"):
        return False
    if "musl" in (sysconfig.get_config_var("HOST_GNU_TYPE") or ""):
        return True
    libc, _ = platform.libc_ver()
    return not libc


_ENGINE_ARCHES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}


def _is_android():
    """Android reports itself as Linux but links Bionic, where no published engine loads.

    Termux is the usual way this is reached.  ``platform.libc_ver()`` finds no glibc
    there, so the Linux path would pick the musl aarch64 wheel and fail to load it.
    """
    if hasattr(sys, "getandroidapilevel"):
        return True
    if _loaded_libc():
        return False
    return bool(os.environ.get("ANDROID_ROOT") and os.environ.get("ANDROID_DATA"))


def _platform_tag():
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
        return "macosx_11_0_" + arch
    if _is_android():
        raise RuntimeError(
            "no Cactus engine wheel is published for Android (Bionic): the Linux wheels link "
            "glibc or musl, so libcactus fails to load with a missing libm.so.6. The Android "
            "builds ship a static library and the cactus CLI (cactus download android-arm64), "
            "not a shared library for ctypes; to use the Python API here, build a shared "
            "libcactus for Bionic and point CACTUS_LIB_PATH at it.")
    arch = _ENGINE_ARCHES.get(machine)
    if arch is None:


        system = "Windows" if sys.platform == "win32" else "Linux"
        raise RuntimeError(
            f"no Cactus engine is published for {system} on {platform.machine() or 'this machine'}; "
            "the engine builds are for x86_64 and arm64. Build libcactus for this machine "
            "and point CACTUS_LIB_PATH at it.")
    if sys.platform == "win32":
        return "win_arm64" if arch == "aarch64" else "win_amd64"
    family = "musllinux_1_2_" if _is_musl() else "manylinux2014_"
    return family + arch


def other_libc_tag():
    """The tag for the other Linux libc family, or None off Linux."""
    tag = _platform_tag()
    if tag.startswith("manylinux2014_"):
        return tag.replace("manylinux2014_", "musllinux_1_2_")
    if tag.startswith("musllinux_1_2_"):
        return tag.replace("musllinux_1_2_", "manylinux2014_")
    return None


def engine_repo(generation=3):
    _engine(generation)
    return ENGINE_REPO


def engine_version(generation=3):
    _engine(generation)
    return ENGINE_VERSION


def repo_prefix(generation=3):
    _engine(generation)
    return f"releases/{ENGINE_VERSION}/"


def engine_wheel(version, tag, generation=3):
    _engine(generation)
    return f"releases/{version}/python/cactus_runtime-{version}-py3-none-{tag}.whl"


def _cache_root():
    return os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "cactus")


def engine_cache_dir():
    return os.path.join(_cache_root(), "engine", ENGINE_VERSION, ENGINE_REVISION)


def cache_dir(generation=3):
    key = _engine(generation)
    return os.path.join(_cache_root(), "models", BASE_WEIGHTS[key][:-5], MODEL_REVISIONS[key])


def base_weights(generation=3):
    return BASE_WEIGHTS[_engine(generation)]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".cactus-", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _manifest():
    from huggingface_hub import hf_hub_download
    if len(ENGINE_REVISION) != 40 or len(MANIFEST_SHA256) != 64:
        raise RuntimeError("this Cactus client has no verified engine release pin")
    path = hf_hub_download(repo_id=ENGINE_REPO, filename=repo_prefix() + "manifest.json", revision=ENGINE_REVISION)
    if _sha256(path) != MANIFEST_SHA256:
        raise RuntimeError("engine release manifest checksum mismatch")
    with open(path) as source:
        manifest = json.load(source)
    if manifest.get("version") != ENGINE_VERSION or manifest.get("api") != "cactus":
        raise RuntimeError("incompatible engine release manifest")
    return manifest


def _artifact(filename):
    from huggingface_hub import hf_hub_download
    manifest = _manifest()
    entry = manifest["files"][filename]
    path = hf_hub_download(repo_id=ENGINE_REPO, filename=repo_prefix() + filename, revision=ENGINE_REVISION)
    if os.path.getsize(path) != entry["size"] or _sha256(path) != entry["sha256"]:
        raise RuntimeError(f"engine artifact checksum mismatch: {filename}")
    return path


def unpublished_engine_wheels():
    """Validate every pinned runtime wheel's remote checksum before a client release."""
    from huggingface_hub import HfApi
    manifest = _manifest()
    paths = [engine_wheel(ENGINE_VERSION, tag) for tag in WHEEL_TAGS]
    entries = {entry.path: entry for entry in HfApi().get_paths_info(ENGINE_REPO, paths, revision=ENGINE_REVISION)}
    missing = []
    for path in paths:
        entry = entries.get(path)
        expected = manifest["files"].get(path[len(repo_prefix()):])
        if not entry or not expected or entry.size != expected["size"]:
            missing.append(ENGINE_REPO + "/" + path)
            continue
        if entry.lfs:
            valid = entry.lfs.sha256 == expected["sha256"]
        else:
            try:
                _artifact(path[len(repo_prefix()):])
                valid = True
            except RuntimeError:
                valid = False
        if not valid:
            missing.append(ENGINE_REPO + "/" + path)
    return missing


def _register_download(generation=3):
    from huggingface_hub import hf_hub_download
    key = _engine(generation)
    try:
        hf_hub_download(repo_id=MODEL_REPOS[key], filename="config.json", revision=MODEL_REVISIONS[key], force_download=True)
    except Exception:
        pass


def download_platform(name, out_dir, generation=3, dest=None):
    _engine(generation)
    if name not in PLATFORMS:
        raise ValueError(f"unsupported platform: {name}")
    manifest = _manifest()
    files = [path for path in manifest["files"] if path.startswith(name + "/")]
    if not files:
        raise FileNotFoundError(f"{name} is missing from the pinned engine release")
    dest = dest or os.path.join(out_dir, name)
    outputs = []
    for filename in files:
        cached = _artifact(filename)
        target = os.path.join(dest, os.path.basename(filename))
        with open(cached, "rb") as source:
            _atomic_write(target, source.read())
        if os.path.basename(target) in ("cactus", "cactus.exe"):
            os.chmod(target, os.stat(target).st_mode | 0o111)
        outputs.append(target)
    return outputs


def fetch_weights(generation=3, dest_dir=None, force=False):
    from huggingface_hub import hf_hub_download
    key = _engine(generation)
    override = os.environ.get("CACTUS_" + ("NEEDLE" if key == 3 else key.upper()) + "_WEIGHTS")
    if override:
        return os.fspath(override)
    _register_download(key)
    name = base_weights(key)
    dest_dir = dest_dir or cache_dir(key)
    out = os.path.join(dest_dir, name)
    expected = MODEL_SHA256.get(key)
    if os.path.exists(out) and not force and expected and _sha256(out) == expected:
        return out
    cached = hf_hub_download(repo_id=MODEL_REPOS[key], filename=name, revision=MODEL_REVISIONS[key], force_download=force)
    if expected and _sha256(cached) != expected:
        raise RuntimeError(f"model checksum mismatch: {name}")
    with open(cached, "rb") as source:
        return _atomic_write(out, source.read())


def fetch_checkpoint(name, dest_dir, generation=3, force=False):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    key = _engine(generation)
    _register_download(key)
    base = os.path.basename(name)
    for candidate in (f"{CHECKPOINT_PREFIX}/{base}", base):
        try:
            cached = hf_hub_download(repo_id=MODEL_REPOS[key], filename=candidate, revision=MODEL_REVISIONS[key], force_download=force)
            with open(cached, "rb") as source:
                return _atomic_write(os.path.join(dest_dir, base), source.read())
        except EntryNotFoundError:
            continue
    raise FileNotFoundError(f"{base} is not published in {MODEL_REPOS[key]}")


def fetch_library(version=None, dest_dir=None, tag=None, generation=3):
    _engine(generation)
    if version is not None and version != ENGINE_VERSION:
        raise ValueError("only the pinned engine version can be downloaded")
    tag = tag or _platform_tag()
    if tag not in WHEEL_TAGS:
        raise ValueError(f"unsupported wheel tag: {tag}")
    path = _artifact(f"python/cactus_runtime-{ENGINE_VERSION}-py3-none-{tag}.whl")
    member = "cactus/" + lib_name(tag=tag)
    with zipfile.ZipFile(path) as archive:
        data = archive.read(member)
    out = os.path.join(dest_dir or engine_cache_dir(), lib_name(tag=tag))
    if not os.path.isfile(out) or _sha256(out) != hashlib.sha256(data).hexdigest():
        _atomic_write(out, data)
    return out
