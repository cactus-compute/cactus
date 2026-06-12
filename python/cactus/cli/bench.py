import csv
import glob
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import BLUE, PROJECT_ROOT, RED, YELLOW, print_color
from .test import _bundle_dir, _ensure_bundle

DEVICE_DIR = "/data/local/tmp/cactus-bench"
BENCH_ENV = {"CACTUS_DISABLE_CLOUD_HANDOFF": "1", "CACTUS_NO_CLOUD_TELE": "1"}


def _cmake_build(src, *defines):
    build_dir = src / "build"
    subprocess.run(
        ["cmake", "-S", str(src), "-B", str(build_dir), *defines,
         "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_RULE_MESSAGES=OFF", "-DCMAKE_VERBOSE_MAKEFILE=OFF"],
        check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "llm_bench",
         "-j", str(os.cpu_count() or 4)],
        check=True)
    return build_dir / "llm_bench"


def _build_local():
    return _cmake_build(PROJECT_ROOT / "cactus-engine" / "tests")


def _find_ndk():
    candidates = []
    if os.environ.get("ANDROID_NDK_HOME"):
        candidates.append(os.environ["ANDROID_NDK_HOME"])
    roots = []
    if os.environ.get("ANDROID_HOME"):
        roots.append(f"{os.environ['ANDROID_HOME']}/ndk")
    roots.append(os.path.expanduser("~/Library/Android/sdk/ndk"))
    for root in roots:
        if os.path.isdir(root):
            candidates.extend(f"{root}/{v}" for v in sorted(os.listdir(root), reverse=True))
    candidates.extend(sorted(glob.glob(
        "/opt/homebrew/Caskroom/android-ndk/*/AndroidNDK*.app/Contents/NDK"), reverse=True))
    candidates.append("/opt/homebrew/share/android-ndk")
    for ndk in candidates:
        if os.path.isfile(f"{ndk}/build/cmake/android.toolchain.cmake"):
            return ndk
    return None


def _build_android():
    ndk = _find_ndk()
    if not ndk:
        print_color(RED, "Android NDK not found. Set ANDROID_NDK_HOME.")
        raise SystemExit(1)
    curl_root = os.environ.get("CACTUS_CURL_ROOT",
                               str(PROJECT_ROOT / "cactus-engine" / "libs" / "curl"))
    return _cmake_build(
        PROJECT_ROOT / "cactus-engine" / "tests" / "android",
        f"-DCMAKE_TOOLCHAIN_FILE={ndk}/build/cmake/android.toolchain.cmake",
        "-DANDROID_ABI=arm64-v8a",
        f"-DANDROID_PLATFORM={os.environ.get('ANDROID_PLATFORM', 'android-21')}",
        f"-DCACTUS_CURL_ROOT={curl_root}")


def _adb(serial):
    return ["adb", "-s", serial] if serial else ["adb"]


def _select_device(serial):
    out = subprocess.run(["adb", "devices"], check=True, capture_output=True, text=True).stdout
    devices = [line.split("\t")[0] for line in out.splitlines() if line.endswith("\tdevice")]
    if serial:
        if serial not in devices:
            print_color(RED, f"Device {serial} not found. Connected: {', '.join(devices) or 'none'}")
            raise SystemExit(1)
        return serial
    if len(devices) == 1:
        return devices[0]
    if not devices:
        print_color(RED, "No Android device connected.")
    else:
        print_color(RED, f"Multiple devices connected, pass --serial: {', '.join(devices)}")
    raise SystemExit(1)


def _warn_if_not_charging(adb):
    out = subprocess.run(adb + ["shell", "dumpsys", "battery"],
                         capture_output=True, text=True).stdout
    if not any(f"{kind} powered: true" in out for kind in ("AC", "USB", "Wireless")):
        print_color(YELLOW,
                    "WARNING: device not charging - numbers will be power-limited "
                    "(see docs/benchmarking.md)")


def _push_weights(adb, name, local_dir):
    if subprocess.run(adb + ["shell", "test", "-d", f"{DEVICE_DIR}/weights/{name}"],
                      capture_output=True).returncode == 0:
        return
    print_color(BLUE, f"Pushing weights: {name}")
    subprocess.run(adb + ["push", str(local_dir), f"{DEVICE_DIR}/weights/"],
                   check=True, capture_output=True)


def _driver_args(args, device_tokens_path=None):
    if args.prompt is not None:
        return ["--prompt", args.prompt, "--max-tokens", str(args.max_tokens), "--warmup"]
    return ["--tokens", device_tokens_path, "--decode", str(args.decode_tokens)]


def _parse_result(stdout):
    try:
        result = json.loads(stdout.strip())
    except json.JSONDecodeError:
        print_color(RED, f"Unparseable benchmark output:\n{stdout.strip()}")
        raise SystemExit(1)
    return result


def _write_tokens_csv(prefill_tokens):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    f.write(",".join(str(1000 + i) for i in range(prefill_tokens)))
    f.close()
    return f.name


def _exec_round(cmd, env=None):
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print_color(RED, proc.stderr.strip() or proc.stdout.strip() or "benchmark run failed")
        raise SystemExit(1)
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)
    return proc.stdout


def _run_rounds(args, run_round):
    results = []
    if args.prompt is None:
        run_round()
    for i in range(args.rounds):
        result = _parse_result(run_round())
        ttft = f", ttft {result['time_to_first_token_ms']:.0f}ms" if args.prompt is not None else ""
        print(f"round {i + 1}: prefill {result['prefill_tps']:.1f} tps, "
              f"decode {result['decode_tps']:.1f} tps{ttft}")
        results.append(result)
    return results


def _report(args, model_name, results):
    if not results:
        return
    mean_prefill = sum(r["prefill_tps"] for r in results) / len(results)
    mean_decode = sum(r["decode_tps"] for r in results) / len(results)
    print_color(BLUE, f"mean over {len(results)} rounds: "
                      f"prefill {mean_prefill:.1f} tps, decode {mean_decode:.1f} tps")
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "mode", "round", "prefill_tps", "decode_tps",
                             "time_to_first_token_ms"])
            mode = "prompt" if args.prompt is not None else "tokens"
            for i, r in enumerate(results):
                writer.writerow([model_name, mode, i + 1, r["prefill_tps"],
                                 r["decode_tps"], r.get("time_to_first_token_ms", "")])
        print(f"Results written to: {args.output}")


def _bench_local(args, bundle):
    binary = _build_local()
    tokens_csv = None if args.prompt is not None else _write_tokens_csv(args.prefill_tokens)
    cmd = [str(binary), str(bundle)] + _driver_args(args, tokens_csv)
    env = {**os.environ, **BENCH_ENV}
    try:
        return _run_rounds(args, lambda: _exec_round(cmd, env))
    finally:
        if tokens_csv:
            os.unlink(tokens_csv)


def _bench_android(args, bundle):
    binary = _build_android()
    adb = _adb(_select_device(args.serial))
    _warn_if_not_charging(adb)
    subprocess.run(adb + ["shell", "mkdir", "-p", f"{DEVICE_DIR}/weights"], check=True)
    subprocess.run(adb + ["push", str(binary), f"{DEVICE_DIR}/llm_bench"],
                   check=True, capture_output=True)
    subprocess.run(adb + ["shell", "chmod", "+x", f"{DEVICE_DIR}/llm_bench"], check=True)
    _push_weights(adb, bundle.name, bundle)

    device_tokens = None
    if args.prompt is None:
        tokens_csv = _write_tokens_csv(args.prefill_tokens)
        device_tokens = f"{DEVICE_DIR}/tokens.csv"
        subprocess.run(adb + ["push", tokens_csv, device_tokens],
                       check=True, capture_output=True)
        os.unlink(tokens_csv)

    runner = (f"taskset {shlex.quote(args.cpu_mask)} ./llm_bench" if args.cpu_mask
              else "./llm_bench")
    driver = " ".join(shlex.quote(a) for a in _driver_args(args, device_tokens))
    env = " ".join(f"{k}={v}" for k, v in BENCH_ENV.items())
    shell_cmd = (f"cd {DEVICE_DIR} && {env} {runner} "
                 f"{shlex.quote(f'weights/{bundle.name}')} {driver}")
    return _run_rounds(args, lambda: _exec_round(adb + ["shell", shell_cmd]))


def cmd_bench(args):
    if args.cpu_mask and not args.android:
        print_color(RED, "--cpu-mask requires --android (taskset runs on the device)")
        return 2
    if args.prompt is not None and not args.prompt.strip():
        print_color(RED, "--prompt must not be empty")
        return 2

    candidate = Path(args.model_id)
    if (candidate / "components" / "manifest.json").exists():
        bundle = candidate
    else:
        if not _ensure_bundle(args.model_id):
            return 1
        bundle = _bundle_dir(args.model_id)

    if args.prompt is not None:
        workload = f"prompt + {args.max_tokens} decode, {args.rounds} rounds (per-round warmup)"
    else:
        workload = (f"{args.prefill_tokens}-token prefill + {args.decode_tokens} decode, "
                    f"1 warmup + {args.rounds} rounds")
    target = "android" if args.android else "local"
    print_color(BLUE, f"==> {bundle.name}: {workload} ({target})")

    try:
        results = _bench_android(args, bundle) if args.android else _bench_local(args, bundle)
    except subprocess.CalledProcessError as exc:
        print_color(RED, f"command failed ({exc.returncode}): {' '.join(map(str, exc.cmd))}")
        return 1
    _report(args, bundle.name, results)
    return 0
