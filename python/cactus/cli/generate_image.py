import json
import struct
import time
import zlib
from pathlib import Path

from .common import apply_runtime_env, print_color, GREEN, RED


def cmd_generate_image(args):
    from .model import prepare_bundle

    apply_runtime_env(args)

    bundle_dir = prepare_bundle(args)
    if bundle_dir is None:
        return 1

    if bundle_execution_strategy(bundle_dir) != "iterative_denoise":
        print_color(RED, f"{args.model_id} is not a text-to-image model")
        return 1

    from ..bindings.cactus import cactus_destroy, cactus_generate_image, cactus_init

    output_path = Path(args.output).expanduser()
    print_color(GREEN, f"Generating {args.steps}-step image with model: {args.model_id}")
    print()

    model = cactus_init(str(bundle_dir))
    try:
        started = time.perf_counter()
        pixels, width, height = cactus_generate_image(
            model,
            args.prompt,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - started
    except RuntimeError as exc:
        print_color(RED, str(exc))
        return 1
    finally:
        cactus_destroy(model)

    write_png(output_path, pixels, width, height)
    print_color(GREEN, f"{width}x{height} image written to {output_path} in {elapsed:.1f}s")
    return 0


def bundle_execution_strategy(bundle_dir: Path) -> str:
    manifest = Path(bundle_dir) / "components" / "manifest.json"
    if not manifest.is_file():
        return ""
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("runtime_execution_strategy", "")
    except (OSError, ValueError):
        return ""


def write_png(path: Path, pixels: bytes, width: int, height: int) -> None:
    stride = width * 3
    raw = b"".join(b"\x00" + pixels[row * stride:(row + 1) * stride] for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
