import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from transpiler.Converter import constants
from transpiler.Converter.convert import convert
from transpiler.Converter.models import LayerMap
from transpiler.Generator.generate import generate_bundle, materialize_runtime_bundle_files
from transpiler.ModelProfiles.profiles import GEMMA4_E2B_PROFILE


def main() -> None:
    model_id = "google/gemma-4-E2B"
    input_modalities = ("text", "vision", "audio")
    output_dir = constants.CONVERTER_JSON_DIR
    repo_root = Path(__file__).resolve().parents[3]
    weights_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CACTUS_TRANSPILER_WEIGHTS_DIR")
    weights_dir = Path(weights_arg).expanduser() if weights_arg else repo_root / "weights" / "gemma4-e2b"
    bundle_dir = weights_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    prefill_path = output_dir / "output_prefill_with_cache.json"
    prefill_simplified_path = output_dir / "output_prefill_with_cache_simplified.json"
    decode_path = output_dir / "output_decode_with_cache.json"
    decode_simplified_path = output_dir / "output_decode_with_cache_simplified.json"

    convert(
        model_id=model_id,
        input_modalities=input_modalities,
        output_path=str(prefill_path),
        simplified_output_path=str(prefill_simplified_path),
        inference_mode=constants.PREFILL_WITH_CACHE_MODE,
    )
    convert(
        model_id=model_id,
        input_modalities=input_modalities,
        output_path=str(decode_path),
        simplified_output_path=str(decode_simplified_path),
        inference_mode=constants.DECODE_WITH_CACHE_MODE,
    )

    prefill = LayerMap.model_validate_json(prefill_simplified_path.read_text(encoding="utf-8"))
    decode = LayerMap.model_validate_json(decode_simplified_path.read_text(encoding="utf-8"))

    materialize_runtime_bundle_files(weights_dir, bundle_dir)

    result = generate_bundle(
        {"prefill": prefill, "decode": decode},
        bundle_dir,
        model_profile=GEMMA4_E2B_PROFILE,
        weights_dir=weights_dir,
        strict=True,
    )
    print(f"Wrote component bundle to {bundle_dir}")
    print(f"Wrote runtime plan to {result.runtime_plan_path}")


if __name__ == "__main__":
    main()
