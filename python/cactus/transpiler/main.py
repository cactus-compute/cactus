import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from transpiler.Converter import constants
from transpiler.Converter.convert import convert


def main() -> None:
    model_id = "google/gemma-4-E2B"
    input_modalities = ("text", "vision", "audio")
    output_dir = constants.CONVERTER_JSON_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    convert(
        model_id=model_id,
        input_modalities=input_modalities,
        output_path=str(output_dir / "output_prefill_with_cache.json"),
        inference_mode=constants.PREFILL_WITH_CACHE_MODE,
    )
    convert(
        model_id=model_id,
        input_modalities=input_modalities,
        output_path=str(output_dir / "output_decode_with_cache.json"),
        inference_mode=constants.DECODE_WITH_CACHE_MODE,
    )


if __name__ == "__main__":
    main()
