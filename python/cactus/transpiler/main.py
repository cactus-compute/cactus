import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from transpiler.Converter.convert import convert


def main() -> None:
    convert(
        model_id="google/gemma-4-E2B",
        input_modalities=("text", "vision", "audio"),
    )


if __name__ == "__main__":
    main()
