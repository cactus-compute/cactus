import argparse

from ..agent import fetch

HELP = """usage: cactus <command> [options]

  run            run a checkpoint on a query (JAX, Needle 3)
  finetune       train a LoRA adapter on JSONL data
  generate-data  synthesise training data via OpenRouter
  build          export a checkpoint (+ adapter) to a .cact archive
  platform       fine-tune, generate data and download models on cactuscompute.com
  download       needle3 | needle3.safetensors | <platform> | model-<id> | <org>/<repo>[/<file>.cact]
  fetch          fetch the engine library for this platform
  playground     serve the browser playground
  whistle        speech to text: playground | compare | finetune | build
  mirage         vision: describe | extract

cactus <command> --help for the options of one command.
Check the readme for the rest."""


def parser():
    parser = argparse.ArgumentParser(prog="cactus", add_help=False)
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("run")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--query", type=str, default=None, help="Query text for tool-call generation")
    p.add_argument("--tools", type=str, default=None, help="Tools JSON for tool-call generation")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy)")

    p = sub.add_parser("finetune")
    p.add_argument("jsonl_path", type=str, help="Path to JSONL training data")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Base model checkpoint (auto-downloads from HuggingFace if omitted)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA adapter rank (default: 16)")
    p.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA scaling alpha (default: 32)")
    p.add_argument("--max-len", type=int, default=1024, help="Max training sequence length")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of examples held out for validation (0 disables)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for LoRA init, validation split, and epoch shuffling")
    p.add_argument("--generate", type=int, default=0,
                   help="Generate N extra examples via OpenRouter before training (0 = off)")
    p.add_argument("--model", type=str, default="deepseek/deepseek-flash-latest",
                   help="OpenRouter model for --generate")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent OpenRouter requests when generating (default: 8)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--out", type=str, default=None,
                   help="Output adapter path (.safetensors)")

    p = sub.add_parser("generate-data")
    p.add_argument("--tools", type=str, default=None, help="Tool schemas JSON to seed generation")
    p.add_argument("--augment", type=str, default=None, help="Existing JSONL to expand")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--workers", type=int, default=16,
                   help="Concurrent OpenRouter requests (default: 16)")
    p.add_argument("--model", type=str, default="deepseek/deepseek-flash-latest")
    p.add_argument("--output", type=str, default=None)

    p = sub.add_parser("build")
    p.add_argument("checkpoint", type=str, nargs="?", default=None,
                   help="Base checkpoint (.safetensors); defaults to the adapter's "
                        "base, else the Needle 3 base (auto-downloads)")
    p.add_argument("--lora", type=str, default=None, help="LoRA adapter to merge before export")
    p.add_argument("--out", type=str, default=None, help="Output .cact path")
    p.add_argument("--upload", action="store_true", help="Push the .cact to $CACTUS_HF_REPO")
    p.add_argument("--layers", type=int, default=None,
                   help="Export the N-layer rung of the base (2..20); default the full 20")
    p.add_argument("--platform", type=str, default=None, choices=fetch.PLATFORMS,
                   help="Also download that platform's engine and header, and place the "
                        "archive beside them as needle3.cact (--out is then a directory)")

    p = sub.add_parser("download")
    p.add_argument("spec", type=str,
                   help="needle3 (base weights), needle3.safetensors (the checkpoint to "
                        "fine-tune), a platform "
                        "folder (e.g. macos-arm64), a fine-tuned model-<id> from "
                        "cactuscompute.com, or a Hugging Face spec: "
                        "<org>/<repo>/<file>.cact, or <org>/<repo> if it holds one archive")
    p.add_argument("--out", type=str, default=".", help="Directory to place the files")
    p.add_argument("--depth", type=int, default=None,
                   help="One size of a fine-tuned model-<id> (default: every size)")
    p.add_argument("--model", choices=["needle3", "whistle", "mirage"],
                   help="Optionally include model weights beside a platform build")

    p = sub.add_parser("platform")
    verbs = p.add_subparsers(dest="verb")
    v = verbs.add_parser("finetune")
    v.add_argument("train", type=str, help="Training .jsonl (a path, or a file-<id> already uploaded)")
    v.add_argument("validation", type=str, help="Validation .jsonl or file-<id>")
    v.add_argument("test", type=str, help="Test .jsonl or file-<id>")
    v.add_argument("--max-depth", type=int, default=None,
                   help="Largest size to train, 2 up to the base depth (default: the base depth)")
    v.add_argument("--suffix", type=str, default=None, help="Name for the job and its models")
    v.add_argument("--out", type=str, default=".", help="Directory for the downloaded .cact files")
    v.add_argument("--depth", type=int, default=None,
                   help="Download one size only (default: every size)")
    v.add_argument("--no-wait", action="store_true",
                   help="Submit and print the job id without polling or downloading")
    v = verbs.add_parser("generate")
    v.add_argument("--tools", type=str, required=True, help="Tool definitions .json, flat or OpenAI form")
    v.add_argument("--examples", type=int, default=1000, help="Examples to generate, 100 to 10,000")
    v.add_argument("--description", type=str, default=None, help="What the product does")
    v.add_argument("--message", action="append", default=None,
                   help="An example user message; repeat for several")
    v.add_argument("--suffix", type=str, default=None, help="Name for the job and its files")
    v.add_argument("--out", type=str, default=".", help="Directory for the generated .jsonl files")
    v.add_argument("--no-wait", action="store_true",
                   help="Submit and print the job id without polling or downloading")
    v = verbs.add_parser("jobs")
    v.add_argument("job_id", type=str, nargs="?", default=None, help="One job to show in full")
    v.add_argument("--wait", action="store_true", help="Poll that job until it ends")
    v.add_argument("--out", type=str, default=None,
                   help="Download what the finished job produced into this directory")
    v.add_argument("--depth", type=int, default=None, help="One size of a fine-tuned model (default: every size)")
    v = verbs.add_parser("models")
    v.add_argument("model_id", type=str, nargs="?", default=None, help="One model to show with its sizes")
    v = verbs.add_parser("files")
    verbs.add_parser("billing")

    p = sub.add_parser("fetch")
    p.add_argument("--out", type=str, default=None,
                   help="Directory to place the engine (default: the cache)")
    p.add_argument("--platform-tag", type=str, default=None,
                   help="Fetch the build for another device, e.g. manylinux2014_aarch64")


    p = sub.add_parser("playground")
    p.add_argument("--weights", type=str, default=None,
                   help="Tuned .cact to serve (defaults to the base model from HuggingFace)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="127.0.0.1")

    from ..agent.whistle import LANGUAGES
    p = sub.add_parser("whistle")
    verbs = p.add_subparsers(dest="verb")
    v = verbs.add_parser("playground")
    v.add_argument("audio", type=str, nargs="?", default=None,
                   help="A WAV file to transcribe once (default: press Enter to speak)")
    v.add_argument("--language", type=str, default=None,
                   help="Force the language, one the weights know: " + " ".join(LANGUAGES) + " for the published "
                        "weights (default: detect it)")
    v.add_argument("--keywords", type=str, default="",
                   help="Words and phrases to favour, comma separated")
    v.add_argument("--word-timestamps", action="store_true",
                   help="Show each word's start, end and probability")
    v.add_argument("--weights", type=str, default=None,
                   help="whistle.cact to load (default: the published weights)")
    v = verbs.add_parser("compare")
    v.add_argument("audio", type=str, nargs="?", default=None,
                   help="A WAV file to run through every model once (default: press Enter to speak)")
    v.add_argument("--weights", type=str, default=None,
                   help="whistle.cact to load (default: the published weights)")
    v = verbs.add_parser("finetune")
    v.add_argument("train", type=str,
                   help='Training JSONL of {"audio": WAV path, "text": transcript, "language": code}, one clip per line')
    v.add_argument("validation", type=str, help="Validation JSONL, scored after every epoch")
    v.add_argument("test", type=str, help="Test JSONL, scored once after training")
    v.add_argument("--language", type=str, default=None,
                   help="Language of lines that do not name one; a new code adds that language")
    v.add_argument("--tokens", type=str, default=None,
                   help="File of tokens to add to the vocabulary, one per line")
    v.add_argument("--checkpoint", type=str, default=None,
                   help="Base checkpoint (default: checkpoints/whistle.safetensors, downloaded if missing)")
    v.add_argument("--epochs", type=int, default=3)
    v.add_argument("--batch-size", type=int, default=16)
    v.add_argument("--lr", type=float, default=1e-4)
    v.add_argument("--lora-rank", type=int, default=16, help="LoRA adapter rank (default: 16)")
    v.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA scaling alpha (default: 32)")
    v.add_argument("--seed", type=int, default=0,
                   help="Random seed for LoRA init and epoch shuffling")
    v.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    v.add_argument("--out", type=str, default=None,
                   help="Output adapter path (.safetensors)")
    v = verbs.add_parser("build")
    v.add_argument("checkpoint", type=str, nargs="?", default=None,
                   help="Base checkpoint (.safetensors); defaults to the adapter's "
                        "base, else the Whistle base (auto-downloads)")
    v.add_argument("--lora", type=str, default=None, help="LoRA adapter to merge before export")
    v.add_argument("--out", type=str, default=None, help="Output .cact path")
    v.add_argument("--upload", action="store_true", help="Push the .cact to $CACTUS_HF_REPO")
    v.add_argument("--layers", type=int, default=None,
                   help="Export the N-layer decoder rung of the base (2..8); default the full 8")
    v.add_argument("--platform", type=str, default=None, choices=fetch.PLATFORMS,
                   help="Also download that platform's engine and header, and place the "
                        "archive beside them as whistle.cact (--out is then a directory)")

    p = sub.add_parser("mirage")
    verbs = p.add_subparsers(dest="verb", required=True)
    for verb in ("describe", "extract"):
        v = verbs.add_parser(verb)
        v.add_argument("image", help="Image file")
        v.add_argument("--weights", help="Mirage .cact; defaults to the pinned model")
        v.add_argument("--language")
        v.add_argument("--keywords", nargs="*")
        v.add_argument("--options", default="{}", help="JSON options for cactus_describe")
        if verb == "extract":
            v.add_argument("--form", required=True, help="JSON file containing the ordered form fields")

    return parser
