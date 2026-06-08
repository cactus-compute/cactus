from pathlib import Path

from .common import BLUE, GREEN, RED, YELLOW, print_color, weights_root
from .download import get_weights_dir


def _resolve_model_arg(model: str | None) -> tuple[Path | None, str | None]:
    if not model:
        return None, None
    path = Path(model).expanduser()
    if path.is_dir():
        return path, path.name
    candidate = weights_root() / model
    if candidate.is_dir():
        return candidate, candidate.name
    hf_candidate = get_weights_dir(model)
    if hf_candidate.is_dir():
        return hf_candidate, hf_candidate.name
    return None, model


def _is_valid_bundle(path: Path) -> bool:
    return (path / "config.txt").exists() and (path / "components" / "manifest.json").exists()


def cmd_serve(args):
    """Start the OpenAI-compatible HTTP server."""
    model_path, model_name = _resolve_model_arg(args.model)
    if args.model and model_path is None:
        print_color(RED, f"Error: model not found: {args.model}")
        print("Prepare a v2 bundle first with `cactus run <model>` (or `cactus convert <model>` then `cactus transpile <model>`).")
        return 1
    if model_path is not None and not _is_valid_bundle(model_path):
        print_color(RED, f"Error: not a valid v2 Cactus bundle: {model_path}")
        print("Expected config.txt and components/manifest.json.")
        return 1

    try:
        import uvicorn
    except ImportError:
        print_color(RED, "Error: uvicorn not installed. Install the serve extra or run `pip install fastapi uvicorn python-multipart`.")
        return 1

    try:
        from ..server import create_app
    except ImportError:
        print_color(RED, "Error: server dependencies not installed. Install the serve extra or run `pip install fastapi uvicorn python-multipart`.")
        return 1

    try:
        application = create_app(
            weights_root=weights_root(),
            model_path=model_path,
            default_model=model_name,
        )
    except RuntimeError as exc:
        print_color(RED, f"Error: {exc}")
        print("Prepare a v2 bundle first with `cactus run <model>` (or `cactus convert <model>` then `cactus transpile <model>`).")
        return 1

    models = sorted(application.state.registry.models)
    print_color(GREEN, f"Available models: {', '.join(models)}")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print_color(YELLOW, f"Warning: binding to {args.host} exposes the server beyond loopback; no auth is enforced.")
    print_color(BLUE, f"Starting server on {args.host}:{args.port}")
    uvicorn.run(application, host=args.host, port=args.port, log_level="info")
    return 0
