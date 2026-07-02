from pathlib import Path

from .common import BLUE, GREEN, RED, YELLOW, apply_runtime_env, is_valid_bundle, print_color, weights_root
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


def _ensure_engine_library() -> bool:
    from .runtime import ensure_python_runtime_library

    try:
        ensure_python_runtime_library()
        return True
    except Exception as exc:
        print_color(RED, f"Error: could not build the Cactus engine: {exc}")
        print_color(YELLOW, "Build it manually with `cactus build --python` (needs cmake and a C++ toolchain).")
        return False


def cmd_serve(args):
    """Start the OpenAI-compatible HTTP server."""
    apply_runtime_env(args)
    if not _ensure_engine_library():
        return 1
    model_path, model_name = _resolve_model_arg(args.model_id)
    if args.model_id and model_path is None:
        from .model import prepare_bundle
        built = prepare_bundle(args, fail_prefix=f"Error: could not prepare {args.model_id}")
        if built is None:
            return 1
        model_path, model_name = built, built.name
    if model_path is not None and not is_valid_bundle(model_path):
        print_color(RED, f"Error: not a valid v2 Cactus bundle: {model_path}")
        print("Expected config.txt and components/manifest.json.")
        return 1

    try:
        import uvicorn
    except ImportError:
        print_color(RED, "Error: uvicorn not installed. Install the serve extra or run `pip install fastapi uvicorn python-multipart`.")
        return 1

    backend = getattr(args, "backend", "auto")
    if backend and backend != "auto":
        from ..bindings.cactus import cactus_set_backend
        if cactus_set_backend(backend) != 0:
            print_color(YELLOW, f"Backend '{backend}' not available; using auto")

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
            auto_handoff=not args.no_cloud_handoff,
            confidence_threshold=args.confidence_threshold,
            cloud_timeout_ms=args.cloud_timeout_ms,
        )
    except RuntimeError as exc:
        print_color(RED, f"Error: {exc}")
        print("Prepare a bundle first with `cactus run <model>` (or `cactus convert <model>`).")
        return 1

    models = sorted(application.state.registry.models)
    print_color(GREEN, f"Available models: {', '.join(models)}")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print_color(YELLOW, f"Warning: binding to {args.host} exposes the server beyond loopback; no auth is enforced.")
    print_color(BLUE, f"Starting server on {args.host}:{args.port}")
    uvicorn.run(application, host=args.host, port=args.port, log_level="info",
                access_log=not getattr(args, "no_access_log", False))
    return 0
