import sys
from importlib import import_module

from .parser import HELP, parser

COMMANDS = {
    "run": ("model.run", "main"),
    "finetune": ("model.finetune", "finetune_local"),
    "generate-data": ("model.finetune", "generate_main"),
    "build": ("model.build", "build_main"),
    "download": ("cli.download", "download"),
    "fetch": ("cli.download", "engine"),
    "mirage": ("cli.vision", "run"),
    "platform": ("platform", "main"),
    "playground": ("playground.server", "main"),
}
SPEECH = {
    "playground": ("cli.speech", "playground"),
    "compare": ("cli.speech", "compare"),
    "finetune": ("model.whistle_finetune", "finetune_main"),
    "build": ("model.whistle_finetune", "build_main"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        return
    args = parser().parse_args()
    if not args.command:
        print(HELP)
        return
    if args.command == "whistle" and args.verb not in SPEECH:
        raise SystemExit("cactus whistle playground | compare | finetune | build")
    module, function = SPEECH[args.verb] if args.command == "whistle" else COMMANDS[args.command]
    if module.startswith("model."):
        from .logging import configure
        configure()
    from .._telemetry import track
    track("cli:" + args.command, {})
    getattr(import_module(".." + module, __package__), function)(args)
