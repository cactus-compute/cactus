"""Shared command line entry point for Cactus MLX."""
import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(prog='cactus-mlx', description='Run Cactus models with MLX')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('generate', add_help=False, help='Generate text from a model')
    commands.add_parser('server', add_help=False, help='Start the local OpenAI-compatible server')
    args, remaining = parser.parse_known_args()
    from ._environment import require_runtime
    try:
        require_runtime()
    except RuntimeError as error:
        parser.error(str(error))
    original = sys.argv[:]
    try:
        sys.argv[:] = [f'cactus-mlx {args.command}', *remaining]
        importlib.import_module(f'.{args.command}', __package__).main()
    finally:
        sys.argv[:] = original


if __name__ == '__main__':
    main()
