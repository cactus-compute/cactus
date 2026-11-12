"""Use MLX-LM's HTTP server with the CQ loader and batch-one decoding."""
import gc
import argparse
import functools
import logging
import sys

import mlx.core as mx
from mlx_lm import server

from .runtime import load_runtime


class CQModelProvider(server.ModelProvider):
    def __init__(self, args):
        if not args.model:
            raise ValueError('--model is required: provide a Hugging Face repo ID or local directory')
        if args.kv_bits is not None or args.adapter_path or args.draft_model or args.pipeline:
            raise ValueError('This CQ release supports native KV, no adapters, and no draft model')
        super().__init__(args)
        if self.is_distributed:
            raise ValueError('CQ serving currently supports one Apple GPU')

    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        if model_path != self.cli_args.model or adapter_path or draft_model_path:
            raise ValueError('This server serves only the CQ model selected at startup')
        self.reset()
        gc.collect()
        mx.clear_cache()
        self.model, self.tokenizer = load_runtime(model_path, progress=logging.info)
        self.processor = self.model.image_processor
        self.model_key = (model_path, None, None)
        self.is_batchable = False


def main():
    defaults = {
        '--prefill-step-size': '4',
        '--prompt-cache-size': '1',
        '--chat-template-args': '{"enable_thinking":false}',
        '--num-draft-tokens': '3',
    }
    original_argv = sys.argv[:]
    original_provider = server.ModelProvider
    original_stream = server.stream_generate
    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--mtp', action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument('--mtp-min-prob', type=float, default=0.8)
        mtp_args, remaining = parser.parse_known_args(sys.argv[1:])
        if not 0 <= mtp_args.mtp_min_prob <= 1:
            parser.error('--mtp-min-prob must be between 0 and 1')
        sys.argv[:] = [sys.argv[0], *remaining]
        if '--help' in remaining or '-h' in remaining:
            print('Cactus MLX: greedy MTP is on by default; --no-mtp disables it; --mtp-min-prob sets draft confidence (default: 0.8).')
        if mtp_args.mtp:
            from .generation import stream_generate
            server.stream_generate = functools.partial(stream_generate, mtp=True,
                                                        min_draft_prob=mtp_args.mtp_min_prob)
        for flag, value in defaults.items():
            if not any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:]):
                sys.argv.extend([flag, value])
        server.ModelProvider = CQModelProvider
        server.main()
    finally:
        server.stream_generate = original_stream
        server.ModelProvider = original_provider
        sys.argv[:] = original_argv


if __name__ == '__main__':
    main()
