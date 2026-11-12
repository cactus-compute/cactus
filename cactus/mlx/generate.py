import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import wired_limit
from .runtime import load_runtime


def main():
    parser = argparse.ArgumentParser(description='Text and image generation directly from the consolidated CQ file')
    parser.add_argument('--model', required=True, help='Local model directory, safetensors file, or Hugging Face repo')
    parser.add_argument('--config-dir', help='Directory with config.json and tokenizer files')
    parser.add_argument('--prompt', default='What is the capital of France? Answer in one sentence.')
    parser.add_argument('--image', action='append', help='Image path, HTTP(S) URL or data URL; repeat for multiple images')
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--prefill-step', type=int, default=4)
    parser.add_argument('--raw', action='store_true', help='Skip chat template')
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--mtp', action=argparse.BooleanOptionalAction, default=True,
                        help='Use verified greedy MTP (default: on)')
    parser.add_argument('--num-draft-tokens', type=int, default=3, choices=range(1, 8),
                        help='Maximum adaptive draft length (default: 3)')
    parser.add_argument('--mtp-min-prob', type=float, default=0.8,
                        help='Minimum draft confidence; 0 disables confidence gating (default: 0.8)')
    parser.add_argument('--output', help='Save generation and timing JSON')
    args = parser.parse_args()
    if args.max_tokens < 1 or args.prefill_step < 1:
        parser.error('max-tokens and prefill-step must be positive')
    if not 0 <= args.mtp_min_prob <= 1:
        parser.error('mtp-min-prob must be between 0 and 1')
    log = lambda message: print(message, file=sys.stderr, flush=True)
    model, tokenizer = load_runtime(args.model, args.config_dir, progress=log)
    with wired_limit(model):
        run(args, model, tokenizer, log)


def run(args, model, tokenizer, log):
    if args.raw:
        prompt = args.prompt
    else:
        content = ([{'type': 'image'} for _ in args.image] + [{'type': 'text', 'text': args.prompt}]
                   if args.image else args.prompt)
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=args.thinking)
    input_embeddings = None
    if args.image:
        prompt, pixels = model.image_processor(prompt, images=args.image)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not tokens:
        parser.error('The prompt must contain at least one token')
    if args.image:
        log(f'Encoding {len(args.image)} image(s)')
        input_embeddings = model.embed_inputs(mx.array(tokens), pixel_values=pixels)
        mx.eval(input_embeddings)
    if args.mtp:
        from .generation import mtp_generate_step
        started = time.perf_counter()
        generated, emitted = [], ''
        iterator = mtp_generate_step(tokens, model, max_tokens=args.max_tokens,
            draft_tokens=args.num_draft_tokens, min_draft_prob=args.mtp_min_prob,
            prefill_step_size=args.prefill_step,
            progress=lambda n, total: log(f'Prefill {n}/{total} tokens'), input_embeddings=input_embeddings)
        try:
            for i, (token, _, _) in enumerate(iterator):
                if i == 0:
                    prefill_s = time.perf_counter() - started
                    decode_started = time.perf_counter()
                if token in tokenizer.eos_token_ids:
                    break
                generated.append(token)
                text = tokenizer.decode(generated)
                if not text.endswith('\ufffd'):
                    print(text[len(emitted):], end='', flush=True)
                    emitted = text
        finally:
            iterator.close()
        decode_s = time.perf_counter() - decode_started
        text = tokenizer.decode(generated)
        print(text[len(emitted):], flush=True)
        report = {'prompt': args.prompt, 'rendered_prompt': prompt, 'text': text,
                  'token_ids': generated, 'prompt_tokens': len(tokens),
                  'prefill_s': prefill_s, 'decode_s': decode_s,
                  'decode_tokens_per_s': max(0, len(generated) - 1) / max(decode_s, 1e-9),
                  'peak_memory_gib': mx.get_peak_memory() / 2**30,
                  'device': mx.device_info()['device_name'], 'mtp': model._last_mtp_stats}
        save_report(report, args.output, log)
        return
    cache = model.make_cache()
    mx.eval([c.state for c in cache])
    started = time.perf_counter()
    for offset in range(0, len(tokens), args.prefill_step):
        chunk = mx.array([tokens[offset:offset + args.prefill_step]])
        embeddings = None if input_embeddings is None else input_embeddings[offset:offset + args.prefill_step][None]
        logits = model(chunk, cache=cache, input_embeddings=embeddings)[:, -1, :]
        mx.eval(logits, [c.state for c in cache])
        log(f'Prefill {min(offset + args.prefill_step, len(tokens))}/{len(tokens)} tokens')
    prefill_s = time.perf_counter() - started
    next_token = int(mx.argmax(logits[0]).item())
    generated = []
    emitted = ''
    decode_started = time.perf_counter()
    for i in range(args.max_tokens):
        if next_token in tokenizer.eos_token_ids:
            break
        generated.append(next_token)
        text = tokenizer.decode(generated)
        if not text.endswith('\ufffd'):
            print(text[len(emitted):], end='', flush=True)
            emitted = text
        if i + 1 < args.max_tokens:
            logits = model(mx.array([[next_token]]), cache=cache)[:, -1, :]
            next_token = int(mx.argmax(logits[0]).item())
    decode_s = time.perf_counter() - decode_started
    text = tokenizer.decode(generated)
    print(text[len(emitted):], flush=True)
    report = {'prompt': args.prompt, 'rendered_prompt': prompt, 'text': text,
              'token_ids': generated, 'prompt_tokens': len(tokens),
              'prefill_s': prefill_s, 'decode_s': decode_s,
              'decode_tokens_per_s': max(0, len(generated) - 1) / max(decode_s, 1e-9),
              'peak_memory_gib': mx.get_peak_memory() / 2**30,
              'device': mx.device_info()['device_name']}
    save_report(report, args.output, log)


def save_report(report, output, log):
    log(json.dumps({k: v for k, v in report.items() if k not in ('text', 'rendered_prompt', 'token_ids')}, indent=2))
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
