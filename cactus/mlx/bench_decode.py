"""Measure batch-one prefill and warmed decode on a fixed workload."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import mlx.core as mx
from .runtime import load_runtime


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--config-dir')
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--prompt-tokens', type=int, help='Exact prompt length (at least 64)')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--mtp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--num-draft-tokens', type=int, default=3, choices=range(1,8))
    p.add_argument('--mtp-min-prob', type=float, default=0.8)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.tokens < 1 or args.repeats < 1:
        p.error('--tokens and --repeats must be positive')
    if args.prompt_tokens is not None and args.prompt_tokens < 64:
        p.error('--prompt-tokens must be at least 64')
    model, tokenizer = load_runtime(args.model, args.config_dir, progress=lambda s: print(s, flush=True))
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Write a Python function is_prime(n) and explain how it works.'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if args.prompt_tokens:
        context = 'A prime number has exactly two positive divisors. Trial division checks possible factors up to the square root. '
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content':
            context * args.prompt_tokens + '\nWrite a Python function is_prime(n) and explain how it works.'}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = tokens[:args.prompt_tokens - 64] + tokens[-64:]

    # Compile the prefill kernels and decode graph before timed trials.
    run_trial = run_mtp if args.mtp else run
    extra = dict(draft_tokens=args.num_draft_tokens, min_draft_prob=args.mtp_min_prob) if args.mtp else {}
    warmup = run_trial(model, tokenizer, tokens[:64], 1, **extra)
    del warmup
    trials = []
    for i in range(args.repeats):
        gc.collect()
        trial = run_trial(model, tokenizer, tokens, args.tokens, **extra)
        trials.append(trial)
        print(json.dumps({'trial': i + 1, 'prefill_tokens_per_s': trial['prefill_tokens_per_s'],
                          'decode_tokens_per_s': trial['tokens_per_s']}), flush=True)
    report = {
        'device': mx.device_info()['device_name'], 'batch_size': 1,
        'prompt_tokens': len(tokens), 'prefill_step': 4, 'decode_warmup_steps': 8,
        'warmup_unit': 'speculative blocks' if args.mtp else 'tokens',
        'mtp': args.mtp, 'num_draft_tokens': args.num_draft_tokens if args.mtp else 0,
        'min_draft_prob': args.mtp_min_prob if args.mtp else None,
        'tokens': args.tokens, 'repeats': args.repeats,
        'prefill_tokens_per_s': statistics.median(t['prefill_tokens_per_s'] for t in trials),
        'tokens_per_s': statistics.median(t['tokens_per_s'] for t in trials),
        'peak_memory_gib': mx.get_peak_memory() / 2**30,
        'memory_scope': 'MLX allocations; measure process footprint separately',
        'trials': trials,
    }
    print(json.dumps({k: v for k, v in report.items() if k != 'trials'}), flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


def run(model, tokenizer, tokens, decode_tokens):
    cache = model.make_cache()
    mx.eval([c.state for c in cache])
    started = time.perf_counter()
    for start in range(0, len(tokens), 4):
        logits = model(mx.array([tokens[start:start+4]]), cache=cache)[:, -1, :]
        mx.eval(logits, [c.state for c in cache])
    prefill_s = time.perf_counter() - started
    token = int(mx.argmax(logits[0]).item())
    generated = [token]
    for _ in range(8):
        logits = model(mx.array([[token]]), cache=cache)
        token = int(mx.argmax(logits[0,-1]).item())
        generated.append(token)
    elapsed, build = [], []
    for _ in range(decode_tokens):
        t = time.perf_counter()
        logits = model(mx.array([[token]]), cache=cache)
        build.append(time.perf_counter() - t)
        token = int(mx.argmax(logits[0,-1]).item())
        elapsed.append(time.perf_counter() - t)
        generated.append(token)
    return {'tokens': decode_tokens, 'seconds': sum(elapsed),
              'prefill_s': prefill_s, 'prefill_tokens_per_s': len(tokens) / prefill_s,
              'tokens_per_s': decode_tokens / sum(elapsed), 'step_ms': [1000*t for t in elapsed],
              'graph_build_ms': sum(build) / len(build) * 1000, 'compiled': True,
              'text': tokenizer.decode(generated), 'token_ids': generated,
              'peak_memory_gib': mx.get_peak_memory() / 2**30}


def run_mtp(model, tokenizer, tokens, decode_tokens, *, draft_tokens=3, min_draft_prob=0.8):
    from .models.nemotron_speculative import MTPDecoder
    decoder = MTPDecoder(model, draft_tokens=draft_tokens, min_draft_prob=min_draft_prob)
    started = time.perf_counter()
    seed, _ = decoder.prefill(tokens)
    prefill_s = time.perf_counter() - started
    generated = [seed]
    for _ in range(8):
        block, _ = decoder.step()
        generated.extend(block)
        decoder.finish_prefix(len(block))
    warmup_tokens = len(generated) - 1
    for key in decoder.stats:
        decoder.stats[key] = 0
    elapsed, count = [], 0
    while count < decode_tokens:
        started = time.perf_counter()
        block, _ = decoder.step()
        emitted = min(len(block), decode_tokens - count)
        decoder.finish_prefix(emitted)
        elapsed.append(time.perf_counter() - started)
        count += emitted
        generated.extend(block[:emitted])
    return {'tokens': count, 'seconds': sum(elapsed), 'prefill_s': prefill_s,
            'prefill_tokens_per_s': len(tokens) / prefill_s,
            'tokens_per_s': count / sum(elapsed), 'step_ms': [1000*t for t in elapsed],
            'compiled': True, 'warmup_tokens': warmup_tokens, 'mtp_stats': decoder.stats,
            'text': tokenizer.decode(generated), 'token_ids': generated,
            'peak_memory_gib': mx.get_peak_memory() / 2**30}


if __name__ == '__main__':
    main()
