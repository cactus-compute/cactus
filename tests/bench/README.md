# E2E Decode Benchmark

Minimal Cactus-only decode speed comparison copied forward from the `gemma-assistant-speed-fix` branch.

```bash
./tests/bench_e2e_decode.sh
```

The wrapper sources `./venv/bin/activate`, runs `cactus build`, compiles `tests/bench/e2e_cactus_decode.cpp`, then benchmarks each existing model path in `tests/bench/e2e_models.json`. Missing model directories are skipped.

By default it runs a harder prompt suite that avoids trivial 100% acceptance cases:

```text
json_object
python_function
explain_schrodinger
rank_tradeoffs
debug_plan
constrained_json
code_review
```

Use `--prompt-suite long_context` to measure decode after a roughly 1k-token natural prose prefill. That suite uses a hand-written launch-readiness memo with conflicting facts and a synthesis question so MTP behavior is not dominated by repetitive generated records. `--prompt-suite all` runs both suites.

By default the CSV includes baseline (`0`) and adaptive MTP draft 2 runs with 2 warmups and 5 measured reps. Use `--mtp-max-drafts 0,2,3,4` for sweeps, or `--mtp-max-draft 3` for one setting. For fixed-policy diagnostics, add `--mtp-fixed-draft`. Sampled runs can be measured with `--temperature`, `--top-p`, `--top-k`, `--min-p`, and `--seed`.

Useful overrides:

```bash
./tests/bench_e2e_decode.sh --reps 5 --max-tokens 512 --model qwen3-0.6b
./tests/bench_e2e_decode.sh --config /path/to/models.json --output decode.csv
./tests/bench_e2e_decode.sh --prompt "who are you" --mtp-max-draft 0
./tests/bench_e2e_decode.sh --model gemma-4-e2b-it --prompt-suite long_context --long-context-repeats 192 --max-tokens 256 --mtp-max-drafts 0,3
./tests/bench_e2e_decode.sh --model gemma-4-e2b-it --mtp-max-draft 2 --mtp-fixed-draft
./tests/bench_e2e_decode.sh --model gemma-4-e2b-it --prompt "Count from 1 to 100, separated by commas." --max-tokens 128 --mtp-max-drafts 0,2,3,4,5
./tests/bench_e2e_decode.sh --model gemma-4-e2b-it --temperature 0.7 --top-p 0.95 --top-k 64 --min-p 0 --seed 1234 --mtp-max-draft 3
```

CSV columns:

```text
backend,model,prompt,shape,mtp_max_draft,rep,prefill_tokens,decode_tokens,prefill_tps,decode_tps,ttft_ms,total_ms,mtp_requested,mtp_enabled,mtp_drafted_tokens,mtp_accepted_tokens,mtp_rejected_tokens,mtp_rounds,mtp_fallback_reason,assistant_draft_ms,target_verify_ms,sampling_or_argmax_ms,kv_transaction_ms,callback_stream_ms
```
