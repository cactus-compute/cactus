# Cactus Hybrid Handoff — live demo

A multi-turn agent runs inside a mock **sandbox** (email, calendar, expense,
incident) and the terminal shows, **per turn**, whether the Cactus engine
answered **on-device** or escalated to the **cloud** — driven by the Gemma-4 E2B
handoff probe (`confidence = 1 − p_wrong`). Each agent turn is one
`cactus_complete()` call, so the local↔cloud decision is made independently every
step.

```
╭─ ☁ ESCALATED TO CLOUD ───────────────────────────────────────────────╮
│ #0  search_inbox  ☁ CLOUD                        2568ms  ↑707 tok  $0.0003 │
│ ███───┃────────────  conf 0.19  thr 0.35                                   │
│   → search_inbox(sender='Jake', query='meeting')                          │
│   reason: low confidence (probe)                                           │
╰───────────────────────────────────────────────────────────────────────╯
...
╭─ totals ──────────────────────────────────────────────────────────────╮
│ turns 4 local / 1 cloud   on-device 80%   tokens 3892/707   cost $0.0003 │
╰───────────────────────────────────────────────────────────────────────╯
```

## Setup (one time)

```bash
source ./setup                 # activates ./venv with the cactus package + FFI lib
cp demos/hybrid_handoff/.env.example demos/hybrid_handoff/.env
# put your CACTUS_CLOUD_KEY in that .env  (it is gitignored)
```

Requires the **probe bundle** at `weights/gemma-4` (the only bundle that ships
`handoff_probe.bin`; `gemma-4-e2b-it*` fall back to entropy — do not use them).
`rich` is already in the venv.

## Run it

`--threshold` IS the mode. Run three terminals side by side:

```bash
# pure on-device (baseline: everything local, $0)
./venv/bin/python demos/hybrid_handoff/run.py --sandbox email --threshold 0.0

# probe hybrid (the demo: a mix of on-device + cloud)
./venv/bin/python demos/hybrid_handoff/run.py --sandbox email --threshold 0.35

# forced cloud (ceiling: every turn to the cloud)
./venv/bin/python demos/hybrid_handoff/run.py --sandbox email --threshold 1.0
```

Open-ended queries work — the tools really search the data pool:

```bash
./venv/bin/python demos/hybrid_handoff/run.py --sandbox email --threshold 0.35 \
  --query "Summarize my unread emails about the offsite and say who to reply to."
```

Sandboxes: `--sandbox {email,calendar,expense,incident}` (`--list-sandboxes`).

## Picking the threshold (important)

On agentic/tool turns this E2B probe runs **pessimistic**: per-turn confidence
sits around **0.19–0.46** (it's least sure on early, low-context turns and most
sure once it has gathered data). Consequences:

- `0.0` → never escalates (pure on-device).
- `0.5`+ → escalates *almost everything* (it looks like pure cloud).
- **`~0.30–0.40` is the useful hybrid band** — that's where you get a real mix.
  Start at **0.35**. Lower → more on-device (cheaper); higher → more cloud.

This is the honest behavior of the trained probe, not a demo trick — tune
`--threshold` live to show the cost/quality knob.

## How it works

- `run.py` loads `weights/gemma-4` once and owns the Rich `Live` TUI.
- `agent.py` runs the loop: each turn calls `cactus_complete(..., confidence_threshold=T)`;
  the engine decides local vs cloud and returns `cloud_handoff`, `confidence`,
  `cloud_handoff_reason`, and (via our engine patch) `cloud_prompt_tokens` /
  `cloud_completion_tokens` for exact cost. Tool calls are dispatched to the
  sandbox, which returns **real queried data**, fed back for the next turn.
- A loop-guard + forced finalize guarantees a closing synthesis turn (small local
  models otherwise loop on tool calls).
- `cost.py` prices cloud tokens (gemini-2.5-flash); `tui.py` renders the timeline.

## Extending

- **Ask anything:** just change `--query`.
- **Add a tool:** add a `ToolSpec` (schema + a Python fn that filters the pool)
  in the sandbox's `__init__.py`.
- **Add data:** edit the JSON under `sandboxes/<name>/data/`.
- **Add a sandbox:** new `sandboxes/<name>/__init__.py` subclassing `Sandbox`,
  then register it in `sandboxes/__init__.py`.

## Relationship to `benchmarks/handoff_multistep/`

Same idea (multi-turn agent + the probe), but the benchmark uses **static**
`tool_stubs` (one fixed result per tool) — not query-extensible — and teacher-
forced scoring. These sandboxes are **dynamic** mock environments. To run a
sandbox through the benchmark harness, see `benchmark_adapter.py`.

Existing benchmark tasks that already produce a local/cloud split and work as
*canned* demos (not open-ended): `conflict_reschedule`, `meeting_gap`,
`picnic_weather`, `cheapest_flight`, `priority_pick`, `weather_pack`.

## Engine patch (cloud cost)

`return_cloud_completion` / `construct_response_json` now surface
`cloud_prompt_tokens` + `cloud_completion_tokens` parsed from the cloud
endpoint's `token_usage` block (`cactus-engine/src/{cloud.h,cloud.cpp,complete.cpp,utils.h}`).
Rebuild with `cactus build --python` after pulling. Without the patch, `cost.py`
falls back to estimating output tokens from response length.
