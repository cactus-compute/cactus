# Hybrid Handoff — OpenTUI shell

A terminal-native "hybrid inference console": send a task to an on-device agent and
watch **each turn route local vs cloud** — with the below-threshold **escalation**
visual front and center.

Built with [OpenTUI](https://opentui.com) (Zig core + React reconciler on Bun). It is a
**pure presentation client**: all model work happens in Python (`../hybrid_handoff/`),
because the probe `confidence` + cloud token usage are only exposed through the Cactus
FFI, not the OpenAI HTTP server.

## Layout — a four-region shell
```
┌ HEADER (shared chrome) ───────────────────────────────────────────────┐
│ ❯_  CACTUS HYBRID INFERENCE     SANDBOX: …  · THRESHOLD 35%  · READY    │
├──────────────────────────────────────────┬────────────────────────────┤
│ BODY (sandbox-specific)                  │ TRACE PANEL (shared aside)  │
│ email triage view, derived from turns    │ stats + per-turn local↔cloud│
│                                          │ handoff timeline            │
├──────────────────────────────────────────┴────────────────────────────┤
│ FOOTER — command input: prompt · /slash commands · ⌃K palette          │
└────────────────────────────────────────────────────────────────────────┘
```
- **Header, footer, and the trace panel are shared chrome** owned by `TuiShell`.
  Sandboxes own **only the body** (`components/sandboxes/`).
- The trace panel is a fixed right rail (~52 cols). On a **narrow terminal** (`< 120`
  cols) it drops **under** the body — header/footer stay pinned, body + trace scroll
  internally.
- **The trace panel is the point:** each turn shows a segmented confidence meter with the
  amber threshold marker. A turn that routed to cloud renders as the **escalation pair** —
  an on-device probe flagged `⚠ BELOW THRESHOLD → ESCALATE`, then an indented `↳ ☁ CLOUD`
  follow-up. Above-threshold turns show `✓ CLEARED THRESHOLD`.

## Run
Prereqs: the repo's `venv` with the Cactus package built (`cactus build --python`),
the `weights/gemma-4` bundle, and [Bun](https://bun.sh).

```bash
cd demos/hybrid_handoff_tui
bun install
bun start
```

### Controls
- Type a **prompt** + Enter → runs it for the active sandbox.
- **Slash commands:** `/email` `/expense` `/calendar` `/incident` switch sandbox;
  `/run` runs the sandbox's default task; `/reset` clears the run.
- **⌃K** opens the command palette (sandboxes + commands; ↑↓ + Enter).
- **⌃T** cycles the threshold mode LOCAL (0%) / HYBRID (35%) / CLOUD (100%).
- **⌃C** / **Esc** quits.

### Sandboxes
All four Python sandboxes (email/calendar/expense/incident) are wired and switch
instantly (the gemma-4 model stays resident). Only **email** has a custom body view so
far; the others show a placeholder body while the trace panel works fully. Adding a body
later = append one entry to `src/lib/sandboxes.ts`.

### Colors (locked semantics)
green = brand + on-device · cyan = cloud only · amber = threshold / escalation · dim =
secondary. (CRT scanline/glow effects from the web mock don't port to a cell terminal;
we keep square ASCII boxes, monospace, and the blinking input cursor.)

## Verify (headless, no terminal needed)
```bash
bun run verify.tsx mock     # scripted events — asserts all four regions + escalation + palette
bun run verify.tsx narrow   # width 90 — asserts the trace stacks under the body
bun run verify.tsx live      # real gemma-4 through the shell
```

## Files
- `src/app.tsx` — thin: builds the snapshot from bridge events, parses commands.
- `src/components/shell/` — `TuiShell`, `Header`, `Footer`, `CommandPalette`.
- `src/components/trace/` — `TracePanel`, `TraceStats`, `TraceEntry` (escalation pair).
- `src/components/tui/` — `ConfidenceMeter`, `RouteBadge`, `AsciiBox`.
- `src/components/sandboxes/` — `EmailBody`, `GenericBody` (fallback).
- `src/lib/` — `sandboxes.ts` (registry), `snapshot.ts` (snapshot/totals + the body contract).
- `src/bridge.ts` — spawns `../hybrid_handoff/bridge.py`, NDJSON in/out (incl. sandbox switch).
