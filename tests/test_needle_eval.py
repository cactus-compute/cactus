#!/usr/bin/env python3
"""
Needle function-calling evaluation suite.

Usage:
    source ./setup
    cactus build --python

    # Export weights (use correct tokenizer for SFT checkpoints):
    python python/src/needle.py weights/my-checkpoint \
        --checkpoint-path /path/to/sft_12_512_best.pkl \
        --checkpoint-file needle_12_512_best.pkl \
        --tokenizer-revision 5a50f268260b546cbcff02a2b5d4e1a51ac03ef1 \
        --precision FP16

    # Run eval:
    python test_needle_eval.py weights/my-checkpoint

    # Compare multiple checkpoints:
    python test_needle_eval.py weights/sft-1ep weights/sft-3ep weights/pretrain
"""

import json
import sys

sys.path.insert(0, "python")
from src.cactus import cactus_init, cactus_destroy, cactus_complete

# ── Tools (short names, minimal descriptions — best-performing format) ──

TOOLS = json.dumps([
    {"type": "function", "function": {"name": "take_note",
        "description": "Save text as a note",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "The note text"}
        }, "required": ["text"]}}},
    {"type": "function", "function": {"name": "create_reminder",
        "description": "Set a reminder with a time",
        "parameters": {"type": "object", "properties": {
            "message": {"type": "string", "description": "What to remember"},
            "time": {"type": "string", "description": "When"}
        }, "required": ["message"]}}},
    {"type": "function", "function": {"name": "set_timer",
        "description": "Start a countdown timer",
        "parameters": {"type": "object", "properties": {
            "duration": {"type": "string", "description": "How long"}
        }, "required": ["duration"]}}},
    {"type": "function", "function": {"name": "set_alarm",
        "description": "Set a wake-up alarm",
        "parameters": {"type": "object", "properties": {
            "time": {"type": "string", "description": "What time"}
        }, "required": ["time"]}}},
    {"type": "function", "function": {"name": "add_to_list",
        "description": "Add item to a list",
        "parameters": {"type": "object", "properties": {
            "list_name": {"type": "string", "description": "Which list"},
            "item": {"type": "string", "description": "What to add"}
        }, "required": ["list_name", "item"]}}},
    {"type": "function", "function": {"name": "send_message",
        "description": "Send a text message",
        "parameters": {"type": "object", "properties": {
            "contact": {"type": "string", "description": "Who"},
            "text": {"type": "string", "description": "What to say"}
        }, "required": ["contact", "text"]}}},
])

OPTIONS = json.dumps({"temperature": 0.0, "top_p": 0.95, "top_k": 40, "max_tokens": 256})

# ── Test cases ──────────────────────────────────────────────────────
# (query, expected_tool or None for no-call, required_params or [])

TESTS = [
    # -- Reminders --
    ("Remind me at 3pm to pick up the kids",                     "create_reminder", ["message"]),
    ("Remember to send a contract to JP tomorrow",               "create_reminder", ["message"]),
    ("Remind me in 6 hours to add sunscreen to the packing list","create_reminder", ["message"]),
    ("Remember to check in with the keyboard guy tomorrow at 10am","create_reminder",["message"]),
    ("Remind me to call the dentist next Monday",                "create_reminder", ["message"]),
    ("Remember to buy more milk on the way home",                "create_reminder", ["message"]),

    # -- Timers --
    ("Set a timer for 25 minutes",       "set_timer", ["duration"]),
    ("Start a timer for 45 minutes",     "set_timer", ["duration"]),
    ("Timer for 10 minutes",             "set_timer", ["duration"]),

    # -- Alarms --
    ("Set an alarm for 7am",             "set_alarm", ["time"]),
    ("Set an alarm for 6:30 in the morning", "set_alarm", ["time"]),
    ("Wake me up at 8am tomorrow",       "set_alarm", ["time"]),

    # -- Lists (multi-param) --
    ("Add sunscreen to my shopping list", "add_to_list", ["list_name", "item"]),
    ("Add eggs and butter to the grocery list", "add_to_list", ["list_name", "item"]),

    # -- Messages (multi-param) --
    ("Send a message to Liz saying where are you guys",  "send_message", ["contact", "text"]),
    ("Message Eric and tell him I'm running late",        "send_message", ["contact", "text"]),

    # -- Should NOT call a tool (expect [] — app falls back to note) --
    ("The wifi password for the office is cactus2024",    None, []),
    ("Meeting went well, design team approved mockups",   None, []),
    ("Liz recommended a book called Pandora's Star",      None, []),
    ("The restaurant on 5th has great tacos",             None, []),
    ("Tom's phone number is 555-0123",                    None, []),
    ("I had a really productive day today",               None, []),
    ("Coffee order: two oat lattes one espresso",         None, []),
    ("The parking spot is level 3 row F",                 None, []),
]


def run_eval(weights_dir):
    model = cactus_init(weights_dir, None, False)

    tool_total, tool_ok, param_ok = 0, 0, 0
    nocall_total, nocall_ok = 0, 0
    rows = []

    for query, expected, req_params in TESTS:
        messages = json.dumps([{"role": "user", "content": query}])
        response = cactus_complete(model, messages, OPTIONS, TOOLS, None)

        try:
            result = json.loads(response)
            fc = result.get("function_calls", [])
        except:
            fc = []

        called = len(fc) > 0
        got_tool, got_args = None, {}
        if called:
            call = fc[0] if isinstance(fc[0], dict) else json.loads(fc[0])
            got_tool = call.get("name", "")
            got_args = call.get("arguments", {})
            if isinstance(got_args, str):
                try: got_args = json.loads(got_args)
                except: got_args = {}

        if expected is not None:
            # Should call a tool
            tool_total += 1
            t_ok = got_tool == expected
            p_ok = t_ok and all(p in got_args for p in req_params)
            if t_ok: tool_ok += 1
            if p_ok: param_ok += 1

            if t_ok:
                mark = "✓✓" if p_ok else "✓✗"
            elif called:
                mark = "✗~"  # wrong tool
            else:
                mark = "✗✗"  # missed

            args_s = ", ".join(f'{k}="{str(v)[:22]}"' for k, v in got_args.items())
            call_s = f"{got_tool}({args_s})" if got_tool else "[] MISS"
            rows.append((mark, query, expected, call_s))
        else:
            # Should NOT call
            nocall_total += 1
            if not called or got_tool == "take_note":
                nocall_ok += 1
                if called:
                    rows.append(("✓ ", query, "no-call", f"take_note (OK)"))
                else:
                    rows.append(("✓ ", query, "no-call", "[] (OK)"))
            else:
                args_s = ", ".join(f'{k}="{str(v)[:18]}"' for k, v in got_args.items())
                rows.append(("✗ ", query, "no-call", f"{got_tool}({args_s}) FALSE+"))

    cactus_destroy(model)
    return tool_ok, tool_total, param_ok, nocall_ok, nocall_total, rows


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <weights_dir> [weights_dir2] ...")
        sys.exit(1)

    all_results = []

    for weights_dir in sys.argv[1:]:
        print(f"\n{'='*75}")
        print(f"  {weights_dir}")
        print(f"{'='*75}")

        tok, ttot, pok, nok, ntot, rows = run_eval(weights_dir)

        for mark, query, expected, call_s in rows:
            q = query[:50] + ("..." if len(query) > 50 else "")
            print(f"  {mark} {q}")
            print(f"      -> {call_s[:65]}")

        combined = tok + nok
        total = ttot + ntot
        print(f"\n  {'━'*71}")
        print(f"  Tool correct:   {tok}/{ttot} ({100*tok//ttot if ttot else 0}%)")
        print(f"  Params correct: {pok}/{ttot} ({100*pok//ttot if ttot else 0}%)")
        print(f"  No-call correct:{nok}/{ntot} ({100*nok//ntot if ntot else 0}%)")
        print(f"  Combined:       {combined}/{total} ({100*combined//total if total else 0}%)")
        print(f"  {'━'*71}")

        all_results.append((weights_dir, tok, ttot, pok, nok, ntot))

    if len(all_results) > 1:
        print(f"\n{'='*75}")
        print(f"  COMPARISON")
        print(f"{'='*75}")
        print(f"  {'Checkpoint':<35s} {'Tool':>7s} {'Params':>7s} {'NoCal':>7s} {'Total':>7s}")
        print(f"  {'─'*35} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
        for wd, tok, ttot, pok, nok, ntot in all_results:
            name = wd.split("/")[-1][:35]
            comb = tok + nok
            tot = ttot + ntot
            print(f"  {name:<35s} {tok}/{ttot:>2}    {pok}/{ttot:>2}    {nok}/{ntot:>2}    {comb}/{tot} ({100*comb//tot}%)")


if __name__ == "__main__":
    main()
