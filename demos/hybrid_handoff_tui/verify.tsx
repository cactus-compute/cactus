/** Headless verification of the four-region shell using OpenTUI's test renderer.
 *
 *   bun run verify.tsx mock   # scripted events, deterministic, no engine
 *   bun run verify.tsx live   # real bridge + gemma-4 through the shell
 */
import { testRender } from "@opentui/react/test-utils";
import { App, type BridgeLike } from "./src/app";
import { Bridge } from "./src/bridge";
import type { BridgeEvent } from "./src/types";

const READY: BridgeEvent = {
  type: "ready",
  sandbox: "email",
  default_query: "Reply to Jake about the meeting; check my calendar for a conflict.",
  tools: ["search_inbox", "read_email", "get_calendar", "search_docs"],
  inbox: [
    { id: "m001", from: "Jake Tran <jake@partnerco.com>", subject: "Re: Our meeting + the Cactus Hybrid discussion", date: "2026-06-22T09:14:00", unread: true },
    { id: "m005", from: "Priya Nair <priya@cactus.ai>", subject: "Q3 budget review — numbers needed", date: "2026-06-20T08:05:00", unread: true },
    { id: "m009", from: "billing@cloudvendor.example", subject: "Your May invoice is ready", date: "2026-06-17T03:11:00", unread: false },
  ],
};

function turn(i: number, location: "local" | "cloud", tool: string | null, adv: number): BridgeEvent {
  const cloud = location === "cloud";
  return {
    type: "turn",
    index: i,
    status: "done",
    kind: tool ?? "FINAL",
    location,
    confidence: null,
    advantage: adv,
    threshold: 0.05,
    reason: cloud ? "high cloud advantage (probe)" : "advantage below threshold",
    latency_ms: cloud ? 2300 : 1900,
    tokens_local: cloud ? 0 : 660,
    tokens_cloud_prompt: cloud ? 680 : 0,
    tokens_cloud_out: cloud ? 18 : 0,
    tokens_cloud: cloud ? 698 : 0,
    cost_usd: cloud ? 0.00025 : 0,
    content_snippet: tool ? "" : "Here is the drafted reply and the 2pm calendar conflict…",
    tool_name: tool,
    tool_args: tool === "read_email" ? { id: "m001" } : tool ? { sender: "Jake", query: "meeting" } : null,
    error: null,
    tool_result_snippet: tool ? '{"count": 1, "emails": [{"id": "m001"}]}' : undefined,
  };
}

function assert(cond: boolean, msg: string): void {
  if (!cond) {
    console.error(`  ✗ ${msg}`);
    process.exitCode = 1;
  } else {
    console.error(`  ✓ ${msg}`);
  }
}

async function mountApp(width: number, height: number) {
  let emit: ((e: BridgeEvent) => void) | null = null;
  const factory = (onEvent: (e: BridgeEvent) => void): BridgeLike => {
    emit = onEvent;
    return { send() {}, quit() {} };
  };
  const t = await testRender(<App createBridge={factory} />, { width, height });
  for (let i = 0; i < 60 && !emit; i += 1) {
    await t.flush();
    await Bun.sleep(5);
  }
  if (!emit) throw new Error("App effect never ran (emit unset)");
  const settle = async () => {
    await t.flush();
    await Bun.sleep(10);
    await t.flush();
  };
  const send = async (e: BridgeEvent) => {
    emit!(e);
    await settle();
  };
  return { t, send, settle };
}

async function runMock() {
  const { t, send, settle } = await mountApp(130, 44);

  await send(READY);
  let f = t.captureCharFrame();
  assert(f.includes("CACTUS HYBRID INFERENCE"), "header title canvas");
  assert(f.includes("SANDBOX: EMAIL TRIAGE AGENT"), "header shows active sandbox");
  assert(f.includes("HANDOFF λ 0.05"), "header shows handoff lambda");
  assert(f.includes(">> INBOX"), "body = email inbox");
  assert(f.includes(">> HYBRID INFERENCE TRACE"), "trace panel present");
  assert(f.includes("ON-DEVICE 0") && f.includes("CLOUD 0"), "trace stats header");
  assert(f.includes("cactus:~/email $"), "footer command prompt");

  await send({ type: "turn_start", index: 0 });
  assert(t.captureCharFrame().includes("probing confidence"), "active trace row");

  await send(turn(0, "cloud", "search_inbox", 0.87));
  await send(turn(1, "local", "read_email", 0.03));
  await send(turn(2, "local", null, 0.01));
  f = t.captureCharFrame();
  assert(f.includes("ADVANTAGE > λ → ESCALATED TO CLOUD"), "single card shows escalation inline");
  assert(f.includes("☁ CLOUD"), "escalated turn flips to a cloud card");
  assert(f.includes("✓ ADVANTAGE ≤ λ"), "kept-local flag on low-advantage turn");
  assert(f.includes("adv ") && f.includes("λ 0.05"), "advantage meter with lambda");

  await send({
    type: "final",
    text: "## Summary\n\nI found **3 emails** from Jake:\n\n1. **ID: m001** — Re: Our meeting\n2. **ID: m015** — Re: Intro\n\nNext: reply with `read m001`.",
  });
  await send({ type: "run_complete", turns_local: 2, turns_cloud: 1, tokens_local_total: 1320, tokens_cloud_total: 698, cost_usd_total: 0.00025, latency_ms_total: 6100, pct_local: 66.7 });
  f = t.captureCharFrame();
  assert(f.includes("AGENT RESULT"), "final answer in body");
  assert(f.includes("Summary") && f.includes("3 emails"), "markdown content rendered");
  assert(!f.includes("**") && !f.includes("##"), "markdown markers concealed (no raw **/##)");
  assert(f.includes("ESCALATIONS"), "escalation count in stats");

  await t.mockInput.pressKey("k", { ctrl: true });
  await settle();
  f = t.captureCharFrame();
  assert(f.includes("COMMAND PALETTE"), "⌃K opens command palette");
  assert(f.includes("/calendar") && f.includes("/expense"), "palette lists sandboxes");

  console.error("\n----- captured frame (mock) -----");
  console.log(t.captureCharFrame());
  t.renderer.destroy();
}

async function runNarrow() {
  const { t, send } = await mountApp(90, 50);
  await send(READY);
  await send(turn(0, "cloud", "search_inbox", 0.28));
  const f = t.captureCharFrame();
  assert(f.includes(">> INBOX") && f.includes(">> HYBRID INFERENCE TRACE"), "narrow: body + trace both render (stacked)");
  console.error("\n----- captured frame (narrow 90w) -----");
  console.log(f);
  t.renderer.destroy();
}

async function runLive() {
  let real: Bridge | null = null;
  const factory = (onEvent: (e: BridgeEvent) => void): BridgeLike => {
    real = new Bridge(onEvent);
    return real;
  };
  const t = await testRender(<App createBridge={factory} />, { width: 130, height: 46 });

  const until = async (pred: (f: string) => boolean, ms: number, label: string) => {
    const deadline = Date.now() + ms;
    while (Date.now() < deadline) {
      await t.flush();
      if (pred(t.captureCharFrame())) return true;
      await Bun.sleep(150);
    }
    console.error(`  ✗ timeout: ${label}`);
    process.exitCode = 1;
    return false;
  };

  if (await until((f) => f.includes("READY"), 60000, "model ready")) {
    console.error("  ✓ model loaded, shell ready");
    real!.send({ type: "query", text: "Reply to Jake about the meeting; check my calendar for a conflict and pull the context doc.", threshold: 0.05 });
    if (await until((f) => f.includes("AGENT RESULT"), 120000, "final answer")) {
      console.error("  ✓ live run produced a final answer through the shell");
    }
    console.error("\n----- captured frame (live) -----");
    console.log(t.captureCharFrame());
  }
  real?.quit();
  t.renderer.destroy();
}

const mode = process.argv[2] || "mock";
if (mode === "live") await runLive();
else if (mode === "narrow") await runNarrow();
else await runMock();
process.exit(process.exitCode || 0);
