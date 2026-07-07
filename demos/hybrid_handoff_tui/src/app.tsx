import { useKeyboard } from "@opentui/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Bridge } from "./bridge";
import { TuiShell } from "./components/shell/TuiShell";
import { computeTotals, type Snapshot, type Status } from "./lib/snapshot";
import { getSandbox, sandboxBySlash } from "./lib/sandboxes";
import { MODES } from "./theme";
import type { BridgeEvent, Command, InboxItem, TurnEvent } from "./types";

export interface BridgeLike {
  send(cmd: Command): void;
  quit(): void;
}

export type BridgeFactory = (onEvent: (event: BridgeEvent) => void) => BridgeLike;

export function App({ createBridge }: { createBridge?: BridgeFactory } = {}) {
  const bridgeRef = useRef<BridgeLike | null>(null);
  const [sandboxId, setSandboxId] = useState("email");
  const [modeIdx, setModeIdx] = useState(1);
  const [lambdaOverride, setLambdaOverride] = useState<number | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [inbox, setInbox] = useState<InboxItem[]>([]);
  const [turns, setTurns] = useState<TurnEvent[]>([]);
  const [inflight, setInflight] = useState<number | null>(null);
  const [finalText, setFinalText] = useState("");
  const [statusLine, setStatusLine] = useState("");
  const [query, setQuery] = useState("");
  const [defaultQuery, setDefaultQuery] = useState("");
  const mode = MODES[modeIdx];
  const threshold = lambdaOverride ?? mode.threshold;

  useEffect(() => {
    const handle = (event: BridgeEvent) => {
      switch (event.type) {
        case "loading":
          setStatus("loading");
          break;
        case "ready":
          setStatus("ready");
          setSandboxId(event.sandbox);
          setInbox(event.inbox);
          setDefaultQuery(event.default_query);
          break;
        case "turn_start":
          setStatus("running");
          setInflight(event.index);
          break;
        case "turn":
          setTurns((prev) => [...prev, event]);
          setInflight(null);
          break;
        case "final":
          setFinalText(event.text);
          break;
        case "run_complete":
          setStatus("ready");
          setInflight(null);
          break;
        case "error":
          setStatusLine(`⚠ ${event.message}`);
          setStatus("ready");
          setInflight(null);
          break;
      }
    };
    const make: BridgeFactory = createBridge ?? ((onEvent) => new Bridge(onEvent));
    const bridge = make(handle);
    bridgeRef.current = bridge;
    const cleanup = () => bridge.quit();
    process.on("exit", cleanup);
    return () => {
      process.off("exit", cleanup);
      bridge.quit();
    };
  }, []);

  useKeyboard((key) => {
    if (key.name === "t" && key.ctrl) {
      setModeIdx((m) => (m + 1) % MODES.length);
      setLambdaOverride(null); // cycling modes resets any /lambda override
    }
  });

  const clearRun = () => {
    setTurns([]);
    setFinalText("");
    setInflight(null);
    setQuery("");
  };

  const startRun = (text: string, note: string) => {
    clearRun();
    setQuery(text || defaultQuery);
    setStatus("running");
    setStatusLine(note);
    bridgeRef.current?.send({ type: "query", text, threshold });
  };

  const onCommand = (text: string) => {
    if (text.startsWith("/")) {
      const token = text.split(/\s+/)[0];
      const sb = sandboxBySlash(token);
      if (sb) {
        clearRun();
        setSandboxId(sb.id);
        setStatusLine(`→ switched to ${sb.label}`);
        bridgeRef.current?.send({ type: "sandbox", name: sb.id });
        return;
      }
      if (token === "/run") {
        startRun("", `› running default task · ${getSandbox(sandboxId).label}`);
        return;
      }
      if (token === "/reset") {
        clearRun();
        setStatusLine("↺ reset");
        return;
      }
      if (token === "/lambda") {
        const val = Number(text.split(/\s+/)[1]);
        if (Number.isNaN(val)) {
          setStatusLine("usage: /lambda <number>   (advantage handoff threshold; lower = more cloud)");
          return;
        }
        setLambdaOverride(val);
        setStatusLine(`λ set to ${val.toFixed(2)} · hands off when cloud advantage > ${val.toFixed(2)}`);
        return;
      }
      setStatusLine(`unknown command: ${token}`);
      return;
    }
    startRun(text, `› prompt sent to ${getSandbox(sandboxId).label}`);
  };

  const onQuit = () => {
    bridgeRef.current?.quit();
    process.exit(0);
  };

  const snapshot: Snapshot = useMemo(
    () => ({
      sandboxId,
      threshold,
      status,
      query,
      inbox,
      completedTurns: turns,
      activeTurn: inflight,
      finalText,
      totals: computeTotals(turns),
    }),
    [sandboxId, threshold, status, query, inbox, turns, inflight, finalText],
  );

  return <TuiShell snapshot={snapshot} statusLine={statusLine} onCommand={onCommand} onQuit={onQuit} />;
}
