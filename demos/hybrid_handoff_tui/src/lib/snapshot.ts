/** The single immutable view the UI renders from. Body components receive a
 * slice of this and nothing else — they never touch the bridge (keeps the
 * engine boundary clean so a different engine could feed the same snapshot). */
import type { InboxItem, TurnEvent } from "../types";

export type Status = "loading" | "ready" | "running";

export interface Totals {
  local: number;
  cloud: number;
  escalations: number;
  tokLocal: number;
  tokCloud: number;
  cost: number;
  latency: number;
}

export interface BodyProps {
  completedTurns: TurnEvent[];
  activeTurn: number | null;
  allTurns: TurnEvent[];
  finalText: string;
  inbox: InboxItem[];
}

export interface Snapshot {
  sandboxId: string;
  threshold: number;
  status: Status;
  query: string;
  inbox: InboxItem[];
  completedTurns: TurnEvent[];
  activeTurn: number | null;
  finalText: string;
  totals: Totals;
}

export function statusLabel(s: Status): string {
  return s === "loading" ? "LOADING" : s === "running" ? "RUNNING" : "READY";
}

export function computeTotals(turns: TurnEvent[]): Totals {
  const t: Totals = {
    local: 0,
    cloud: 0,
    escalations: 0,
    tokLocal: 0,
    tokCloud: 0,
    cost: 0,
    latency: 0,
  };
  for (const x of turns) {
    t.latency += x.latency_ms;
    t.cost += x.cost_usd;
    if (x.location === "cloud") {
      t.cloud += 1;
      t.tokCloud += x.tokens_cloud;
      if (x.confidence !== null && x.confidence < x.threshold) t.escalations += 1;
    } else {
      t.local += 1;
      t.tokLocal += x.tokens_local;
    }
  }
  return t;
}

export function pctLocal(t: Totals): number {
  const total = t.local + t.cloud;
  return total ? Math.round((100 * t.local) / total) : 0;
}
