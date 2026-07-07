/** TypeScript mirror of the NDJSON protocol emitted by demos/hybrid_handoff/bridge.py. */

export interface InboxItem {
  id: string;
  from: string;
  subject: string;
  date: string;
  unread: boolean;
}

export interface LoadingEvent {
  type: "loading";
}

export interface ReadyEvent {
  type: "ready";
  sandbox: string;
  default_query: string;
  tools: string[];
  inbox: InboxItem[];
}

export interface TurnStartEvent {
  type: "turn_start";
  index: number;
}

export interface TurnEvent {
  type: "turn";
  index: number;
  status: string;
  kind: string;
  location: "local" | "cloud";
  confidence: number | null;
  advantage: number | null;
  threshold: number;
  reason: string;
  latency_ms: number;
  tokens_local: number;
  tokens_cloud_prompt: number;
  tokens_cloud_out: number;
  tokens_cloud: number;
  cost_usd: number;
  content_snippet: string;
  tool_name: string | null;
  tool_args: Record<string, unknown> | null;
  error: string | null;
  tool_result_snippet?: string;
}

export interface FinalEvent {
  type: "final";
  text: string;
}

export interface RunCompleteEvent {
  type: "run_complete";
  turns_local: number;
  turns_cloud: number;
  tokens_local_total: number;
  tokens_cloud_total: number;
  cost_usd_total: number;
  latency_ms_total: number;
  pct_local: number;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type BridgeEvent =
  | LoadingEvent
  | ReadyEvent
  | TurnStartEvent
  | TurnEvent
  | FinalEvent
  | RunCompleteEvent
  | ErrorEvent;

export type Command =
  | { type: "query"; text: string; threshold: number }
  | { type: "sandbox"; name: string }
  | { type: "quit" };
