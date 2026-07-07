import { COLORS, routeColor } from "../../theme";
import type { TurnEvent } from "../../types";
import { argsPreview, truncate, useTween } from "../../util";
import { AdvantageMeter } from "../tui/ConfidenceMeter";
import { RouteBadge } from "../tui/RouteBadge";

const pad = (n: number) => String(n).padStart(2, "0");

export function TraceEntry({ turn, n }: { turn: TurnEvent; n: number }) {
  const appear = useTween(1, 200);
  const marginLeft = Math.round((1 - appear) * 3);
  const cloud = turn.location === "cloud";
  const escalated = turn.advantage !== null && turn.advantage > turn.threshold;
  const color = routeColor(turn.location);
  const toolLine = turn.tool_name
    ? `${turn.tool_name}(${argsPreview(turn.tool_args)})`
    : "synthesize final answer";
  const snippet = turn.tool_result_snippet || truncate(turn.content_snippet, 110);

  return (
    <box
      style={{
        flexDirection: "column",
        border: true,
        borderColor: color,
        backgroundColor: COLORS.panel,
        marginLeft,
        marginBottom: 1,
        paddingX: 1,
      }}
    >
      <box style={{ flexDirection: "row", justifyContent: "space-between" }}>
        <box style={{ flexDirection: "row" }}>
          <text fg={COLORS.dim}>{`${pad(n)} `}</text>
          <RouteBadge location={turn.location} />
        </box>
        <text fg={COLORS.faint}>{turn.tool_name ? "TOOL" : "LLM"}</text>
      </box>
      <text fg={COLORS.text}>{turn.kind}</text>
      <text fg={COLORS.dim}>{toolLine}</text>
      {snippet ? <text fg={COLORS.faint}>{snippet}</text> : null}
      <AdvantageMeter advantage={turn.advantage} threshold={turn.threshold} />
      {escalated ? (
        <text fg={COLORS.threshold}>{"⚠ ADVANTAGE > λ → ESCALATED TO CLOUD"}</text>
      ) : cloud ? (
        <text fg={COLORS.cloud}>{"☁ ROUTED TO CLOUD"}</text>
      ) : (
        <text fg={COLORS.local}>{"✓ ADVANTAGE ≤ λ · KEPT ON-DEVICE"}</text>
      )}
    </box>
  );
}
