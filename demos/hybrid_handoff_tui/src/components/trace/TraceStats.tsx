import { COLORS } from "../../theme";
import { pctLocal, type Totals } from "../../lib/snapshot";
import { fmtCost } from "../../util";

export function TraceStats({ totals }: { totals: Totals }) {
  const pct = pctLocal(totals);
  return (
    <box style={{ flexDirection: "column", paddingX: 1, paddingTop: 1 }}>
      <box style={{ flexDirection: "row", justifyContent: "space-between" }}>
        <text fg={COLORS.local}>{`ON-DEVICE ${totals.local}`}</text>
        <text fg={COLORS.dim}>{`${pct}% LOCAL`}</text>
        <text fg={COLORS.cloud}>{`CLOUD ${totals.cloud}`}</text>
      </box>
      <box style={{ flexDirection: "row", justifyContent: "space-between" }}>
        <text fg={COLORS.dim}>{`HANDOFFS ${totals.cloud}`}</text>
        <text fg={COLORS.threshold}>{`${totals.escalations} ESCALATIONS`}</text>
        <text fg={COLORS.cloud}>{`spend ${fmtCost(totals.cost)}`}</text>
      </box>
    </box>
  );
}
