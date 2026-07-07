import { COLORS } from "../../theme";
import type { Totals } from "../../lib/snapshot";
import type { TurnEvent } from "../../types";
import { useSpinner } from "../../util";
import { TraceEntry } from "./TraceEntry";
import { TraceStats } from "./TraceStats";

function ActiveRow({ n }: { n: number }) {
  const spin = useSpinner(true);
  return (
    <box
      style={{
        flexDirection: "row",
        border: true,
        borderColor: COLORS.threshold,
        backgroundColor: COLORS.panel,
        marginBottom: 1,
        paddingX: 1,
      }}
    >
      <text fg={COLORS.threshold}>{`${spin} ${String(n).padStart(2, "0")} probing confidence · deciding route…`}</text>
    </box>
  );
}

export function TracePanel({
  completedTurns,
  activeTurn,
  totals,
}: {
  completedTurns: TurnEvent[];
  activeTurn: number | null;
  totals: Totals;
}) {
  const empty = completedTurns.length === 0 && activeTurn === null;
  const turnCount = completedTurns.length + (activeTurn !== null ? 1 : 0);
  return (
    <box style={{ flexDirection: "column", flexGrow: 1, backgroundColor: COLORS.bg }}>
      <box
        style={{
          flexDirection: "column",
          flexShrink: 0,
          border: ["bottom"],
          borderColor: COLORS.border,
          backgroundColor: COLORS.panelAlt,
        }}
      >
        <box style={{ flexDirection: "row", justifyContent: "space-between", paddingX: 1 }}>
          <text fg={COLORS.local}>{">> HYBRID INFERENCE TRACE"}</text>
          <text fg={COLORS.dim}>{`↯ ${turnCount} TURNS`}</text>
        </box>
        <TraceStats totals={totals} />
      </box>
      <scrollbox
        stickyScroll
        stickyStart="bottom"
        scrollY
        style={{ flexGrow: 1, paddingX: 1, paddingTop: 1, backgroundColor: COLORS.bg }}
      >
        {empty ? <text fg={COLORS.faint}>no turns yet — run a task below</text> : null}
        {completedTurns.map((t, i) => (
          <TraceEntry key={i} turn={t} n={i + 1} />
        ))}
        {activeTurn !== null ? <ActiveRow n={activeTurn + 1} /> : null}
      </scrollbox>
    </box>
  );
}
