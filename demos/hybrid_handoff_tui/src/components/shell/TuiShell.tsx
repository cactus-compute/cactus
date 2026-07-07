import { useKeyboard, useRenderer, useTerminalDimensions } from "@opentui/react";
import { useState } from "react";
import { getSandbox } from "../../lib/sandboxes";
import { type Snapshot, statusLabel } from "../../lib/snapshot";
import { COLORS } from "../../theme";
import { TracePanel } from "../trace/TracePanel";
import { CommandPalette } from "./CommandPalette";
import { Footer } from "./Footer";
import { Header } from "./Header";

export function TuiShell({
  snapshot,
  statusLine,
  onCommand,
  onQuit,
}: {
  snapshot: Snapshot;
  statusLine: string;
  onCommand: (text: string) => void;
  onQuit: () => void;
}) {
  const { width } = useTerminalDimensions();
  const narrow = width < 120;
  const [palette, setPalette] = useState(false);
  const renderer = useRenderer();

  useKeyboard((key) => {
    if (key.name === "k" && key.ctrl) {
      setPalette((p) => !p);
      return;
    }
    if (key.name === "escape") {
      if (palette) {
        setPalette(false);
        return;
      }
      renderer.destroy();
      onQuit();
    }
  });

  const sb = getSandbox(snapshot.sandboxId);
  const Body = sb.body;
  const label = statusLabel(snapshot.status);
  const traceStyle = narrow
    ? { height: "50%" as const, flexShrink: 0, border: ["top"] as const, borderColor: COLORS.border }
    : { width: 52, flexShrink: 0, border: ["left"] as const, borderColor: COLORS.border };

  return (
    <box style={{ flexDirection: "column", width: "100%", height: "100%", backgroundColor: COLORS.bg }}>
      <Header sandboxLabel={sb.label} threshold={snapshot.threshold} status={label} />
      {snapshot.query ? (
        <box
          style={{
            flexDirection: "column",
            flexShrink: 0,
            paddingX: 1,
            backgroundColor: COLORS.panelAlt,
            border: ["bottom"],
            borderColor: COLORS.border,
          }}
        >
          <text fg={COLORS.dim}>TASK</text>
          <text fg={COLORS.text}>{snapshot.query}</text>
        </box>
      ) : null}
      <box style={{ flexDirection: narrow ? "column" : "row", flexGrow: 1 }}>
        <box style={{ flexDirection: "column", flexGrow: 1 }}>
          <Body
            completedTurns={snapshot.completedTurns}
            activeTurn={snapshot.activeTurn}
            allTurns={snapshot.completedTurns}
            finalText={snapshot.finalText}
            inbox={snapshot.inbox}
          />
        </box>
        <box style={{ flexDirection: "column", ...traceStyle }}>
          <TracePanel
            completedTurns={snapshot.completedTurns}
            activeTurn={snapshot.activeTurn}
            totals={snapshot.totals}
          />
        </box>
      </box>
      <Footer
        sandboxId={snapshot.sandboxId}
        disabled={snapshot.status !== "ready" || palette}
        statusLine={statusLine}
        onCommand={onCommand}
      />
      {palette ? (
        <CommandPalette
          onPick={(value) => {
            setPalette(false);
            onCommand(value);
          }}
        />
      ) : null}
    </box>
  );
}
