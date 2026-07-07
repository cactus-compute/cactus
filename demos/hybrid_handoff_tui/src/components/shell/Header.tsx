import { COLORS } from "../../theme";

export function Header({
  sandboxLabel,
  threshold,
  status,
}: {
  sandboxLabel: string;
  threshold: number;
  status: string;
}) {
  const statusColor =
    status === "RUNNING" ? COLORS.threshold : status === "READY" ? COLORS.local : COLORS.dim;
  return (
    <box
      style={{
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        flexShrink: 0,
        paddingX: 1,
        backgroundColor: COLORS.panel,
        border: ["bottom"],
        borderColor: COLORS.border,
      }}
    >
      <box style={{ flexDirection: "row", alignItems: "center" }}>
        <text fg={COLORS.ink} bg={COLORS.local}>{" ❯_ "}</text>
        <text fg={COLORS.local}>{"  CACTUS HYBRID INFERENCE"}</text>
      </box>
      <box style={{ flexDirection: "row", alignItems: "center" }}>
        <text fg={COLORS.dim}>{`SANDBOX: ${sandboxLabel}`}</text>
        <text fg={COLORS.threshold}>{`     HANDOFF λ ${threshold.toFixed(2)}`}</text>
        <text fg={statusColor}>{`     ${status}`}</text>
      </box>
    </box>
  );
}
