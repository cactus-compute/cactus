import { COLORS } from "../../theme";
import { clamp, useTween } from "../../util";

// Renders the probe's raw "cloud advantage" — the metric the engine actually decides on
// (hand off when advantage > threshold/lambda). Shown as a nominal value, NOT a percentage.
// (confidence = sigmoid(-advantage) is a squashed, different-scale view — don't compare it here.)
export function AdvantageMeter({
  advantage,
  threshold,
  segments = 16,
}: {
  advantage: number | null;
  threshold: number;
  segments?: number;
}) {
  const adv = advantage ?? 0;
  const frac = useTween(clamp(adv), 480); // bar is clamped to [0,1] for display; label shows the true value
  const escalated = advantage !== null && advantage > threshold;
  const color = escalated ? COLORS.threshold : COLORS.local;
  const filled = Math.round(frac * segments);
  const thrIdx = Math.min(segments - 1, Math.max(0, Math.round(clamp(threshold) * segments)));
  const cells = [];
  for (let i = 0; i < segments; i += 1) {
    if (i === thrIdx) cells.push(<span key={i} fg={COLORS.threshold}>│</span>);
    else if (i < filled) cells.push(<span key={i} fg={color}>█</span>);
    else cells.push(<span key={i} fg={COLORS.borderDim}>▁</span>);
  }
  return (
    <box style={{ flexDirection: "row" }}>
      <text>{cells}</text>
      <text fg={COLORS.dim}>{`  adv ${advantage === null ? "n/a" : advantage.toFixed(2)}`}</text>
      <text fg={COLORS.threshold}>{`  λ ${threshold.toFixed(2)}`}</text>
    </box>
  );
}
