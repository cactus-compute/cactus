import { COLORS } from "../../theme";

export function RouteBadge({ location }: { location: "local" | "cloud" }) {
  const cloud = location === "cloud";
  return (
    <text fg={COLORS.ink} bg={cloud ? COLORS.cloud : COLORS.local}>
      {cloud ? " ☁ CLOUD " : " ⚙ ON-DEVICE "}
    </text>
  );
}
