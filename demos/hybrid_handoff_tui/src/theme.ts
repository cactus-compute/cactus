/** Semantic palette — locked meanings (do not repurpose):
 *   green     = primary brand + local / on-device
 *   cyan      = cloud ONLY
 *   amber     = threshold + below-threshold escalation ONLY
 *   dim/faint = secondary
 */

export const COLORS = {
  bg: "#0a0e12",
  panel: "#0e141a",
  panelAlt: "#0b1116",
  border: "#1d2a33",
  borderDim: "#152028",
  text: "#cdd9e0",
  dim: "#7c8b96",
  faint: "#4f5e68",
  local: "#35d07f",
  cloud: "#22d3ee",
  threshold: "#f5b21a",
  ink: "#05080b",
} as const;

export const BRAND = COLORS.local;

export interface Mode {
  label: string;
  threshold: number;
  color: string;
}

// HYBRID's handoff lambda (advantage threshold) can be set at launch via CACTUS_DEMO_LAMBDA,
// e.g. `CACTUS_DEMO_LAMBDA=0.2 bun start`. Runtime: `/lambda <value>` in the command bar.
const _envLambda = process.env.CACTUS_DEMO_LAMBDA;
export const HYBRID_LAMBDA =
  _envLambda !== undefined && _envLambda !== "" && !Number.isNaN(Number(_envLambda))
    ? Number(_envLambda)
    : 0.05;

export const MODES: Mode[] = [
  { label: "LOCAL", threshold: 0.0, color: COLORS.local },
  { label: "HYBRID", threshold: HYBRID_LAMBDA, color: COLORS.threshold },
  { label: "CLOUD", threshold: 1.0, color: COLORS.cloud },
];

export function routeColor(location: "local" | "cloud"): string {
  return location === "cloud" ? COLORS.cloud : COLORS.local;
}
