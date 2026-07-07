/** Small presentation helpers + interval-driven tween/spinner hooks.

The tweens use a plain timer rather than the engine timeline so animation is
robust regardless of frame scheduling; useTimeline is used separately for the
header live pulse to exercise the engine's animation path.
*/
import { useEffect, useRef, useState } from "react";

export function clamp(x: number, lo = 0, hi = 1): number {
  return Math.max(lo, Math.min(hi, x));
}

export function senderName(from: string): string {
  const m = from.match(/^\s*"?([^"<]+?)"?\s*</);
  if (m) return m[1].trim();
  return from.split("@")[0];
}

export function fmtCost(usd: number): string {
  if (!usd) return "$0";
  const cents = usd * 100;
  if (cents < 1) return `${cents.toFixed(2)}¢`;
  if (usd < 1) return `${cents.toFixed(1)}¢`;
  return `$${usd.toFixed(2)}`;
}

export function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

export function truncate(s: string, n: number): string {
  s = (s || "").trim();
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

export function argsPreview(args: Record<string, unknown> | null, n = 38): string {
  if (!args || Object.keys(args).length === 0) return "";
  const parts = Object.entries(args).map(([k, v]) => `${k}=${JSON.stringify(v)}`);
  return truncate(parts.join(", "), n);
}

export function bar(frac: number, width: number, fillCh = "█", emptyCh = "·"): string {
  const f = Math.round(clamp(frac) * width);
  return fillCh.repeat(f) + emptyCh.repeat(Math.max(0, width - f));
}

/** Animate a number from its previous displayed value to `to` over `ms`. */
export function useTween(to: number, ms = 450): number {
  const [v, setV] = useState(0);
  const fromRef = useRef(0);
  useEffect(() => {
    const from = fromRef.current;
    const start = Date.now();
    let timer: ReturnType<typeof setTimeout>;
    const tick = () => {
      const t = clamp((Date.now() - start) / ms);
      const eased = 1 - (1 - t) * (1 - t);
      setV(from + (to - from) * eased);
      if (t < 1) {
        timer = setTimeout(tick, 16);
      } else {
        fromRef.current = to;
      }
    };
    tick();
    return () => clearTimeout(timer);
  }, [to, ms]);
  return v;
}

const SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export function useSpinner(active: boolean): string {
  const [i, setI] = useState(0);
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setI((n) => (n + 1) % SPIN.length), 80);
    return () => clearInterval(id);
  }, [active]);
  return SPIN[i];
}
