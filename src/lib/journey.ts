/**
 * The "follow one point" journey through the network, as a single progress value p in [0, 7]:
 *
 *   0 x  →  1 block 1  →  2 h1  →  3 block 2  →  4 h2  →  5 block 3  →  6 h3  →  7 ŷ
 *
 * Integer stops line up with the columns of the architecture diagram.
 */

export const JOURNEY_STOPS = 8;
export const JOURNEY_END = JOURNEY_STOPS - 1;

export type Stage =
  | { kind: "input"; index: 0 }
  | { kind: "block"; index: number }
  | { kind: "rep"; index: number }
  | { kind: "output"; index: 0 };

const clampP = (p: number): number => Math.min(JOURNEY_END, Math.max(0, p));

/** The stop the point has most recently reached. */
export function stageAt(p: number): Stage {
  const s = Math.floor(clampP(p) + 1e-9);
  if (s === 0) return { kind: "input", index: 0 };
  if (s === JOURNEY_END) return { kind: "output", index: 0 };
  return s % 2 === 1 ? { kind: "block", index: (s - 1) / 2 } : { kind: "rep", index: s / 2 };
}

/** Representation depth to plot (0 = x, 3 = h3): glides across each block, then holds at h3. */
export function depthAt(p: number): number {
  return Math.min(3, clampP(p) / 2);
}

/** Smoothstep within each segment, so motion slows into every stop and leaves it gently. */
export function easeJourney(p: number): number {
  const c = clampP(p);
  const base = Math.floor(c);
  if (base >= JOURNEY_END) return JOURNEY_END;
  const f = c - base;
  return base + f * f * (3 - 2 * f);
}
