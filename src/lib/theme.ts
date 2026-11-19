/**
 * One color per idea, used everywhere (points, KaTeX, prose highlights, 3D scene).
 * Mirrors the CSS custom properties in styles/global.css.
 */
export const COLORS = {
  bg: "#0b0d12",
  panel: "#11141c",
  ink: "#ebe7dc",
  muted: "#8d92a0",
  faint: "#262b38",
  class0: "#58c4dd",
  class1: "#f2b33d",
  wrong: "#ff5a6a",
  grid: "rgba(150, 170, 215, 0.24)",
  line: "#ffffff",
  /** input x, h1, h2, h3 */
  layers: ["#d9dde6", "#c59af0", "#6fe0b8", "#ff7f9f"],
} as const;

export const LAYER_LABELS = ["x", "h₁", "h₂", "h₃"];
export const LAYER_NAMES = ["the input", "representation 1", "representation 2", "representation 3"];

export const classColor = (label: number): string => (label === 1 ? COLORS.class1 : COLORS.class0);

export function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export const formatPct = (v: number): string => `${(v * 100).toFixed(1)}%`;
