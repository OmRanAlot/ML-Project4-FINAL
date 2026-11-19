/** Pure geometry helpers: lines, accuracy of a line, extents, interpolation. */
import type { Extent, Line } from "./types";

export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

export const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

/** Smooth ease for scroll-driven transitions. */
export const easeInOut = (t: number): number => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

export function lineThrough(p: [number, number], q: [number, number]): Line {
  const a = q[1] - p[1];
  const b = p[0] - q[0];
  return { a, b, c: -(a * p[0] + b * p[1]) };
}

export const sideOf = (line: Line, x: number, y: number): number => line.a * x + line.b * y + line.c;

export interface LineScore {
  correct: number;
  wrong: number;
  accuracy: number;
  /** Which side (sign) the line assigns to class 1, chosen to maximize accuracy. */
  positiveIsClass1: boolean;
  /** 1 where the point is misclassified. */
  wrongMask: Uint8Array;
}

/**
 * Score a straight line as a classifier. The line itself has no "blue side",
 * so we pick whichever side assignment gets more points right.
 */
export function scoreLine(line: Line, points: Float32Array, labels: number[]): LineScore {
  const n = labels.length;
  let agree = 0;
  const pos = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const s = sideOf(line, points[2 * i], points[2 * i + 1]) > 0 ? 1 : 0;
    pos[i] = s;
    if (s === labels[i]) agree++;
  }
  const positiveIsClass1 = agree >= n - agree;
  const correct = positiveIsClass1 ? agree : n - agree;
  const wrongMask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const predicted = positiveIsClass1 ? pos[i] : 1 - pos[i];
    wrongMask[i] = predicted === labels[i] ? 0 : 1;
  }
  return { correct, wrong: n - correct, accuracy: correct / n, positiveIsClass1, wrongMask };
}

/**
 * Fit a logistic-regression line (the "simple model") with plain gradient descent.
 * Points are standardized internally for stable steps.
 */
export function fitLogisticLine(points: Float32Array, labels: number[], iterations = 400): Line {
  const n = labels.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += points[2 * i];
    my += points[2 * i + 1];
  }
  mx /= n;
  my /= n;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += (points[2 * i] - mx) ** 2;
    sy += (points[2 * i + 1] - my) ** 2;
  }
  sx = Math.sqrt(sx / n) || 1;
  sy = Math.sqrt(sy / n) || 1;

  let w0 = 0;
  let w1 = 0;
  let b = 0;
  const lr = 0.5;
  for (let it = 0; it < iterations; it++) {
    let g0 = 0;
    let g1 = 0;
    let gb = 0;
    for (let i = 0; i < n; i++) {
      const u = (points[2 * i] - mx) / sx;
      const v = (points[2 * i + 1] - my) / sy;
      const p = 1 / (1 + Math.exp(-(w0 * u + w1 * v + b)));
      const e = p - labels[i];
      g0 += e * u;
      g1 += e * v;
      gb += e;
    }
    w0 -= (lr * g0) / n;
    w1 -= (lr * g1) / n;
    b -= (lr * gb) / n;
  }
  // Undo the standardization: w0*(x-mx)/sx + w1*(y-my)/sy + b = 0
  return { a: w0 / sx, b: w1 / sy, c: b - (w0 * mx) / sx - (w1 * my) / sy };
}

/** Clip an infinite line to a square extent; returns the segment endpoints or null. */
export function clipLine(line: Line, ext: Extent): [[number, number], [number, number]] | null {
  const x0 = ext.cx - ext.half;
  const x1 = ext.cx + ext.half;
  const y0 = ext.cy - ext.half;
  const y1 = ext.cy + ext.half;
  const hits: [number, number][] = [];
  if (Math.abs(line.b) > 1e-9) {
    for (const x of [x0, x1]) {
      const y = -(line.a * x + line.c) / line.b;
      if (y >= y0 - 1e-9 && y <= y1 + 1e-9) hits.push([x, y]);
    }
  }
  if (Math.abs(line.a) > 1e-9) {
    for (const y of [y0, y1]) {
      const x = -(line.b * y + line.c) / line.a;
      if (x >= x0 - 1e-9 && x <= x1 + 1e-9) hits.push([x, y]);
    }
  }
  // A line through a corner can produce duplicate hits: keep the two farthest apart.
  let best: [[number, number], [number, number]] | null = null;
  let bestD = 1e-12;
  for (let i = 0; i < hits.length; i++) {
    for (let j = i + 1; j < hits.length; j++) {
      const d = (hits[i][0] - hits[j][0]) ** 2 + (hits[i][1] - hits[j][1]) ** 2;
      if (d > bestD) {
        bestD = d;
        best = [hits[i], hits[j]];
      }
    }
  }
  return best;
}

/** Bounding square around a set of flat point buffers, with padding. */
export function extentOf(buffers: Float32Array[], pad = 0.12): Extent {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const buf of buffers) {
    for (let i = 0; i < buf.length; i += 2) {
      minX = Math.min(minX, buf[i]);
      maxX = Math.max(maxX, buf[i]);
      minY = Math.min(minY, buf[i + 1]);
      maxY = Math.max(maxY, buf[i + 1]);
    }
  }
  const half = Math.max(maxX - minX, maxY - minY) / 2;
  return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, half: half * (1 + pad) };
}

export const lerpExtent = (a: Extent, b: Extent, t: number): Extent => ({
  cx: lerp(a.cx, b.cx, t),
  cy: lerp(a.cy, b.cy, t),
  half: lerp(a.half, b.half, t),
});

/** Linear interpolation between two equally-sized flat buffers. */
export function lerpBuffers(a: Float32Array, b: Float32Array, t: number): Float32Array {
  if (t <= 0) return a;
  if (t >= 1) return b;
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  return out;
}
