/**
 * NetworkModel wraps the exported training snapshots and answers every question the
 * visuals ask: where is each point at (frame, depth)? how is the coordinate grid warped?
 * what does the classifier predict across a whole representation space?
 *
 * Results are memoized per snapshot because the visuals redraw every animation frame.
 */
import { compileNet, forwardAll, forwardFrom, headLogit, sigmoid, N_LAYERS, type CompiledNet } from "./mlp";
import { extentOf, fitLogisticLine, lerpBuffers, lerpExtent, clamp } from "./geometry";
import type { Extent, Line, NetworkData, RunKey } from "./types";

/** Input-space grid: lines at x = const and y = const, each sampled densely. */
const GRID_VALUES = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5];
const GRID_SPAN = 2.75;
const GRID_SAMPLES = 64;

function buildInputGrid(): Float32Array[] {
  const lines: Float32Array[] = [];
  for (const v of GRID_VALUES) {
    const vertical = new Float32Array(GRID_SAMPLES * 2);
    const horizontal = new Float32Array(GRID_SAMPLES * 2);
    for (let i = 0; i < GRID_SAMPLES; i++) {
      const s = -GRID_SPAN + (2 * GRID_SPAN * i) / (GRID_SAMPLES - 1);
      vertical[2 * i] = v;
      vertical[2 * i + 1] = s;
      horizontal[2 * i] = s;
      horizontal[2 * i + 1] = v;
    }
    lines.push(vertical, horizontal);
  }
  return lines;
}

export interface NetworkModel {
  readonly data: NetworkData;
  readonly labels: number[];
  readonly n: number;
  frameCount(run: RunKey): number;
  stepOf(run: RunKey, frame: number): number;
  net(run: RunKey, frame: number): CompiledNet;
  /** Data-point positions at each layer for an exact snapshot. */
  reps(run: RunKey, frame: number): Float32Array[];
  /** Point positions at a fractional snapshot and fractional depth (0 = input, 3 = h3). */
  positionsAt(run: RunKey, frame: number, depth: number): Float32Array;
  /** Warped grid polylines at a fractional snapshot and depth. */
  gridAt(run: RunKey, frame: number, depth: number): Float32Array[];
  /** View window for a (fractional) depth; fixed across training so motion is honest. */
  extentAt(run: RunKey, depth: number): Extent;
  /** p(class 1) sampled on a res x res lattice covering `ext` in layer-`layer` coordinates. */
  field(run: RunKey, frame: number, layer: number, ext: Extent, res: number): Float32Array;
  /** Best straight line (logistic regression) in a layer at a snapshot. */
  probeLine(run: RunKey, frame: number, layer: number): Line;
  /** Run arbitrary input points through a snapshot. */
  forwardPoints(run: RunKey, frame: number, pts: Float32Array): Float32Array[];
}

export function createModel(data: NetworkData): NetworkModel {
  const labels = data.data.y;
  const n = labels.length;
  const inputPoints = Float32Array.from(data.data.x.flat());
  const inputGrid = buildInputGrid();

  const netCache = new Map<string, CompiledNet>();
  const repCache = new Map<string, Float32Array[]>();
  const gridCache = new Map<string, Float32Array[][]>();
  const fieldCache = new Map<string, Float32Array>();
  const probeCache = new Map<string, Line>();
  const extentCache = new Map<RunKey, Extent[]>();

  const frameCount = (run: RunKey) => data[run].frames.length;
  const clampFrame = (run: RunKey, f: number) => clamp(Math.round(f), 0, frameCount(run) - 1);

  function net(run: RunKey, frame: number): CompiledNet {
    const f = clampFrame(run, frame);
    const key = `${run}|${f}`;
    let cached = netCache.get(key);
    if (!cached) {
      cached = compileNet(data[run].frames[f].params, data[run].useRelu);
      netCache.set(key, cached);
    }
    return cached;
  }

  function reps(run: RunKey, frame: number): Float32Array[] {
    const f = clampFrame(run, frame);
    const key = `${run}|${f}`;
    let cached = repCache.get(key);
    if (!cached) {
      cached = forwardAll(net(run, f), inputPoints);
      repCache.set(key, cached);
    }
    return cached;
  }

  function grids(run: RunKey, frame: number): Float32Array[][] {
    const f = clampFrame(run, frame);
    const key = `${run}|${f}`;
    let cached = gridCache.get(key);
    if (!cached) {
      const perLine = inputGrid.map((line) => forwardAll(net(run, f), line));
      // Re-index as [layer][line]
      cached = Array.from({ length: N_LAYERS }, (_, k) => perLine.map((layers) => layers[k]));
      gridCache.set(key, cached);
    }
    return cached;
  }

  function layerExtents(run: RunKey): Extent[] {
    let cached = extentCache.get(run);
    if (!cached) {
      const all = Array.from({ length: frameCount(run) }, (_, f) => reps(run, f));
      cached = Array.from({ length: N_LAYERS }, (_, k) => extentOf(all.map((r) => r[k])));
      extentCache.set(run, cached);
    }
    return cached;
  }

  /** Bilinear blend across (frame, depth) of any per-snapshot, per-layer buffer. */
  function blend<T>(
    run: RunKey,
    frame: number,
    depth: number,
    pick: (f: number, k: number) => T,
    mix: (a: T, b: T, t: number) => T,
  ): T {
    const fMax = frameCount(run) - 1;
    const fc = clamp(frame, 0, fMax);
    const f0 = Math.floor(fc);
    const f1 = Math.min(f0 + 1, fMax);
    const ft = fc - f0;
    const dc = clamp(depth, 0, N_LAYERS - 1);
    const k0 = Math.floor(dc);
    const k1 = Math.min(k0 + 1, N_LAYERS - 1);
    const kt = dc - k0;
    const atK0 = mix(pick(f0, k0), pick(f1, k0), ft);
    if (kt === 0) return atK0;
    const atK1 = mix(pick(f0, k1), pick(f1, k1), ft);
    return mix(atK0, atK1, kt);
  }

  return {
    data,
    labels,
    n,
    frameCount,
    stepOf: (run, frame) => data[run].frames[clampFrame(run, frame)].step,
    net,
    reps,
    positionsAt: (run, frame, depth) => blend(run, frame, depth, (f, k) => reps(run, f)[k], lerpBuffers),
    gridAt: (run, frame, depth) =>
      blend(
        run,
        frame,
        depth,
        (f, k) => grids(run, f)[k],
        (a, b, t) => a.map((line, i) => lerpBuffers(line, b[i], t)),
      ),
    extentAt: (run, depth) => {
      const ext = layerExtents(run);
      const dc = clamp(depth, 0, N_LAYERS - 1);
      const k0 = Math.floor(dc);
      const k1 = Math.min(k0 + 1, N_LAYERS - 1);
      return lerpExtent(ext[k0], ext[k1], dc - k0);
    },
    field: (run, frame, layer, ext, res) => {
      const f = clampFrame(run, frame);
      const key = `${run}|${f}|${layer}|${res}|${ext.cx.toFixed(3)}|${ext.cy.toFixed(3)}|${ext.half.toFixed(3)}`;
      let cached = fieldCache.get(key);
      if (!cached) {
        const compiled = net(run, f);
        cached = new Float32Array(res * res);
        for (let j = 0; j < res; j++) {
          const y = ext.cy + ext.half - ((j + 0.5) / res) * 2 * ext.half;
          for (let i = 0; i < res; i++) {
            const x = ext.cx - ext.half + ((i + 0.5) / res) * 2 * ext.half;
            const [hx, hy] = forwardFrom(compiled, layer, x, y);
            cached[j * res + i] = sigmoid(headLogit(compiled, hx, hy));
          }
        }
        if (fieldCache.size > 400) fieldCache.clear();
        fieldCache.set(key, cached);
      }
      return cached;
    },
    probeLine: (run, frame, layer) => {
      const f = clampFrame(run, frame);
      const key = `${run}|${f}|${layer}`;
      let cached = probeCache.get(key);
      if (!cached) {
        // Prefer the exact sklearn probe exported by training, so the drawn line and its
        // reported accuracy always agree; fall back to an in-browser fit for older files.
        const exported = data[run].frames[f].probeLines?.[layer];
        cached = exported
          ? { a: exported[0], b: exported[1], c: exported[2] }
          : fitLogisticLine(reps(run, f)[layer], labels, 3000);
        probeCache.set(key, cached);
      }
      return cached;
    },
    forwardPoints: (run, frame, pts) => forwardAll(net(run, frame), pts),
  };
}
