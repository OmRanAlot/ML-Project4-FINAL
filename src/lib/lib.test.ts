import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { blockForward, compileNet, forwardAll, HIDDEN } from "./mlp";
import { clipLine, extentOf, fitLogisticLine, lerpBuffers, lineThrough, scoreLine } from "./geometry";
import { createModel } from "./model";
import type { NetParams, NetworkData } from "./types";

function makeParams(scale: number): NetParams {
  const W1 = Array.from({ length: HIDDEN }, (_, j) => [Math.cos(j), Math.sin(j)]);
  const b1 = Array.from({ length: HIDDEN }, (_, j) => (j % 7) / 7 - 0.5);
  const W2 = [
    Array.from({ length: HIDDEN }, (_, j) => (scale * Math.cos(2 * j)) / HIDDEN),
    Array.from({ length: HIDDEN }, (_, j) => (scale * Math.sin(3 * j)) / HIDDEN),
  ];
  const block = { W1, b1, W2, b2: [0.1, -0.2] };
  return { blocks: [block, block, block], head: { w: [1, -1], b: 0 } };
}

describe("mlp forward pass", () => {
  test("ReLU block matches a direct matrix computation", () => {
    const params = makeParams(1);
    const net = compileNet(params, true);
    const [x, y] = [0.3, -0.7];
    const b = params.blocks[0];
    const expected = [0, 1].map(
      (r) => b.b2[r] + b.W2[r].reduce((acc, w, j) => acc + w * Math.max(0, b.W1[j][0] * x + b.W1[j][1] * y + b.b1[j]), 0),
    );
    const out = blockForward(net.blocks[0], true, x, y);
    expect(out[0]).toBeCloseTo(expected[0], 5);
    expect(out[1]).toBeCloseTo(expected[1], 5);
  });

  test("without ReLU a block is an affine map (midpoints map to midpoints)", () => {
    const net = compileNet(makeParams(2), false);
    const a = blockForward(net.blocks[0], false, -1, 2);
    const c = blockForward(net.blocks[0], false, 3, -4);
    const mid = blockForward(net.blocks[0], false, 1, -1);
    expect(mid[0]).toBeCloseTo((a[0] + c[0]) / 2, 4);
    expect(mid[1]).toBeCloseTo((a[1] + c[1]) / 2, 4);
  });

  test("forwardAll returns input plus three representations of equal size", () => {
    const pts = Float32Array.from([0, 0, 1, 1, -1, 0.5]);
    const layers = forwardAll(compileNet(makeParams(1), true), pts);
    expect(layers).toHaveLength(4);
    layers.forEach((l) => expect(l).toHaveLength(pts.length));
    expect(layers[0]).toBe(pts);
  });
});

describe("geometry", () => {
  const pts = Float32Array.from([-1, -1, -2, -0.5, 1, 1, 2, 0.5]);
  const labels = [0, 0, 1, 1];

  test("scoreLine picks the better side assignment", () => {
    const line = lineThrough([0, -3], [0, 3]); // x = 0
    const s = scoreLine(line, pts, labels);
    expect(s.accuracy).toBe(1);
    expect(s.wrong).toBe(0);
    const flipped = scoreLine(lineThrough([0, 3], [0, -3]), pts, labels);
    expect(flipped.accuracy).toBe(1);
  });

  test("scoreLine marks misclassified points", () => {
    const s = scoreLine(lineThrough([-3, 0], [3, 0]), pts, [0, 1, 1, 1]);
    expect(s.correct + s.wrong).toBe(4);
    expect(Array.from(s.wrongMask).reduce((a, b) => a + b, 0)).toBe(s.wrong);
  });

  test("fitLogisticLine separates a separable set", () => {
    const line = fitLogisticLine(pts, labels);
    expect(scoreLine(line, pts, labels).accuracy).toBe(1);
  });

  test("clipLine returns a segment inside the extent", () => {
    const seg = clipLine({ a: 1, b: -1, c: 0 }, { cx: 0, cy: 0, half: 1 });
    expect(seg).not.toBeNull();
    seg!.forEach(([x, y]) => {
      expect(Math.abs(x)).toBeLessThanOrEqual(1 + 1e-9);
      expect(Math.abs(y)).toBeLessThanOrEqual(1 + 1e-9);
    });
    expect(clipLine({ a: 0, b: 1, c: -5 }, { cx: 0, cy: 0, half: 1 })).toBeNull();
  });

  test("extentOf covers every point and lerpBuffers interpolates", () => {
    const ext = extentOf([pts], 0);
    expect(ext.cx - ext.half).toBeLessThanOrEqual(-2);
    expect(ext.cx + ext.half).toBeGreaterThanOrEqual(2);
    const mid = lerpBuffers(Float32Array.from([0, 0]), Float32Array.from([2, 4]), 0.5);
    expect(Array.from(mid)).toEqual([1, 2]);
  });
});

describe("trained network data", () => {
  const data = JSON.parse(readFileSync(resolve(__dirname, "../../public/data/network.json"), "utf8")) as NetworkData;
  const model = createModel(data);
  const last = model.frameCount("relu") - 1;

  test("browser forward pass reproduces the exported training accuracy", () => {
    const net = model.net("relu", last);
    const h3 = model.reps("relu", last)[3];
    let correct = 0;
    for (let i = 0; i < model.n; i++) {
      const z = net.head.w0 * h3[2 * i] + net.head.w1 * h3[2 * i + 1] + net.head.b;
      if ((z > 0 ? 1 : 0) === model.labels[i]) correct++;
    }
    expect(correct / model.n).toBeCloseTo(data.relu.frames[last].accuracy, 2);
  });

  test("ReLU network ends linearly separable; the no-ReLU twin does not", () => {
    expect(data.relu.frames[last].probe[3]).toBeGreaterThan(0.99);
    expect(data.linear.frames[last].probe[3]).toBeLessThan(0.9);
  });

  test("separability grows with depth in the trained network", () => {
    const probe = data.relu.frames[last].probe;
    for (let k = 1; k < probe.length; k++) expect(probe[k]).toBeGreaterThanOrEqual(probe[k - 1]);
  });

  test("blocks start near the identity at step 0", () => {
    const reps = model.reps("relu", 0);
    let maxDev = 0;
    for (let i = 0; i < reps[0].length; i++) maxDev = Math.max(maxDev, Math.abs(reps[3][i] - reps[0][i]));
    expect(maxDev).toBeLessThan(0.25);
  });

  test("positionsAt at integer coordinates equals the exact snapshot", () => {
    expect(model.positionsAt("relu", 5, 2)).toBe(model.reps("relu", 5)[2]);
  });
});
