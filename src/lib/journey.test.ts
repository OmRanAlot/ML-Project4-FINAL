import { describe, expect, test } from "vitest";
import { JOURNEY_STOPS, depthAt, easeJourney, stageAt } from "./journey";

describe("journey timeline", () => {
  test("has a stop for x, each block, each representation and the output", () => {
    expect(JOURNEY_STOPS).toBe(8);
  });

  test("stageAt names the stop the point has most recently reached", () => {
    expect(stageAt(0)).toEqual({ kind: "input", index: 0 });
    expect(stageAt(0.6)).toEqual({ kind: "input", index: 0 });
    expect(stageAt(1)).toEqual({ kind: "block", index: 0 });
    expect(stageAt(2.2)).toEqual({ kind: "rep", index: 1 });
    expect(stageAt(3)).toEqual({ kind: "block", index: 1 });
    expect(stageAt(5.9)).toEqual({ kind: "block", index: 2 });
    expect(stageAt(6)).toEqual({ kind: "rep", index: 3 });
    expect(stageAt(7)).toEqual({ kind: "output", index: 0 });
  });

  test("stageAt clamps out-of-range progress", () => {
    expect(stageAt(-1)).toEqual({ kind: "input", index: 0 });
    expect(stageAt(99)).toEqual({ kind: "output", index: 0 });
  });

  test("depthAt glides through each block and stops at h3", () => {
    expect(depthAt(0)).toBe(0);
    expect(depthAt(1)).toBe(0.5);
    expect(depthAt(2)).toBe(1);
    expect(depthAt(6)).toBe(3);
    expect(depthAt(7)).toBe(3);
  });

  test("easeJourney dwells at stops but stays continuous and monotonic", () => {
    expect(easeJourney(0)).toBe(0);
    expect(easeJourney(3)).toBe(3);
    expect(easeJourney(3.5)).toBeCloseTo(3.5);
    expect(easeJourney(3.1)).toBeLessThan(3.1);
    expect(easeJourney(3.9)).toBeGreaterThan(3.9);
    expect(easeJourney(7)).toBe(7);
    let prev = -1;
    for (let p = 0; p <= 7; p += 0.05) {
      const e = easeJourney(p);
      expect(e).toBeGreaterThanOrEqual(prev);
      prev = e;
    }
  });
});
