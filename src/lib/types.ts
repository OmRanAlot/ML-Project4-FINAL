/** Shapes of public/data/network.json (written by training/train.py). */

export interface BlockParams {
  W1: number[][]; // 128 x 2
  b1: number[]; // 128
  W2: number[][]; // 2 x 128
  b2: number[]; // 2
}

export interface NetParams {
  blocks: BlockParams[];
  head: { w: number[]; b: number };
}

export interface Frame {
  step: number;
  loss: number;
  accuracy: number;
  /** Best linear-classifier accuracy in [input, h1, h2, h3]. */
  probe: number[];
  /** The fitted probe line [a, b, c] (a*x + b*y + c = 0) for each representation. */
  probeLines?: number[][];
  params: NetParams;
}

export interface Run {
  useRelu: boolean;
  frames: Frame[];
}

export interface NetworkData {
  meta: {
    architecture: string;
    seed: number;
    noise: number;
    learningRate: number;
    smoothLambda: number;
    hidden: number;
  };
  data: { x: number[][]; y: number[] };
  relu: Run;
  linear: Run;
}

export type RunKey = "relu" | "linear";

/** Axis-aligned square view window in representation coordinates. */
export interface Extent {
  cx: number;
  cy: number;
  half: number;
}

/** A line a*x + b*y + c = 0. */
export interface Line {
  a: number;
  b: number;
  c: number;
}
