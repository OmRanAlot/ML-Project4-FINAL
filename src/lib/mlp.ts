/**
 * A tiny, allocation-light forward pass for the explainer's MLP:
 *   h_{k+1} = W2 · act(W1 · h_k + b1) + b2      (k = 0, 1, 2)
 *   p(class 1) = sigmoid(w · h_3 + b)
 * act is ReLU for the main network and the identity for the "no ReLU" twin.
 */
import type { NetParams } from "./types";

export const HIDDEN = 128;
export const N_LAYERS = 4; // input, h1, h2, h3

export interface CompiledBlock {
  /** Row-major 128 x 2. */
  w1: Float32Array;
  b1: Float32Array;
  /** Row-major 2 x 128. */
  w2: Float32Array;
  b2: Float32Array;
}

export interface CompiledNet {
  blocks: CompiledBlock[];
  head: { w0: number; w1: number; b: number };
  useRelu: boolean;
}

export function compileNet(params: NetParams, useRelu: boolean): CompiledNet {
  const blocks = params.blocks.map((b) => ({
    w1: Float32Array.from(b.W1.flat()),
    b1: Float32Array.from(b.b1),
    w2: Float32Array.from(b.W2.flat()),
    b2: Float32Array.from(b.b2),
  }));
  return {
    blocks,
    head: { w0: params.head.w[0], w1: params.head.w[1], b: params.head.b },
    useRelu,
  };
}

/**
 * Push one 2D point through a block. If `pre`/`post` buffers are given, the 128
 * pre-activations and post-activations are written into them.
 */
export function blockForward(
  block: CompiledBlock,
  useRelu: boolean,
  x: number,
  y: number,
  pre?: Float32Array,
  post?: Float32Array,
): [number, number] {
  const { w1, b1, w2, b2 } = block;
  let o0 = b2[0];
  let o1 = b2[1];
  for (let j = 0; j < HIDDEN; j++) {
    const z = w1[2 * j] * x + w1[2 * j + 1] * y + b1[j];
    const a = useRelu && z < 0 ? 0 : z;
    if (pre) pre[j] = z;
    if (post) post[j] = a;
    o0 += w2[j] * a;
    o1 += w2[HIDDEN + j] * a;
  }
  return [o0, o1];
}

/** Run a point from representation `fromLayer` (0..3) to the last 2D space. */
export function forwardFrom(net: CompiledNet, fromLayer: number, x: number, y: number): [number, number] {
  let p: [number, number] = [x, y];
  for (let k = fromLayer; k < net.blocks.length; k++) {
    p = blockForward(net.blocks[k], net.useRelu, p[0], p[1]);
  }
  return p;
}

export function headLogit(net: CompiledNet, x: number, y: number): number {
  return net.head.w0 * x + net.head.w1 * y + net.head.b;
}

export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

/**
 * Representations of a flat point buffer [x0, y0, x1, y1, ...] at every layer.
 * Returns N_LAYERS buffers of the same length as `points`.
 */
export function forwardAll(net: CompiledNet, points: Float32Array): Float32Array[] {
  const layers: Float32Array[] = [points];
  let current = points;
  for (const block of net.blocks) {
    const next = new Float32Array(current.length);
    for (let i = 0; i < current.length; i += 2) {
      const [a, b] = blockForward(block, net.useRelu, current[i], current[i + 1]);
      next[i] = a;
      next[i + 1] = b;
    }
    layers.push(next);
    current = next;
  }
  return layers;
}

/** Hidden activations of a single point inside block `k`, given its input to that block. */
export function blockActivations(
  net: CompiledNet,
  k: number,
  x: number,
  y: number,
): { pre: Float32Array; post: Float32Array; out: [number, number] } {
  const pre = new Float32Array(HIDDEN);
  const post = new Float32Array(HIDDEN);
  const out = blockForward(net.blocks[k], net.useRelu, x, y, pre, post);
  return { pre, post, out };
}
