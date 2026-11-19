# Make It Linear

An interactive, scroll-driven explainer for CS375H: **how and why a neural network transforms data so that a simple model (a straight line / logistic regression) can classify it.** Built around the assignment's architecture:

```
input(2) → [128 ReLU → 2 linear] ×3 → logistic regression
```

trained on two moons, with every 2D representation (x, h₁, h₂, h₃) visualized across **depth** and across **training**. Following the professor's feedback, backpropagation is deliberately not covered: the gradual change of the representations over training conveys the direction of the updates.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # unit tests (math + trained-data sanity checks)
npm run build      # production build in dist/
```

To regenerate the trained network (needs Python 3 with `torch`, `numpy`, `scikit-learn`):

```bash
npm run train      # writes public/data/network.json
```

## What's on the page

| # | Section | Idea | Interaction |
|---|---------|------|-------------|
| – | Hero | Points flow from x to h₃ and back through a warping grid | — |
| 01 | One line, two moons | A straight line tops out around 87% | Drag the line, score live, reveal the best line |
| – | The idea | Bend the space, not the line | Scroll-driven morph x → h₃ |
| 02 | Follow one point | Forward pass: 2 → 128 → ReLU → 2, three times | Click a point; network diagram; 128 bars with animated ReLU clipping |
| 03 | Through depth | Each block changes things modestly, and the boundary straightens | Scroll-scrubbed morph with warped grid, decision shading, best-line accuracy per layer |
| 04 | Through training | Representations drift toward separable | Scroll-scrubbed training, then a depth × training small-multiples grid, a play/scrub explorer and a probe-accuracy chart |
| 05 | Is it just the size? | Without ReLU, three 128-wide blocks collapse to one 2×2 matrix | Predict-then-reveal; side-by-side ReLU vs no-ReLU grids |
| 06 | The whole machine, in 3D | Everything at once | three.js depth tower: orbit, play training, click to follow a point, 128-bar hidden rings |
| 07 | Your line, one more time | A straight cut in h₃ is a curved cut in x | Drag a line in h₃ and watch its shadow in input space |
| 08 | Recap | Takeaways, misconceptions, how the network was trained | — |

## Making the transformation gradual

Your professor's main challenge: pick an architecture, dataset and training setup whose learned transformation is gradual in **both** the depth and the training directions. Three choices achieve it (all in `training/train.py`):

1. **Near-identity initialization.** Each block's 128→2 layer is solved by least squares so the block starts out as the identity, which means training begins from "do nothing" rather than a random scramble.
2. **Depth-smoothness penalty.** The loss adds `λ Σ‖h_k − h_{k−1}‖²` (λ = 0.08). A sum of squared step sizes is smallest when the work is split evenly across blocks.
3. **Small, steady optimization with displacement-based snapshots.** Full-batch Adam (lr 2e-3). A snapshot is saved whenever the points move a fixed distance, so replay speed is even.

Results: best-line (logistic probe) accuracy at the end of training is x 86.8% → h₁ 99.5% → h₂ 100% → h₃ 100%. The no-ReLU twin stays at about 87% in every layer.

## Architecture

```
training/train.py            PyTorch training → public/data/network.json (weights for 100 snapshots, ReLU + no-ReLU)
src/lib/mlp.ts               forward pass in the browser (from exported weights)
src/lib/model.ts             memoized reps, warped grids, decision fields, probe lines, (frame, depth) interpolation
src/lib/geometry.ts          line scoring/clipping, logistic fit, extents, lerp
src/components/Plane2D.tsx   canvas renderer (points, grid, decision field + contour) + SVG overlay
src/components/DepthTower3D  react-three-fiber 3D scene
src/sections/*               one file per section of the narrative
```

Because the browser re-runs the network from saved weights, any point, grid line or user-drawn line can be pushed through any snapshot.

## Inspiration

3Blue1Brown; Chris Olah, *Neural Networks, Manifolds, and Topology*; Yizhe Ang, *Matrix Explorable*; TensorFlow Playground; Karpathy's ConvNetJS 2D demo; R2D3; Distill's *Grand Tour*.
