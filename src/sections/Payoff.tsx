import { useMemo, useState } from "react";
import { DraggableLine } from "../components/DraggableLine";
import { Plane2D } from "../components/Plane2D";
import { Stat } from "../components/Controls";
import { useNetwork } from "../context/NetworkContext";
import { forwardFrom } from "../lib/mlp";
import { lineThrough, scoreLine, sideOf } from "../lib/geometry";
import { COLORS, formatPct } from "../lib/theme";
import type { Extent, Line } from "../lib/types";
import type { CompiledNet } from "../lib/mlp";

type Pt = [number, number];
const RES = 72;

/** Where each cell of a RES x RES lattice over the input lands in h3 (flat [x, y, ...]). */
function latticeToH3(net: CompiledNet, ext: Extent): Float32Array {
  const out = new Float32Array(RES * RES * 2);
  for (let j = 0; j < RES; j++) {
    const y = ext.cy + ext.half - ((j + 0.5) / RES) * 2 * ext.half;
    for (let i = 0; i < RES; i++) {
      const x = ext.cx - ext.half + ((i + 0.5) / RES) * 2 * ext.half;
      const [hx, hy] = forwardFrom(net, 0, x, y);
      out[2 * (j * RES + i)] = hx;
      out[2 * (j * RES + i) + 1] = hy;
    }
  }
  return out;
}

/** Which side of the user's h3 line each input location lands on (its "shadow" in x). */
function pullback(latticeH3: Float32Array, line: Line, positiveIsClass1: boolean): Float32Array {
  const out = new Float32Array(RES * RES);
  const norm = Math.hypot(line.a, line.b) || 1;
  for (let c = 0; c < out.length; c++) {
    const s = (sideOf(line, latticeH3[2 * c], latticeH3[2 * c + 1]) / norm) * (positiveIsClass1 ? 1 : -1);
    out[c] = 1 / (1 + Math.exp(-4 * s));
  }
  return out;
}

export function Payoff() {
  const { model, selected, setSelected } = useNetwork();
  const last = model.frameCount("relu") - 1;
  const net = model.net("relu", last);
  const reps = model.reps("relu", last);
  const hExt = useMemo(() => model.extentAt("relu", 3), [model]);
  const xExt = useMemo(() => model.extentAt("relu", 0), [model]);
  // Push the input lattice through the network once; dragging then only re-tests sides.
  const lattice = useMemo(() => latticeToH3(net, xExt), [net, xExt]);
  // Start the line deliberately bad: nearly flat, slicing through the lower cluster.
  const [p, setP] = useState<Pt>([hExt.cx - hExt.half * 0.8, hExt.cy - hExt.half * 0.3]);
  const [q, setQ] = useState<Pt>([hExt.cx + hExt.half * 0.8, hExt.cy - hExt.half * 0.15]);

  const line = useMemo(() => lineThrough(p, q), [p, q]);
  const score = useMemo(() => scoreLine(line, reps[3], model.labels), [line, reps, model.labels]);
  const shadow = useMemo(() => pullback(lattice, line, score.positiveIsClass1), [lattice, line, score.positiveIsClass1]);

  return (
    <section className="section" id="payoff">
      <div className="prose">
        <h2>
          <span className="num">07</span>Your line, one more time
        </h2>
        <p>
          Back to the opening challenge, but this time you draw your line in the network's final representation{" "}
          <span className="l3">h₃</span>. On the right, every location in the original input is shaded by which side of
          your line it ends up on after passing through the network.
        </p>
        <p className="hint">Drag your line on the left. Click any point to follow it in both views.</p>
      </div>
      <div className="payoff">
        <div>
          <Plane2D
            positions={reps[3]}
            labels={model.labels}
            extent={hExt}
            wrongMask={score.wrongMask}
            highlight={selected}
            onPointPick={setSelected}
            caption={<span className="l3">h₃: your straight line lives here</span>}
            overlay={(s) => (
              <DraggableLine
                scales={s}
                p={p}
                q={q}
                onChange={(np, nq) => {
                  setP(np);
                  setQ(nq);
                }}
              />
            )}
          />
          <div className="stats">
            <Stat label="correct" value={score.correct} tone="good" />
            <Stat label="wrong" value={score.wrong} tone="bad" />
            <Stat label="accuracy" value={formatPct(score.accuracy)} />
          </div>
        </div>
        <div>
          <Plane2D
            positions={reps[0]}
            labels={model.labels}
            extent={xExt}
            wrongMask={score.wrongMask}
            highlight={selected}
            onPointPick={setSelected}
            field={{ values: shadow, res: RES, extent: xExt, contour: true }}
            caption={<span>x: the same line, seen from the input</span>}
          />
          <p className="caption">
            {score.accuracy > 0.99
              ? "A straight cut in h₃ is a curved cut in x. The network's layers did the bending, so your line didn't have to."
              : "Keep adjusting. In h₃ a perfect line exists."}
          </p>
        </div>
      </div>
      <div className="prose">
        <p className="callout" style={{ borderColor: COLORS.layers[3] }}>
          This is the whole idea. A neural network classifier is a <strong>learned change of representation</strong>{" "}
          followed by a <strong>simple model</strong>. Training searches for a representation in which the simple model
          is enough.
        </p>
      </div>
    </section>
  );
}
