import { useEffect, useMemo, useRef, useState } from "react";
import { ArchitectureDiagram } from "../components/ArchitectureDiagram";
import { NeuronBars } from "../components/NeuronBars";
import { Plane2D } from "../components/Plane2D";
import { Segmented } from "../components/Controls";
import { Tex } from "../components/Tex";
import { useNetwork } from "../context/NetworkContext";
import { blockActivations, headLogit, sigmoid } from "../lib/mlp";
import { COLORS, LAYER_LABELS } from "../lib/theme";

/** Tween a number toward `target` over `ms`. */
function useTween(target: number, ms = 700): number {
  const [value, setValue] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    const start = performance.now();
    const v0 = from.current;
    let raf = 0;
    const tick = (now: number) => {
      const k = Math.min(1, (now - start) / ms);
      const e = 1 - (1 - k) ** 3;
      const v = v0 + (target - v0) * e;
      from.current = v;
      setValue(v);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return value;
}

export function InsideBlock() {
  const { model, selected, setSelected } = useNetwork();
  const last = model.frameCount("relu") - 1;
  const net = model.net("relu", last);
  const reps = model.reps("relu", last);
  const [block, setBlock] = useState(0);
  const [reluOn, setReluOn] = useState(false);
  const relu = useTween(reluOn ? 1 : 0);

  const acts = useMemo(
    () => [0, 1, 2].map((b) => blockActivations(net, b, reps[b][2 * selected], reps[b][2 * selected + 1])),
    [net, reps, selected],
  );
  const hidden = useMemo(() => acts.map((a) => a.post), [acts]);
  const repValues = useMemo(
    () => [0, 1, 2, 3].map((k) => [reps[k][2 * selected], reps[k][2 * selected + 1]] as [number, number]),
    [reps, selected],
  );
  const probability = sigmoid(headLogit(net, repValues[3][0], repValues[3][1]));
  const label = model.labels[selected];

  useEffect(() => setReluOn(false), [block, selected]);

  return (
    <section className="section" id="inside">
      <div className="prose">
        <h2>
          <span className="num">02</span>Follow one point
        </h2>
        <p>
          The network we'll study is small and deliberately shaped so we can <em>watch</em> it:
        </p>
        <Tex
          display
          math={String.raw`x \xrightarrow{\text{block 1}} \textcolor{${COLORS.layers[1]}}{h_1} \xrightarrow{\text{block 2}} \textcolor{${COLORS.layers[2]}}{h_2} \xrightarrow{\text{block 3}} \textcolor{${COLORS.layers[3]}}{h_3} \xrightarrow{\text{a line}} \hat y`}
        />
        <p>
          Every block takes a 2D point, expands it into <strong>128 numbers</strong>, applies <strong>ReLU</strong>, and
          compresses it back down to <strong>2 numbers</strong>. Because each in-between representation{" "}
          <Tex math={String.raw`\textcolor{${COLORS.layers[1]}}{h_1}, \textcolor{${COLORS.layers[2]}}{h_2}, \textcolor{${COLORS.layers[3]}}{h_3}`} />{" "}
          is 2D, we can plot it. Below is the trained network carrying one point (the ringed{" "}
          <span className={label ? "c1" : "c0"}>{label ? "gold" : "blue"}</span> one) all the way to a prediction. Click any
          point on the left to follow a different one.
        </p>
      </div>

      {/* Network on the left, the zoomed-in block (with ReLU) on the right, so both stay in view. */}
      <div className="workbench">
        <div className="workbench__net">
          <ArchitectureDiagram
            hidden={hidden}
            reps={repValues}
            probability={probability}
            focusBlock={block}
            onFocusBlock={setBlock}
          />
          <div className="workbench__netfoot">
            <div className="inside__pick">
              <Plane2D
                positions={reps[0]}
                labels={model.labels}
                extent={model.extentAt("relu", 0)}
                highlight={selected}
                onPointPick={setSelected}
                caption="input x: click a point"
              />
            </div>
            <p className="caption">
              Red neurons are active for this point; hollow ones were switched off by ReLU. Click a block in the network
              to zoom into it on the right. The final circle is the prediction ŷ = probability of gold:{" "}
              <b>{probability.toFixed(3)}</b> {(probability > 0.5 ? 1 : 0) === label ? "✓ correct" : "✗ wrong"}.
            </p>
          </div>
        </div>

        <div className="workbench__block">
          <h3>Zoom into one block</h3>
          <Segmented
            ariaLabel="Choose a block"
            value={block}
            onChange={setBlock}
            options={[0, 1, 2].map((b) => ({ value: b, label: `block ${b + 1}`, color: COLORS.layers[b + 1] }))}
          />
          <Tex
            display
            math={String.raw`\begin{gathered}\textcolor{${COLORS.layers[block + 1]}}{h_{${block + 1}}} = W_2\,\mathrm{ReLU}\!\left(W_1\,${block === 0 ? "x" : `\\textcolor{${COLORS.layers[block]}}{h_{${block}}}`} + b_1\right) + b_2 \\ W_1 \in \mathbb{R}^{128\times 2},\quad W_2 \in \mathbb{R}^{2\times 128}\end{gathered}`}
          />
          <div className="stage">
            <span className="stage__num">①</span> <b>Lift: 2 → 128.</b> <Tex math={String.raw`W_1 h + b_1`} /> measures
            the point 128 different ways. Each row of <Tex math="W_1" /> is a direction, so each number says how far the
            point lies along that direction, past some offset. For this point, the 128 measurements are:
          </div>
          <div className="stage stage--wide">
            <NeuronBars pre={acts[block].pre} relu={relu} color={COLORS.layers[block + 1]} />
            <button className="btn" onClick={() => setReluOn((r) => !r)}>
              {reluOn ? "Undo ReLU" : "② Apply ReLU: clip negatives to 0"}
            </button>
          </div>
          <div className="stage">
            <span className="stage__num">③</span> <b>Squash: 128 → 2.</b> <Tex math="W_2" /> blends the surviving
            numbers back into two coordinates. This point moves from{" "}
            <b style={{ color: COLORS.layers[block] }}>
              {LAYER_LABELS[block]} = ({repValues[block][0].toFixed(2)}, {repValues[block][1].toFixed(2)})
            </b>{" "}
            to{" "}
            <b style={{ color: COLORS.layers[block + 1] }}>
              {LAYER_LABELS[block + 1]} = ({repValues[block + 1][0].toFixed(2)}, {repValues[block + 1][1].toFixed(2)})
            </b>
            .
          </div>
        </div>
      </div>

      <div className="prose">
        <p className="callout">
          <b>Why ReLU is the whole trick.</b> Which neurons switch off depends on <em>where the point is</em>. Points in
          different regions get squashed back to 2D by different subsets of the 128 neurons, so different regions of the
          plane get moved by <em>different</em> linear maps. Stitch those pieces together and you get a map that can bend
          and fold space. Without ReLU, every point would be moved by the same matrix, and we'll see why that fails.
        </p>
        <p className="aside">
          A note on honesty: the 128-dimensional stage is real, but we can't draw 128 dimensions. From here on we only
          plot the 2D representations <Tex math="x, h_1, h_2, h_3" />. When points glide from one plot to the next,
          that glide is just our animation of before → after; the network jumps there in a single step, via 128D.
        </p>
      </div>
    </section>
  );
}
