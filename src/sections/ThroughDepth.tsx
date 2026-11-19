import { Plane2D } from "../components/Plane2D";
import { Scrolly } from "../components/Scrolly";
import { Tex } from "../components/Tex";
import { useNetwork } from "../context/NetworkContext";
import { clamp, easeInOut } from "../lib/geometry";
import { COLORS, LAYER_LABELS, formatPct } from "../lib/theme";

const FIELD_RES = 80;

/** Scroll position -> depth, with a pause at each layer so readers can look. */
export function depthFromScroll(t: number): number {
  if (t < 1) return 0;
  if (t >= 4) return 3;
  const base = Math.floor(t) - 1;
  return base + easeInOut(clamp((t - Math.floor(t) - 0.1) / 0.65, 0, 1));
}

export function DepthRail({ depth }: { depth: number }) {
  return (
    <div className="rail" aria-hidden>
      {[0, 1, 2, 3].map((k) => (
        <span key={k} className="rail__stop" style={{ color: COLORS.layers[k], opacity: Math.abs(depth - k) < 0.5 ? 1 : 0.35 }}>
          {LAYER_LABELS[k]}
        </span>
      ))}
      <span className="rail__dot" style={{ left: `${(depth / 3) * 100}%` }} />
    </div>
  );
}

export function ThroughDepth() {
  const { model, selected } = useNetwork();
  const last = model.frameCount("relu") - 1;
  const probe = model.data.relu.frames[last].probe;

  const steps = [
    <>
      <h3>Layer by layer</h3>
      <p>
        Let's slow the trip down. Here is the input with a faint coordinate <strong>grid</strong> drawn over it. The grid
        isn't data: it shows what the network does to <em>the whole plane</em>, not just to our 400 points.
      </p>
      <p>
        The shading is the network's verdict for every location, and the white curve is its decision boundary. In{" "}
        <Tex math="x" /> that boundary is curved. The best a straight line can do: <b>{formatPct(probe[0])}</b>.
      </p>
    </>,
    <>
      <h3 style={{ color: COLORS.layers[1] }}>Block 1 → h₁</h3>
      <p>
        The first block makes a modest-looking change: it slides each moon's tip out from inside the other moon. The grid
        lines, once straight, begin to <em>bend</em>: that's ReLU at work.
      </p>
      <p>
        That small unhooking is the step that matters most. Best straight line in{" "}
        <Tex math={`\\textcolor{${COLORS.layers[1]}}{h_1}`} />: <b>{formatPct(probe[1])}</b>, but only just, with points
        still crowding the line.
      </p>
    </>,
    <>
      <h3 style={{ color: COLORS.layers[2] }}>Block 2 → h₂</h3>
      <p>
        The second block keeps going in the same direction: a clear gap opens between the classes. The boundary (white)
        is getting straighter, because the remaining work is getting easier.
      </p>
      <p>
        Best straight line in <Tex math={`\\textcolor{${COLORS.layers[2]}}{h_2}`} />: <b>{formatPct(probe[2])}</b>.
      </p>
    </>,
    <>
      <h3 style={{ color: COLORS.layers[3] }}>Block 3 → h₃</h3>
      <p>
        The last block widens the gap. Now the decision boundary <em>is</em> a straight line, because from{" "}
        <Tex math={`\\textcolor{${COLORS.layers[3]}}{h_3}`} /> onward the only thing left is the logistic regression.
      </p>
      <p>
        Best straight line in <Tex math={`\\textcolor{${COLORS.layers[3]}}{h_3}`} />: <b>{formatPct(probe[3])}</b>.
      </p>
    </>,
    <>
      <h3>No single step did it</h3>
      <p>
        Each block made a modest change, and the changes added up. The curved boundary in <Tex math="x" /> and the straight
        one in <Tex math="h_3" /> are <em>the same boundary</em>, seen before and after the warp.
      </p>
      <p className="muted">The white-ringed point is the one you followed above.</p>
    </>,
  ];

  return (
    <section className="section section--scrolly" id="depth">
      <div className="prose">
        <h2>
          <span className="num">03</span>Through the depth of the network
        </h2>
      </div>
      <Scrolly
        steps={steps}
        graphic={(t) => {
          const depth = depthFromScroll(t);
          const nearest = Math.round(depth);
          const settle = clamp(1 - Math.abs(depth - nearest) * 4, 0, 1);
          const ext = model.extentAt("relu", depth);
          const fieldExt = model.extentAt("relu", nearest);
          return (
            <div className="depthfig">
              <DepthRail depth={depth} />
              <Plane2D
                positions={model.positionsAt("relu", last, depth)}
                labels={model.labels}
                extent={ext}
                grid={model.gridAt("relu", last, depth)}
                highlight={selected}
                field={
                  settle > 0
                    ? {
                        values: model.field("relu", last, nearest, fieldExt, FIELD_RES),
                        res: FIELD_RES,
                        extent: fieldExt,
                        opacity: settle,
                        contour: true,
                      }
                    : null
                }
                line={settle > 0 ? { line: model.probeLine("relu", last, nearest), dashed: true, color: COLORS.layers[nearest], opacity: 0.85 * settle } : null}
                caption={
                  <span>
                    <b style={{ color: COLORS.layers[nearest] }}>{LAYER_LABELS[nearest]}</b> · dashed = best straight line ·{" "}
                    {formatPct(probe[nearest])}
                  </span>
                }
              />
            </div>
          );
        }}
      />
    </section>
  );
}
