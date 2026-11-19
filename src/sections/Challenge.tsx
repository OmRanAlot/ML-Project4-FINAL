import { useMemo, useState } from "react";
import { DraggableLine } from "../components/DraggableLine";
import { Plane2D } from "../components/Plane2D";
import { Scrolly } from "../components/Scrolly";
import { Stat } from "../components/Controls";
import { useNetwork } from "../context/NetworkContext";
import { clamp, easeInOut, lineThrough, scoreLine } from "../lib/geometry";
import { COLORS, formatPct } from "../lib/theme";

type Pt = [number, number];

/** Section 1: try to separate the moons with a single straight line. */
export function Challenge() {
  const { model } = useNetwork();
  const last = model.frameCount("relu") - 1;
  const input = model.reps("relu", last)[0];
  const extent = model.extentAt("relu", 0);
  const [p, setP] = useState<Pt>([-1.6, -1.2]);
  const [q, setQ] = useState<Pt>([1.4, 1.5]);
  const [revealed, setRevealed] = useState(false);
  const [moves, setMoves] = useState(0);

  const score = useMemo(() => scoreLine(lineThrough(p, q), input, model.labels), [p, q, input, model.labels]);
  const best = model.data.relu.frames[last].probe[0];
  const bestLine = model.probeLine("relu", last, 0);

  return (
    <section className="section" id="challenge">
      <div className="prose">
        <h2>
          <span className="num">01</span>One line, two moons
        </h2>
        <p>
          Here are 400 points from two interlocking half-moons: <span className="c0">blue</span> and{" "}
          <span className="c1">gold</span>. You may use exactly one tool to tell them apart: a{" "}
          <strong>straight line</strong>. Everything on one side is called blue, everything on the other gold.
        </p>
        <p className="hint">Drag the handles to rotate the line. Drag the line itself to slide it.</p>
      </div>
      <div className="challenge">
        <Plane2D
          positions={input}
          labels={model.labels}
          extent={extent}
          wrongMask={score.wrongMask}
          line={revealed ? { line: bestLine, dashed: true, color: COLORS.layers[2], opacity: 0.9 } : null}
          ariaLabel="Two moons dataset with a draggable line"
          overlay={(s) => (
            <DraggableLine
              scales={s}
              p={p}
              q={q}
              onChange={(np, nq) => {
                setP(np);
                setQ(nq);
                setMoves((m) => m + 1);
              }}
            />
          )}
        />
        <aside className="challenge__panel">
          <div className="stats">
            <Stat label="correct" value={score.correct} tone="good" />
            <Stat label="wrong (circled)" value={score.wrong} tone="bad" />
            <Stat label="accuracy" value={formatPct(score.accuracy)} />
          </div>
          <div className="meter" aria-hidden>
            <div className="meter__fill" style={{ width: `${clamp((score.accuracy - 0.5) / 0.5, 0, 1) * 100}%` }} />
            <div className="meter__mark" style={{ left: `${((best - 0.5) / 0.5) * 100}%` }} title="the fitted best line" />
          </div>
          {moves > 6 || revealed ? (
            <>
              <p>
                However you turn it, some points stay on the wrong side. The moons hook into each other, and a line
                cannot follow a curve.
              </p>
              <button className="btn" onClick={() => setRevealed((r) => !r)}>
                {revealed ? "Hide" : "Show"} the best-fit line
              </button>
              {revealed && (
                <p className="callout">
                  A logistic regression, the best straight line it can find, scores <b>{formatPct(best)}</b>, and no line does much better. That ceiling is a fact
                  about the <em>data's shape</em>, not about how cleverly you dragged.
                </p>
              )}
            </>
          ) : (
            <p className="muted">Keep trying… can you get past 90%?</p>
          )}
        </aside>
      </div>
    </section>
  );
}

/** Scroll teaser: instead of bending the line, bend the space. */
export function TheIdea() {
  const { model } = useNetwork();
  const last = model.frameCount("relu") - 1;
  const net = model.net("relu", last);
  const headLine = { a: net.head.w0, b: net.head.w1, c: net.head.b };
  const final = model.data.relu.frames[last];

  const steps = [
    <>
      <h3>So don't bend the line.</h3>
      <p>
        The line is stuck being straight. But nothing says the <em>points</em> have to stay where they are. What if we
        moved them somewhere a line <em>could</em> separate them?
      </p>
    </>,
    <>
      <h3>Bend the space instead.</h3>
      <p>
        That is what a neural network does. Each layer computes a new position for every point: a new{" "}
        <strong>representation</strong> of the same data. Keep scrolling and watch all 400 points travel through the
        trained network.
      </p>
    </>,
    <>
      <h3>Now a line is enough.</h3>
      <p>
        Same 400 points, same colors, but in the network's final representation the classes sit on opposite sides of a
        straight line, and that line gets <b>{formatPct(final.accuracy)}</b> right.
      </p>
      <p>
        The last piece of the network is just that line: a <strong>logistic regression</strong>. The simple model didn't
        get smarter. The <em>data got simpler</em>.
      </p>
    </>,
  ];

  return (
    <section className="section section--scrolly">
      <Scrolly
        steps={steps}
        graphic={(t) => {
          const depth = t < 1 ? 0 : t < 2.6 ? 3 * easeInOut(clamp((t - 1.15) / 1.3, 0, 1)) : 3;
          const showHead = depth > 2.95;
          return (
            <Plane2D
              positions={model.positionsAt("relu", last, depth)}
              labels={model.labels}
              extent={model.extentAt("relu", depth)}
              grid={model.gridAt("relu", last, depth)}
              line={
                showHead
                  ? { line: headLine, color: "#fff" }
                  : depth < 0.05
                    ? { line: model.probeLine("relu", last, 0), dashed: true, opacity: 0.6 }
                    : null
              }
              caption={
                <span>
                  {depth < 0.05 ? "the input" : depth > 2.95 ? "the final representation, h₃" : "moving through the network…"}
                </span>
              }
            />
          );
        }}
      />
    </section>
  );
}
