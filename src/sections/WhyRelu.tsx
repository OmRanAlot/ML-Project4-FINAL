import { useState } from "react";
import { PlayScrubber } from "../components/Controls";
import { Plane2D } from "../components/Plane2D";
import { Tex } from "../components/Tex";
import { useNetwork } from "../context/NetworkContext";
import { usePlayback } from "../hooks/usePlayback";
import { COLORS, LAYER_LABELS, formatPct } from "../lib/theme";
import type { RunKey } from "../lib/types";

const FIELD_RES = 64;
/** Depth runs over x, h1, h2, h3 (0..3). */
const DEPTH_STOPS = 4;
/** Playback speed: 0.75 layers/s, so x → h3 takes about 4 seconds. */
const DEPTH_PER_SECOND = 0.75;
type Guess = "yes" | "no" | null;

function Side({ run, depth, title }: { run: RunKey; depth: number; title: string }) {
  const { model, selected } = useNetwork();
  const last = model.frameCount(run) - 1;
  const k = Math.round(depth);
  const inputExt = model.extentAt(run, 0);
  const frame = model.data[run].frames[last];
  return (
    <div className="compare__side">
      <h4>{title}</h4>
      <Plane2D
        positions={model.positionsAt(run, last, depth)}
        labels={model.labels}
        extent={model.extentAt(run, depth)}
        grid={model.gridAt(run, last, depth)}
        highlight={selected}
        line={Number.isInteger(depth) ? { line: model.probeLine(run, last, k), dashed: true, color: COLORS.layers[k] } : null}
        caption={
          <span>
            <b style={{ color: COLORS.layers[k] }}>{LAYER_LABELS[k]}</b> · best line {formatPct(frame.probe[k])}
          </span>
        }
      />
      <Plane2D
        className="plane--small"
        positions={model.reps(run, last)[0]}
        labels={model.labels}
        extent={inputExt}
        field={{ values: model.field(run, last, 0, inputExt, FIELD_RES), res: FIELD_RES, extent: inputExt, contour: true }}
        pointRadius={1.8}
        caption={
          <span>
            its decision boundary, drawn back in x · accuracy <b>{formatPct(frame.accuracy)}</b>
          </span>
        }
      />
    </div>
  );
}

export function WhyRelu() {
  const [guess, setGuess] = useState<Guess>(null);
  const playback = usePlayback(DEPTH_STOPS, DEPTH_PER_SECOND);
  const depth = playback.frame;

  return (
    <section className="section" id="relu">
      <div className="prose">
        <h2>
          <span className="num">05</span>Is it just the size?
        </h2>
        <p>
          A fair objection: maybe it isn't ReLU at all. The network has three blocks of 128 neurons, which is thousands of
          weights. Surely that much machinery can untangle two moons?
        </p>
        <div className="predict">
          <p>
            <b>Predict:</b> we train the exact same network, same size, same data, same training, but delete every ReLU.
            Can it separate the moons?
          </p>
          <div className="predict__opts">
            <button className={`btn ${guess === "yes" ? "is-on" : ""}`} onClick={() => setGuess("yes")}>
              Yes, it's big enough
            </button>
            <button className={`btn ${guess === "no" ? "is-on" : ""}`} onClick={() => setGuess("no")}>
              No, it'll get stuck
            </button>
          </div>
        </div>
      </div>

      {guess && (
        <div className="reveal">
          <div className="prose">
            <p className="callout">
              {guess === "no" ? "Right." : "Surprisingly, no."} Without ReLU the network is stuck at the same ~87% as a
              single line. Press play (or drag the slider) to push the points through the network and compare the two
              grids.
            </p>
            <PlayScrubber
              subject="depth"
              frame={depth}
              count={DEPTH_STOPS}
              playing={playback.playing}
              onToggle={playback.toggle}
              onScrub={playback.scrub}
              label={
                <>
                  depth <b style={{ color: COLORS.layers[Math.round(depth)] }}>{LAYER_LABELS[Math.round(depth)]}</b>
                </>
              }
            />
          </div>
          <div className="compare">
            <Side run="relu" depth={depth} title="With ReLU" />
            <Side run="linear" depth={depth} title="Without ReLU" />
          </div>
          <div className="prose">
            <p>
              Look at the grid on the right. The lines stay <strong>straight and parallel</strong>, so the whole plane is only
              stretched, sheared and rotated. That's because a block without ReLU collapses into a single matrix:
            </p>
            <Tex display math={String.raw`W_2\,(W_1 h + b_1) + b_2 \;=\; \underbrace{(W_2 W_1)}_{2\times 2}\,h \;+\; \underbrace{(W_2 b_1 + b_2)}_{\text{a shift}}`} />
            <p>
              Stack three of those and you still have one 2×2 matrix plus a shift. A stretched pair of moons is still a
              pair of hooked moons, so the boundary drawn back in <Tex math="x" /> stays a straight line. It's{" "}
              <em>nonlinearity</em>, not size, that lets a network bend space.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
