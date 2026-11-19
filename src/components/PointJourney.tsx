/**
 * The "follow one point" panel: a plot that morphs x → h1 → h2 → h3 in step with the
 * wavefront in the architecture diagram, leaving the followed point's trail behind it,
 * plus a stepper and a running narration of what just happened to the point.
 */
import { useMemo, type ReactNode } from "react";
import { Plane2D, type PlaneScales } from "./Plane2D";
import { PlayScrubber } from "./Controls";
import { useNetwork } from "../context/NetworkContext";
import { JOURNEY_END, JOURNEY_STOPS, depthAt, stageAt } from "../lib/journey";
import { COLORS, LAYER_LABELS } from "../lib/theme";
import type { Line } from "../lib/types";

const TRAIL_STEP = 0.05;
const STOP_LABELS = ["x", "block 1", "h₁", "block 2", "h₂", "block 3", "h₃", "ŷ"];
const ACTIVE = "#ff6b6b";

interface Playback {
  playing: boolean;
  toggle: () => void;
  scrub: (f: number) => void;
}

interface Props {
  /** Raw (linear) journey progress, for the scrubber. */
  progress: number;
  /** Eased progress, for everything drawn. */
  shown: number;
  playback: Playback;
  /** Post-ReLU activations (128) for each block, for the followed point. */
  hidden: Float32Array[];
  /** The followed point's 2D value at x, h1, h2, h3. */
  repValues: [number, number][];
  probability: number;
  /** The network's output line in h3 coordinates. */
  headLine: Line;
}

const fmt = ([a, b]: [number, number]) => `(${a.toFixed(2)}, ${b.toFixed(2)})`;

/** Index of the stop most recently reached. */
function stopIndex(p: number): number {
  return Math.min(JOURNEY_END, Math.max(0, Math.floor(p + 1e-9)));
}

function stopColor(s: number): string {
  if (s === JOURNEY_END) return COLORS.ink;
  return s % 2 === 0 ? COLORS.layers[s / 2] : ACTIVE;
}

interface NarrationProps {
  shown: number;
  hidden: Float32Array[];
  repValues: [number, number][];
  probability: number;
  label: number;
}

function Narration({ shown, hidden, repValues, probability, label }: NarrationProps) {
  const stage = stageAt(shown);
  let text: ReactNode;
  if (stage.kind === "input") {
    text = (
      <>
        The point starts at <b style={{ color: COLORS.layers[0] }}>x = {fmt(repValues[0])}</b>.
      </>
    );
  } else if (stage.kind === "block") {
    const on = hidden[stage.index].reduce((n, v) => n + (v > 0 ? 1 : 0), 0);
    text = (
      <>
        <b>Block {stage.index + 1}</b> lifts it to 128 numbers. ReLU switches off <b>{128 - on}</b> of them; the{" "}
        <b style={{ color: ACTIVE }}>{on}</b> that stay on decide how this point gets moved.
      </>
    );
  } else if (stage.kind === "rep") {
    text = (
      <>
        …and squashes it back to 2D:{" "}
        <b style={{ color: COLORS.layers[stage.index] }}>
          {LAYER_LABELS[stage.index]} = {fmt(repValues[stage.index])}
        </b>
        .{stage.index === 3 && " Notice how far apart the two moons are now."}
      </>
    );
  } else {
    const correct = (probability > 0.5 ? 1 : 0) === label;
    text = (
      <>
        A single straight line reads off h₃: probability of gold <b>{probability.toFixed(3)}</b>.{" "}
        <b style={{ color: correct ? "#7bd88f" : ACTIVE }}>{correct ? "✓ correct" : "✗ wrong"}</b>
      </>
    );
  }
  return (
    <p className="journey__say" aria-live="polite" key={`${stage.kind}-${stage.index}`}>
      {text}
    </p>
  );
}

function Stops({ shown, onPick }: { shown: number; onPick: (s: number) => void }) {
  const now = stopIndex(shown);
  return (
    <ol className="journey__stops" aria-label="Journey stops">
      {STOP_LABELS.map((name, s) => (
        <li key={s}>
          <button
            type="button"
            className={`journey__stop ${shown >= s - 1e-3 ? "is-reached" : ""} ${now === s ? "is-now" : ""}`}
            style={{ ["--stop" as string]: stopColor(s) }}
            onClick={() => onPick(s)}
          >
            {name}
          </button>
        </li>
      ))}
    </ol>
  );
}

export function PointJourney({ progress, shown, playback, hidden, repValues, probability, headLine }: Props) {
  const { model, selected, setSelected } = useNetwork();
  const last = model.frameCount("relu") - 1;
  const depth = depthAt(shown);
  const atEnd = shown >= JOURNEY_END - 1e-3;

  // The followed point's path through depth, sampled once per point.
  const trail = useMemo(() => {
    const pts: [number, number][] = [];
    for (let d = 0; d <= 3 + 1e-9; d += TRAIL_STEP) {
      const p = model.positionsAt("relu", last, d);
      pts.push([p[2 * selected], p[2 * selected + 1]]);
    }
    return pts;
  }, [model, last, selected]);

  const positions = model.positionsAt("relu", last, depth);
  const here: [number, number] = [positions[2 * selected], positions[2 * selected + 1]];
  const layerNow = Math.round(depth);
  const onLayer = Math.abs(depth - layerNow) < 0.02;

  const overlay = (s: PlaneScales) => {
    const upto = Math.floor(depth / TRAIL_STEP + 1e-6) + 1;
    const pts = [...trail.slice(0, upto), here].map(([x, y]) => s.toPx(x, y));
    const [hx, hy] = pts[pts.length - 1];
    const flip = hx > s.width * 0.6;
    return (
      <g pointerEvents="none">
        <polyline points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke="#fff" strokeOpacity={0.6} strokeWidth={1.5} strokeDasharray="2 3" />
        <circle cx={hx} cy={hy} r={13} fill="none" stroke="#fff" strokeOpacity={0.5} className="journey__ping" />
        <text
          x={hx + (flip ? -16 : 16)}
          y={hy - 10}
          textAnchor={flip ? "end" : "start"}
          className="journey__tag"
          fill={onLayer ? COLORS.layers[layerNow] : ACTIVE}
        >
          {onLayer ? LAYER_LABELS[layerNow] : "via 128D…"}
        </text>
      </g>
    );
  };

  return (
    <div className="journey">
      <div className="journey__plot">
        <Plane2D
          positions={positions}
          labels={model.labels}
          extent={model.extentAt("relu", depth)}
          grid={model.gridAt("relu", last, depth)}
          gridOpacity={0.45}
          line={atEnd ? { line: headLine, dashed: true, color: COLORS.layers[3] } : null}
          highlight={selected}
          onPointPick={setSelected}
          overlay={overlay}
          caption={atEnd ? "h₃ with the network's final line · click a point" : "click any point to follow it"}
        />
      </div>
      <div className="journey__side">
        <Stops shown={shown} onPick={playback.scrub} />
        <PlayScrubber
          subject="journey"
          frame={progress}
          count={JOURNEY_STOPS}
          playing={playback.playing}
          onToggle={playback.toggle}
          onScrub={playback.scrub}
          label={
            <>
              now at <b>{STOP_LABELS[stopIndex(shown)]}</b>
            </>
          }
        />
        <Narration shown={shown} hidden={hidden} repValues={repValues} probability={probability} label={model.labels[selected]} />
      </div>
    </div>
  );
}
