import { useMemo, useState } from "react";
import { LineLegend } from "../components/LineLegend";
import { MetricsChart } from "../components/MetricsChart";
import { Plane2D } from "../components/Plane2D";
import { PlayScrubber, Segmented, Toggle } from "../components/Controls";
import { Scrolly } from "../components/Scrolly";
import { SmallMultiples } from "../components/SmallMultiples";
import { useNetwork } from "../context/NetworkContext";
import { usePlayback } from "../hooks/usePlayback";
import { clamp } from "../lib/geometry";
import type { NetworkModel } from "../lib/model";
import { COLORS, LAYER_LABELS, formatPct } from "../lib/theme";

const FIELD_RES = 64;

/** Frame indices whose training step is closest to each requested step. */
function framesNearSteps(model: NetworkModel, steps: number[]): number[] {
  const frames = model.data.relu.frames;
  const picks = steps.map((s) => {
    let best = 0;
    frames.forEach((f, i) => {
      if (Math.abs(f.step - s) < Math.abs(frames[best].step - s)) best = i;
    });
    return best;
  });
  return [...new Set(picks)];
}

function MiniLayers({ frame }: { frame: number }) {
  const { model, selected } = useNetwork();
  const fi = Math.round(frame);
  return (
    <div className="minis">
      {[0, 1, 2, 3].map((k) => (
        <Plane2D
          key={k}
          positions={model.positionsAt("relu", frame, k)}
          labels={model.labels}
          extent={model.extentAt("relu", k)}
          grid={model.gridAt("relu", frame, k)}
          gridOpacity={0.7}
          highlight={selected}
          pointRadius={2}
          caption={
            <span>
              <b style={{ color: COLORS.layers[k] }}>{LAYER_LABELS[k]}</b> {formatPct(model.data.relu.frames[fi].probe[k])}
            </span>
          }
        />
      ))}
    </div>
  );
}

function TrainingScrolly() {
  const { model, selected } = useNetwork();
  const frames = model.data.relu.frames;
  const lastIdx = frames.length - 1;

  const frameFromScroll = (t: number) => {
    if (t < 1) return 0;
    if (t < 2) return clamp((t - 1.1) / 0.8, 0, 1) * lastIdx * 0.35;
    if (t < 3) return lastIdx * (0.35 + 0.65 * clamp((t - 2.05) / 0.8, 0, 1));
    return clamp((t - 3.1) / 0.75, 0, 1) * lastIdx;
  };

  const steps = [
    <>
      <h3>Before training</h3>
      <p>
        Rewind to <b>step 0</b>. We set each block up to start out as (almost) <em>do nothing</em>: it copies its input.
        So right now <span style={{ color: COLORS.layers[3] }}>h₃</span> looks just like <Tex0 />, and the network is no
        better than the best line: <b>{formatPct(frames[0].probe[3])}</b>.
      </p>
    </>,
    <>
      <h3>Training nudges the transformation</h3>
      <p>
        Training shows the network the data, measures how wrong its predictions are, and nudges all the weights a
        little to be less wrong. Repeat a few hundred times. We won't open up <em>how</em> the nudges are computed.
        Watch what they do instead: each nudge moves <span style={{ color: COLORS.layers[3] }}>h₃</span> a little, and
        always in the direction of <em>easier to separate</em>.
      </p>
    </>,
    <>
      <h3>The moons slide apart</h3>
      <p>
        The tips unhook first, where the classes overlapped. Then a gap opens and widens. The chart below the plot shows
        the best-line accuracy for every representation as training goes. Each one climbs, and the deeper the layer, the
        faster it climbs.
      </p>
    </>,
    <>
      <h3>Both directions at once</h3>
      <p>
        Here are all four representations side by side, replaying training. Read across for <em>depth</em>, and watch over
        time for <em>training</em>: change is gradual in both directions, and training shapes every layer at once.
      </p>
    </>,
  ];

  return (
    <Scrolly
      steps={steps}
      graphic={(t, active) => {
        const frame = frameFromScroll(t);
        const fi = Math.round(frame);
        const ext = model.extentAt("relu", 3);
        return (
          <div className="trainfig">
            {active < 3 ? (
              <Plane2D
                positions={model.positionsAt("relu", frame, 3)}
                labels={model.labels}
                extent={ext}
                grid={model.gridAt("relu", frame, 3)}
                highlight={selected}
                field={{ values: model.field("relu", fi, 3, ext, FIELD_RES), res: FIELD_RES, extent: ext, contour: true, opacity: 0.8 }}
                caption={
                  <span>
                    <b style={{ color: COLORS.layers[3] }}>h₃</b> at training step <b>{model.stepOf("relu", frame)}</b> · best line{" "}
                    {formatPct(frames[fi].probe[3])}
                    <br />
                    <LineLegend />
                  </span>
                }
              />
            ) : (
              <MiniLayers frame={frame} />
            )}
            <MetricsChart frames={frames} frame={frame} focusLayer={active < 3 ? 3 : undefined} />
          </div>
        );
      }}
    />
  );
}

function Tex0() {
  return <span style={{ color: COLORS.layers[0] }}>the input x</span>;
}

function TrainingExplorer() {
  const { model, selected, setSelected } = useNetwork();
  const frames = model.data.relu.frames;
  const playback = usePlayback(frames.length, 12, frames.length - 1);
  const [layer, setLayer] = useState(3);
  const [showGrid, setShowGrid] = useState(true);
  const [showField, setShowField] = useState(true);
  const rows = useMemo(() => framesNearSteps(model, [0, 15, 35, 70, 150, frames[frames.length - 1].step]), [model, frames]);
  const fi = Math.round(playback.frame);
  const ext = model.extentAt("relu", layer);

  return (
    <div className="explorer">
      <div className="prose">
        <h3>Explore it yourself</h3>
        <p>
          Every cell below is the same 400 points. Columns go deeper into the network, rows go further into training,
          and the number is the best straight-line accuracy in that cell. Click a cell to open it in the big view, then
          press play.
        </p>
      </div>
      <div className="explorer__grid">
        <SmallMultiples
          run="relu"
          rows={rows}
          activeFrame={fi}
          activeLayer={layer}
          onPick={(f, k) => {
            playback.scrub(f);
            setLayer(k);
          }}
        />
        <div className="explorer__main">
          <Segmented
            ariaLabel="Representation"
            value={layer}
            onChange={setLayer}
            options={[0, 1, 2, 3].map((k) => ({ value: k, label: LAYER_LABELS[k], color: COLORS.layers[k] }))}
          />
          <Plane2D
            positions={model.positionsAt("relu", playback.frame, layer)}
            labels={model.labels}
            extent={ext}
            grid={showGrid ? model.gridAt("relu", playback.frame, layer) : undefined}
            field={showField ? { values: model.field("relu", fi, layer, ext, FIELD_RES), res: FIELD_RES, extent: ext, contour: true, opacity: 0.8 } : null}
            line={{ line: model.probeLine("relu", fi, layer), dashed: true, color: COLORS.layers[layer], opacity: 0.8 }}
            highlight={selected}
            onPointPick={setSelected}
            caption={
              <LineLegend
                probeColor={COLORS.layers[layer]}
                probeLabel={<b>{formatPct(frames[fi].probe[layer])}</b>}
                showBoundary={showField}
              />
            }
          />
          <PlayScrubber
            frame={playback.frame}
            count={frames.length}
            playing={playback.playing}
            onToggle={playback.toggle}
            onScrub={playback.scrub}
            label={
              <>
                step <b>{model.stepOf("relu", playback.frame)}</b> · line {formatPct(frames[fi].probe[layer])}
              </>
            }
          />
          <div className="toggles">
            <Toggle checked={showGrid} onChange={setShowGrid}>
              warped grid
            </Toggle>
            <Toggle checked={showField} onChange={setShowField}>
              network's decision
            </Toggle>
          </div>
          <MetricsChart frames={frames} frame={playback.frame} focusLayer={layer} onScrub={playback.scrub} />
        </div>
      </div>
    </div>
  );
}

export function ThroughTraining() {
  return (
    <section className="section section--scrolly" id="training">
      <div className="prose">
        <h2>
          <span className="num">04</span>Through training
        </h2>
        <p>
          So far we've only looked at the <em>finished</em> network. But the transformation wasn't designed by anyone; it
          was <strong>learned</strong>. Let's watch it form.
        </p>
      </div>
      <TrainingScrolly />
      <TrainingExplorer />
    </section>
  );
}
