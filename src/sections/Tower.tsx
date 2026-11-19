import { useState } from "react";
import { DepthTower3D } from "../components/DepthTower3D";
import { PlayScrubber, Segmented, Toggle } from "../components/Controls";
import { useNetwork } from "../context/NetworkContext";
import { usePlayback } from "../hooks/usePlayback";
import { formatPct } from "../lib/theme";
import type { RunKey } from "../lib/types";

export function Tower() {
  const { model } = useNetwork();
  const [run, setRun] = useState<RunKey>("relu");
  const [showPaths, setShowPaths] = useState(true);
  const count = model.frameCount(run);
  const playback = usePlayback(count, 12, count - 1);
  const fi = Math.min(count - 1, Math.round(playback.frame));
  const frame = model.data[run].frames[fi];

  return (
    <section className="section" id="tower">
      <div className="prose">
        <h2>
          <span className="num">06</span>The whole machine, in 3D
        </h2>
        <p>
          Here's everything at once. The four planes are <span className="l0">x</span>, <span className="l1">h₁</span>,{" "}
          <span className="l2">h₂</span> and <span className="l3">h₃</span>, and every point is threaded through all four,
          so you can see where each one goes. Between the planes, a ring of <strong>128 bars</strong> shows the hidden layer
          of the point you're following (red path). Dark bars are neurons ReLU switched off. The white line on the last
          plane is the final simple model.
        </p>
        <p className="hint">Drag to orbit · scroll to zoom · click a point to follow it · press play to replay training.</p>
      </div>
      <div className="tower-wrap">
        <DepthTower3D run={run} frame={playback.frame} showPaths={showPaths} />
        <div className="tower-controls">
          <PlayScrubber
            frame={playback.frame}
            count={count}
            playing={playback.playing}
            onToggle={playback.toggle}
            onScrub={playback.scrub}
            label={
              <>
                step <b>{frame.step}</b> · accuracy {formatPct(frame.accuracy)}
              </>
            }
          />
          <div className="toggles">
            <Segmented
              ariaLabel="Network"
              value={run}
              onChange={setRun}
              options={[
                { value: "relu", label: "with ReLU" },
                { value: "linear", label: "without ReLU" },
              ]}
            />
            <Toggle checked={showPaths} onChange={setShowPaths}>
              all paths
            </Toggle>
          </div>
        </div>
      </div>
    </section>
  );
}
