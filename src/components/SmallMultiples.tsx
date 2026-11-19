/**
 * The two directions at once: rows = moments in training, columns = depth.
 * Each cell is the same 400 points, colored by class, in that representation.
 */
import { useEffect, useRef } from "react";
import { useNetwork } from "../context/NetworkContext";
import { COLORS, LAYER_LABELS, classColor } from "../lib/theme";
import type { RunKey } from "../lib/types";

interface CellProps {
  run: RunKey;
  frame: number;
  layer: number;
  active: boolean;
  onClick: () => void;
}

function Cell({ run, frame, layer, active, onClick }: CellProps) {
  const { model, selected } = useNetwork();
  const ref = useRef<HTMLCanvasElement>(null);
  const SIZE = 120;

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, SIZE, SIZE);
    const pts = model.reps(run, frame)[layer];
    const ext = model.extentAt(run, layer);
    const span = 2 * ext.half;
    const toPx = (x: number, y: number): [number, number] => [
      ((x - ext.cx + ext.half) / span) * SIZE,
      ((ext.cy + ext.half - y) / span) * SIZE,
    ];
    model.labels.forEach((label, i) => {
      const [px, py] = toPx(pts[2 * i], pts[2 * i + 1]);
      ctx.fillStyle = classColor(label);
      ctx.globalAlpha = 0.85;
      ctx.fillRect(px - 1.1, py - 1.1, 2.2, 2.2);
    });
    ctx.globalAlpha = 1;
    const [sx, sy] = toPx(pts[2 * selected], pts[2 * selected + 1]);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(sx, sy, 4, 0, Math.PI * 2);
    ctx.stroke();
  }, [model, run, frame, layer, selected]);

  const probe = model.data[run].frames[frame].probe[layer];
  return (
    <button className={`sm__cell ${active ? "is-active" : ""}`} onClick={onClick} aria-label={`Step ${model.stepOf(run, frame)}, ${LAYER_LABELS[layer]}`}>
      <canvas ref={ref} style={{ width: SIZE, height: SIZE }} />
      <span className="sm__probe" style={{ color: probe >= 0.999 ? COLORS.layers[2] : COLORS.muted }}>
        {(probe * 100).toFixed(1)}%
      </span>
    </button>
  );
}

interface Props {
  run: RunKey;
  rows: number[];
  activeFrame?: number;
  activeLayer?: number;
  onPick?: (frame: number, layer: number) => void;
}

export function SmallMultiples({ run, rows, activeFrame, activeLayer, onPick }: Props) {
  const { model } = useNetwork();
  return (
    <div className="sm" role="group" aria-label="Representations across depth and training">
      <div className="sm__corner">
        <span>depth →</span>
        <span>training ↓</span>
      </div>
      {[0, 1, 2, 3].map((k) => (
        <div key={k} className="sm__colhead" style={{ color: COLORS.layers[k] }}>
          {LAYER_LABELS[k]}
        </div>
      ))}
      {rows.map((f) => (
        <div className="sm__row" key={f}>
          <div className="sm__rowhead">
            step <b>{model.stepOf(run, f)}</b>
          </div>
          {[0, 1, 2, 3].map((k) => (
            <Cell
              key={k}
              run={run}
              frame={f}
              layer={k}
              active={activeFrame === f && activeLayer === k}
              onClick={() => onPick?.(f, k)}
            />
          ))}
        </div>
      ))}
    </div>
  );
}
