/**
 * "How straight-line-friendly is each representation?" over training.
 * One line per layer (best linear-probe accuracy), plus a marker at the current snapshot.
 */
import { useMemo } from "react";
import { line as d3line, scaleLinear } from "d3";
import { useElementSize } from "../hooks/useElementSize";
import { COLORS, LAYER_LABELS } from "../lib/theme";
import type { Frame } from "../lib/types";

interface Props {
  frames: Frame[];
  frame: number;
  focusLayer?: number;
  onScrub?: (f: number) => void;
}

const M = { top: 14, right: 40, bottom: 26, left: 40 };
const HEIGHT = 170;

export function MetricsChart({ frames, frame, focusLayer, onScrub }: Props) {
  const { ref, width } = useElementSize<HTMLDivElement>();
  const w = Math.max(0, width);
  const x = useMemo(() => scaleLinear().domain([0, frames.length - 1]).range([M.left, w - M.right]), [frames.length, w]);
  const y = useMemo(() => scaleLinear().domain([0.84, 1.0]).range([HEIGHT - M.bottom, M.top]).clamp(true), []);

  const paths = useMemo(
    () =>
      [0, 1, 2, 3].map((k) =>
        d3line<Frame>()
          .x((_, i) => x(i))
          .y((f) => y(f.probe[k]))(frames),
      ),
    [frames, x, y],
  );

  const fi = Math.round(frame);
  const current = frames[Math.min(frames.length - 1, Math.max(0, fi))];
  const handlePointer = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!onScrub || (e.type === "pointermove" && e.buttons !== 1)) return;
    const rect = e.currentTarget.getBoundingClientRect();
    onScrub(Math.max(0, Math.min(frames.length - 1, x.invert(e.clientX - rect.left))));
  };

  return (
    <div className="metrics" ref={ref}>
      {w > 0 && (
        <svg
          width={w}
          height={HEIGHT}
          onPointerDown={handlePointer}
          onPointerMove={handlePointer}
          style={{ cursor: onScrub ? "ew-resize" : "default", touchAction: "pan-y" }}
          role="img"
          aria-label="Best straight-line accuracy in each representation over training"
        >
          {[0.85, 0.9, 0.95, 1].map((t) => (
            <g key={t}>
              <line x1={M.left} x2={w - M.right} y1={y(t)} y2={y(t)} stroke={COLORS.faint} />
              <text x={M.left - 6} y={y(t)} dy="0.32em" textAnchor="end" className="metrics__tick">
                {Math.round(t * 100)}%
              </text>
            </g>
          ))}
          {paths.map((d, k) => (
            <path
              key={k}
              d={d ?? ""}
              fill="none"
              stroke={COLORS.layers[k]}
              strokeWidth={focusLayer === k ? 3 : 1.6}
              opacity={focusLayer == null || focusLayer === k ? 1 : 0.45}
            />
          ))}
          {[0, 1, 2, 3].map((k) => (
            <text
              key={k}
              x={w - M.right + 6}
              y={y(frames[frames.length - 1].probe[k]) + (k === 0 ? 0 : (k - 2) * 11)}
              dy="0.32em"
              fill={COLORS.layers[k]}
              className="metrics__tick"
            >
              {LAYER_LABELS[k]}
            </text>
          ))}
          <line x1={x(frame)} x2={x(frame)} y1={M.top - 4} y2={HEIGHT - M.bottom} stroke="#fff" strokeOpacity={0.6} />
          {[0, 1, 2, 3].map((k) => (
            <circle key={k} cx={x(frame)} cy={y(current.probe[k])} r={3.5} fill={COLORS.layers[k]} />
          ))}
          <text x={M.left} y={HEIGHT - 6} className="metrics__axis">
            training step 0
          </text>
          <text x={w - M.right} y={HEIGHT - 6} textAnchor="end" className="metrics__axis">
            step {frames[frames.length - 1].step}
          </text>
        </svg>
      )}
    </div>
  );
}
