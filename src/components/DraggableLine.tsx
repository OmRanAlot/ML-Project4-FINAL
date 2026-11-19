/**
 * A straight line defined by two handles. Drag a handle to rotate it, drag the line
 * body to slide it. Uses pointer capture so a fast drag never "drops" the line.
 */
import { useRef } from "react";
import { clamp, clipLine, lineThrough } from "../lib/geometry";
import type { PlaneScales } from "./Plane2D";

type Pt = [number, number];
type DragMode = { kind: "p" } | { kind: "q" } | { kind: "body"; start: Pt; p0: Pt; q0: Pt };

interface Props {
  scales: PlaneScales;
  p: Pt;
  q: Pt;
  onChange: (p: Pt, q: Pt) => void;
  color?: string;
}

export function DraggableLine({ scales, p, q, onChange, color = "#fff" }: Props) {
  const drag = useRef<DragMode | null>(null);
  const { extent } = scales;
  const lim = (v: Pt): Pt => [
    clamp(v[0], extent.cx - extent.half * 0.96, extent.cx + extent.half * 0.96),
    clamp(v[1], extent.cy - extent.half * 0.96, extent.cy + extent.half * 0.96),
  ];

  const toData = (e: React.PointerEvent<SVGElement>): Pt => {
    const svg = (e.currentTarget as SVGElement).ownerSVGElement ?? (e.currentTarget as SVGSVGElement);
    const rect = svg.getBoundingClientRect();
    return scales.fromPx(e.clientX - rect.left, e.clientY - rect.top);
  };

  const start = (mode: "p" | "q" | "body") => (e: React.PointerEvent<SVGElement>) => {
    e.preventDefault();
    e.stopPropagation();
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    drag.current = mode === "body" ? { kind: "body", start: toData(e), p0: p, q0: q } : { kind: mode };
  };

  const move = (e: React.PointerEvent<SVGElement>) => {
    const d = drag.current;
    if (!d) return;
    const m = toData(e);
    if (d.kind === "p") {
      const np = lim(m);
      if (Math.hypot(np[0] - q[0], np[1] - q[1]) > 1e-3) onChange(np, q);
    } else if (d.kind === "q") {
      const nq = lim(m);
      if (Math.hypot(nq[0] - p[0], nq[1] - p[1]) > 1e-3) onChange(p, nq);
    } else {
      // Clamp the translation itself so both handles stay inside: the line slides, never rotates.
      const lo = (v: number, c: number) => c - extent.half * 0.96 - v;
      const hi = (v: number, c: number) => c + extent.half * 0.96 - v;
      const dx = clamp(
        m[0] - d.start[0],
        Math.max(lo(d.p0[0], extent.cx), lo(d.q0[0], extent.cx)),
        Math.min(hi(d.p0[0], extent.cx), hi(d.q0[0], extent.cx)),
      );
      const dy = clamp(
        m[1] - d.start[1],
        Math.max(lo(d.p0[1], extent.cy), lo(d.q0[1], extent.cy)),
        Math.min(hi(d.p0[1], extent.cy), hi(d.q0[1], extent.cy)),
      );
      onChange([d.p0[0] + dx, d.p0[1] + dy], [d.q0[0] + dx, d.q0[1] + dy]);
    }
  };

  const end = (e: React.PointerEvent<SVGElement>) => {
    drag.current = null;
    const el = e.currentTarget as Element;
    if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
  };

  const seg = clipLine(lineThrough(p, q), extent);
  const [px, py] = scales.toPx(p[0], p[1]);
  const [qx, qy] = scales.toPx(q[0], q[1]);
  const handlers = { onPointerMove: move, onPointerUp: end, onPointerCancel: end };

  return (
    <g className="dragline">
      {seg && (
        <>
          <line
            {...handlers}
            onPointerDown={start("body")}
            x1={scales.toPx(...seg[0])[0]}
            y1={scales.toPx(...seg[0])[1]}
            x2={scales.toPx(...seg[1])[0]}
            y2={scales.toPx(...seg[1])[1]}
            stroke="transparent"
            strokeWidth={22}
            style={{ pointerEvents: "stroke", cursor: "grab", touchAction: "none" }}
          />
          <line
            x1={scales.toPx(...seg[0])[0]}
            y1={scales.toPx(...seg[0])[1]}
            x2={scales.toPx(...seg[1])[0]}
            y2={scales.toPx(...seg[1])[1]}
            stroke={color}
            strokeWidth={2.5}
            style={{ pointerEvents: "none" }}
          />
        </>
      )}
      {(
        [
          ["p", px, py],
          ["q", qx, qy],
        ] as const
      ).map(([key, x, y]) => (
        <g key={key}>
          <circle cx={x} cy={y} r={16} fill="transparent" style={{ pointerEvents: "all", cursor: "grab", touchAction: "none" }} {...handlers} onPointerDown={start(key)} />
          <circle cx={x} cy={y} r={7} className="dragline__handle" style={{ pointerEvents: "none" }} />
        </g>
      ))}
    </g>
  );
}
