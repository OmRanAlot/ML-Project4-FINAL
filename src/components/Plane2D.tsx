/**
 * A square 2D representation space. Canvas layer: decision shading, warped grid,
 * points. SVG overlay layer: anything interactive (draggable lines, labels).
 */
import { useEffect, useMemo, useRef, type ReactNode } from "react";
import { contours } from "d3";
import { useElementSize } from "../hooks/useElementSize";
import { clipLine } from "../lib/geometry";
import { COLORS, classColor, hexToRgb } from "../lib/theme";
import type { Extent, Line } from "../lib/types";

export interface PlaneScales {
  width: number;
  height: number;
  extent: Extent;
  toPx: (x: number, y: number) => [number, number];
  fromPx: (px: number, py: number) => [number, number];
}

export interface FieldLayer {
  /** Values in [0, 1] (probability of class 1) on a res x res lattice, row 0 = top. */
  values: Float32Array;
  res: number;
  extent: Extent;
  opacity?: number;
  /** Draw the p = 0.5 contour as a crisp boundary. */
  contour?: boolean;
}

export interface Plane2DProps {
  positions: Float32Array;
  labels: number[];
  extent: Extent;
  grid?: Float32Array[];
  gridOpacity?: number;
  field?: FieldLayer | null;
  line?: { line: Line; color?: string; opacity?: number; dashed?: boolean } | null;
  wrongMask?: Uint8Array | null;
  highlight?: number | null;
  pointRadius?: number;
  pointOpacity?: number;
  onPointPick?: (index: number) => void;
  overlay?: (scales: PlaneScales) => ReactNode;
  caption?: ReactNode;
  className?: string;
  ariaLabel?: string;
}

const PICK_RADIUS_PX = 14;

function makeScales(width: number, height: number, extent: Extent): PlaneScales {
  const x0 = extent.cx - extent.half;
  const y1 = extent.cy + extent.half;
  const span = 2 * extent.half;
  return {
    width,
    height,
    extent,
    toPx: (x, y) => [((x - x0) / span) * width, ((y1 - y) / span) * height],
    fromPx: (px, py) => [x0 + (px / width) * span, y1 - (py / height) * span],
  };
}

function drawField(ctx: CanvasRenderingContext2D, field: FieldLayer, s: PlaneScales) {
  const { values, res, extent } = field;
  const off = document.createElement("canvas");
  off.width = res;
  off.height = res;
  const octx = off.getContext("2d");
  if (!octx) return;
  const img = octx.createImageData(res, res);
  const c0 = hexToRgb(COLORS.class0);
  const c1 = hexToRgb(COLORS.class1);
  const opacity = field.opacity ?? 1;
  for (let i = 0; i < values.length; i++) {
    const p = values[i];
    const c = p > 0.5 ? c1 : c0;
    const confidence = Math.abs(p - 0.5) * 2;
    img.data[4 * i] = c[0];
    img.data[4 * i + 1] = c[1];
    img.data[4 * i + 2] = c[2];
    img.data[4 * i + 3] = Math.round(255 * opacity * (0.07 + 0.16 * confidence));
  }
  octx.putImageData(img, 0, 0);
  const [ax, ay] = s.toPx(extent.cx - extent.half, extent.cy + extent.half);
  const [bx, by] = s.toPx(extent.cx + extent.half, extent.cy - extent.half);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, ax, ay, bx - ax, by - ay);

  if (field.contour) {
    const [iso] = contours().size([res, res]).thresholds([0.5])(Array.from(values));
    const cell = (bx - ax) / res;
    ctx.beginPath();
    for (const polygon of iso.coordinates) {
      for (const ring of polygon) {
        ring.forEach(([x, y], i) => {
          // Skip edges along the lattice border: they are not part of the boundary.
          const onEdge = x <= 0.5 || y <= 0.5 || x >= res - 0.5 || y >= res - 0.5;
          const px = ax + x * cell;
          const py = ay + y * cell;
          if (i === 0 || onEdge) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
      }
    }
    ctx.strokeStyle = `rgba(255,255,255,${0.85 * opacity})`;
    ctx.lineWidth = 1.6;
    ctx.stroke();
  }
}

function drawGrid(ctx: CanvasRenderingContext2D, grid: Float32Array[], s: PlaneScales, opacity: number) {
  ctx.strokeStyle = COLORS.grid;
  ctx.globalAlpha = opacity;
  ctx.lineWidth = 1;
  for (const line of grid) {
    ctx.beginPath();
    for (let i = 0; i < line.length; i += 2) {
      const [px, py] = s.toPx(line[i], line[i + 1]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

export function Plane2D(props: Plane2DProps) {
  const { ref, width } = useElementSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const size = Math.max(0, Math.floor(width));
  const scales = useMemo(() => makeScales(size, size, props.extent), [size, props.extent]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    // Reassigning width/height reallocates the backing store, so only do it on resize.
    if (canvas.width !== size * dpr) canvas.width = size * dpr;
    if (canvas.height !== size * dpr) canvas.height = size * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    if (props.field) drawField(ctx, props.field, scales);
    if (props.grid) drawGrid(ctx, props.grid, scales, props.gridOpacity ?? 1);

    if (props.line) {
      const seg = clipLine(props.line.line, scales.extent);
      if (seg) {
        const [a, b] = seg.map(([x, y]) => scales.toPx(x, y));
        ctx.globalAlpha = props.line.opacity ?? 1;
        ctx.strokeStyle = props.line.color ?? COLORS.line;
        ctx.lineWidth = 2;
        ctx.setLineDash(props.line.dashed ? [6, 6] : []);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
    }

    const r = props.pointRadius ?? Math.max(2.2, size / 190);
    const { positions, labels, wrongMask } = props;
    ctx.globalAlpha = props.pointOpacity ?? 0.92;
    for (let i = 0; i < labels.length; i++) {
      const [px, py] = scales.toPx(positions[2 * i], positions[2 * i + 1]);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = classColor(labels[i]);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (wrongMask) {
      ctx.strokeStyle = COLORS.wrong;
      ctx.lineWidth = 1.5;
      for (let i = 0; i < labels.length; i++) {
        if (!wrongMask[i]) continue;
        const [px, py] = scales.toPx(positions[2 * i], positions[2 * i + 1]);
        ctx.beginPath();
        ctx.arc(px, py, r + 3, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    if (props.highlight != null && props.highlight >= 0) {
      const i = props.highlight;
      const [px, py] = scales.toPx(positions[2 * i], positions[2 * i + 1]);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, r + 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(px, py, r + 1.5, 0, Math.PI * 2);
      ctx.fillStyle = classColor(labels[i]);
      ctx.fill();
    }
  }, [props, scales, size]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!props.onPointPick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best = -1;
    let bestD = PICK_RADIUS_PX ** 2;
    for (let i = 0; i < props.labels.length; i++) {
      const [px, py] = scales.toPx(props.positions[2 * i], props.positions[2 * i + 1]);
      const d = (px - mx) ** 2 + (py - my) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best >= 0) props.onPointPick(best);
  };

  return (
    <figure className={`plane ${props.className ?? ""}`}>
      <div className="plane__box" ref={ref} role="img" aria-label={props.ariaLabel}>
        <canvas
          ref={canvasRef}
          style={{ width: size, height: size, cursor: props.onPointPick ? "pointer" : "default" }}
          onClick={handleClick}
        />
        {props.overlay && size > 0 && (
          <svg className="plane__overlay" width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
            {props.overlay(scales)}
          </svg>
        )}
      </div>
      {props.caption && <figcaption className="plane__caption">{props.caption}</figcaption>}
    </figure>
  );
}
