/**
 * input(2) → 128 → h1(2) → 128 → h2(2) → 128 → h3(2) → ŷ
 * Neuron brightness shows the followed point's activation. Dark hollow neurons
 * are the ones ReLU switched off for that point.
 */
import { memo, useMemo } from "react";
import { COLORS, LAYER_LABELS } from "../lib/theme";
import { HIDDEN } from "../lib/mlp";

interface Props {
  /** Post-activation values (128) for each of the 3 blocks. */
  hidden: Float32Array[];
  /** 2D values at x, h1, h2, h3. */
  reps: [number, number][];
  probability: number;
  /** Highlight a block (0..2) or null. */
  focusBlock?: number | null;
  onFocusBlock?: (k: number) => void;
}

const W = 1000;
const H = 330;
const COL_X = [40, 170, 300, 430, 560, 690, 820, 950];
const TWO_Y = [H / 2 - 30, H / 2 + 30];
const DOT_ROWS = 64;
const DOT_GAP = 4.3;
const DOT_TOP = (H - DOT_ROWS * DOT_GAP) / 2;

function hiddenPos(colX: number, j: number): [number, number] {
  const sub = j % 2;
  const row = Math.floor(j / 2);
  return [colX - 3 + sub * 6, DOT_TOP + row * DOT_GAP];
}

function Edges({ fromX, toX, hidden, fromHidden }: { fromX: number; toX: number; hidden: Float32Array; fromHidden: boolean }) {
  const max = Math.max(1e-6, ...hidden);
  return (
    <g>
      {Array.from({ length: HIDDEN }, (_, j) => {
        const on = hidden[j] > 0;
        const [hx, hy] = hiddenPos(fromHidden ? fromX : toX, j);
        return TWO_Y.map((ty, t) => (
          <line
            key={`${j}-${t}`}
            x1={fromHidden ? hx : fromX}
            y1={fromHidden ? hy : ty}
            x2={fromHidden ? toX : hx}
            y2={fromHidden ? ty : hy}
            stroke={on ? "#ff6b6b" : COLORS.muted}
            strokeOpacity={on ? 0.05 + 0.35 * (hidden[j] / max) : 0.025}
            strokeWidth={0.7}
          />
        ));
      })}
    </g>
  );
}

function ArchitectureDiagramImpl({ hidden, reps, probability, focusBlock, onFocusBlock }: Props) {
  const maxes = useMemo(() => hidden.map((h) => Math.max(1e-6, ...h)), [hidden]);
  return (
    <svg className="arch" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Network architecture with the followed point's activations">
      {[0, 1, 2].map((b) => {
        const left = COL_X[2 * b];
        const mid = COL_X[2 * b + 1];
        const right = COL_X[2 * b + 2];
        const isFocus = focusBlock === b;
        return (
          <g key={b} onClick={() => onFocusBlock?.(b)} style={{ cursor: onFocusBlock ? "pointer" : "default" }} opacity={focusBlock == null || isFocus ? 1 : 0.4}>
            <rect x={left + 14} y={8} width={right - left - 28} height={H - 16} rx={14} className={`arch__block ${isFocus ? "is-focus" : ""}`} />
            <text x={mid} y={26} textAnchor="middle" className="arch__blocklabel">
              block {b + 1}
            </text>
            <Edges fromX={left} toX={mid} hidden={hidden[b]} fromHidden={false} />
            <Edges fromX={mid} toX={right} hidden={hidden[b]} fromHidden />
            {Array.from({ length: HIDDEN }, (_, j) => {
              const [x, y] = hiddenPos(mid, j);
              const v = hidden[b][j];
              return v > 0 ? (
                <circle key={j} cx={x} cy={y} r={1.9} fill="#ff6b6b" fillOpacity={0.25 + 0.75 * (v / maxes[b])} />
              ) : (
                <circle key={j} cx={x} cy={y} r={1.6} fill="none" stroke={COLORS.muted} strokeOpacity={0.35} strokeWidth={0.6} />
              );
            })}
            <text x={mid} y={H - 16} textAnchor="middle" className="arch__dim">
              128 · ReLU
            </text>
          </g>
        );
      })}
      {[0, 1, 2, 3].map((k) => (
        <g key={k}>
          {TWO_Y.map((y, t) => (
            <g key={t}>
              <circle cx={COL_X[2 * k]} cy={y} r={9} fill={COLORS.bg} stroke={COLORS.layers[k]} strokeWidth={2} />
              <text x={COL_X[2 * k]} y={y + (t === 0 ? -16 : 24)} textAnchor="middle" className="arch__val">
                {reps[k][t].toFixed(2)}
              </text>
            </g>
          ))}
          <text x={COL_X[2 * k]} y={TWO_Y[0] - 40} textAnchor="middle" className="arch__layer" fill={COLORS.layers[k]}>
            {LAYER_LABELS[k]}
          </text>
        </g>
      ))}
      <line x1={COL_X[6]} y1={TWO_Y[0]} x2={COL_X[7]} y2={H / 2} stroke="#ff6b6b" strokeOpacity={0.6} />
      <line x1={COL_X[6]} y1={TWO_Y[1]} x2={COL_X[7]} y2={H / 2} stroke="#ff6b6b" strokeOpacity={0.6} />
      <circle cx={COL_X[7]} cy={H / 2} r={14} fill={probability > 0.5 ? COLORS.class1 : COLORS.class0} />
      <text x={COL_X[7]} y={H / 2 - 24} textAnchor="middle" className="arch__layer" fill={COLORS.ink}>
        ŷ
      </text>
      <text x={COL_X[7]} y={H / 2 + 34} textAnchor="middle" className="arch__val">
        {probability.toFixed(2)}
      </text>
    </svg>
  );
}

export const ArchitectureDiagram = memo(ArchitectureDiagramImpl);
