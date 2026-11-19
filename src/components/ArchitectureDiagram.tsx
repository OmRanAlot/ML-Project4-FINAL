/**
 * input(2) → 128 → h1(2) → 128 → h2(2) → 128 → h3(2) → ŷ
 * Neuron brightness shows the followed point's activation. Dark hollow neurons
 * are the ones ReLU switched off for that point.
 *
 * With `progress`, the diagram plays the point's journey: a wavefront sweeps left to right,
 * and each layer lights up (values, active neurons, edges) only once the wave reaches it.
 */
import { memo, useId, useMemo } from "react";
import { COLORS, LAYER_LABELS } from "../lib/theme";
import { HIDDEN } from "../lib/mlp";
import { JOURNEY_END } from "../lib/journey";

interface Props {
  /** Post-activation values (128) for each of the 3 blocks. */
  hidden: Float32Array[];
  /** 2D values at x, h1, h2, h3. */
  reps: [number, number][];
  probability: number;
  /** Highlight a block (0..2) or null. */
  focusBlock?: number | null;
  onFocusBlock?: (k: number) => void;
  /** Journey progress in [0, JOURNEY_END]; omit (or pass the end) to show everything. */
  progress?: number;
}

const W = 1000;
const H = 330;
const COL_X = [40, 170, 300, 430, 560, 690, 820, 950];
const TWO_Y = [H / 2 - 30, H / 2 + 30];
const DOT_ROWS = 64;
const DOT_GAP = 4.3;
const DOT_TOP = (H - DOT_ROWS * DOT_GAP) / 2;
const ACTIVE = "#ff6b6b";
const WAVE_WIDTH = 46;
/** Reveal a little ahead of the wave so a node lights as the wave touches it. */
const REVEAL_LEAD = 12;

function hiddenPos(colX: number, j: number): [number, number] {
  const sub = j % 2;
  const row = Math.floor(j / 2);
  return [colX - 3 + sub * 6, DOT_TOP + row * DOT_GAP];
}

/** x position of the wavefront for journey progress p (integer p sits on a column). */
function waveX(p: number): number {
  const c = Math.min(JOURNEY_END, Math.max(0, p));
  const i = Math.min(COL_X.length - 2, Math.floor(c));
  return COL_X[i] + (COL_X[i + 1] - COL_X[i]) * (c - i);
}

function blockBounds(b: number) {
  return { left: COL_X[2 * b], mid: COL_X[2 * b + 1], right: COL_X[2 * b + 2] };
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
            stroke={on ? ACTIVE : COLORS.muted}
            strokeOpacity={on ? 0.05 + 0.35 * (hidden[j] / max) : 0.025}
            strokeWidth={0.7}
          />
        ));
      })}
    </g>
  );
}

/** Everything the point has "lit": activations, values and the prediction. */
const LitLayer = memo(function LitLayer({ hidden, reps, probability, focusBlock }: Omit<Props, "progress" | "onFocusBlock">) {
  const maxes = useMemo(() => hidden.map((h) => Math.max(1e-6, ...h)), [hidden]);
  return (
    <g>
      {[0, 1, 2].map((b) => {
        const { left, mid, right } = blockBounds(b);
        const isFocus = focusBlock === b;
        return (
          <g key={b} opacity={focusBlock == null || isFocus ? 1 : 0.4}>
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
                <circle key={j} cx={x} cy={y} r={1.9} fill={ACTIVE} fillOpacity={0.25 + 0.75 * (v / maxes[b])} />
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
      <line x1={COL_X[6]} y1={TWO_Y[0]} x2={COL_X[7]} y2={H / 2} stroke={ACTIVE} strokeOpacity={0.6} />
      <line x1={COL_X[6]} y1={TWO_Y[1]} x2={COL_X[7]} y2={H / 2} stroke={ACTIVE} strokeOpacity={0.6} />
      <circle cx={COL_X[7]} cy={H / 2} r={14} fill={probability > 0.5 ? COLORS.class1 : COLORS.class0} />
      <text x={COL_X[7]} y={H / 2 - 24} textAnchor="middle" className="arch__layer" fill={COLORS.ink}>
        ŷ
      </text>
      <text x={COL_X[7]} y={H / 2 + 34} textAnchor="middle" className="arch__val">
        {probability.toFixed(2)}
      </text>
    </g>
  );
});

/** The bare network the point hasn't reached yet: structure only, no values. */
const DimLayer = memo(function DimLayer({ focusBlock }: { focusBlock?: number | null }) {
  return (
    <g opacity={0.55}>
      {[0, 1, 2].map((b) => {
        const { left, mid, right } = blockBounds(b);
        const isFocus = focusBlock === b;
        return (
          <g key={b} opacity={focusBlock == null || isFocus ? 1 : 0.5}>
            <rect x={left + 14} y={8} width={right - left - 28} height={H - 16} rx={14} className={`arch__block ${isFocus ? "is-focus" : ""}`} />
            <text x={mid} y={26} textAnchor="middle" className="arch__blocklabel">
              block {b + 1}
            </text>
            {Array.from({ length: HIDDEN }, (_, j) => {
              const [x, y] = hiddenPos(mid, j);
              return <circle key={j} cx={x} cy={y} r={1.4} fill={COLORS.muted} fillOpacity={0.25} />;
            })}
          </g>
        );
      })}
      {[0, 1, 2, 3].map((k) => (
        <g key={k}>
          {TWO_Y.map((y, t) => (
            <circle key={t} cx={COL_X[2 * k]} cy={y} r={9} fill={COLORS.bg} stroke={COLORS.muted} strokeOpacity={0.5} strokeWidth={1.5} />
          ))}
          <text x={COL_X[2 * k]} y={TWO_Y[0] - 40} textAnchor="middle" className="arch__layer" fill={COLORS.muted}>
            {LAYER_LABELS[k]}
          </text>
        </g>
      ))}
      <circle cx={COL_X[7]} cy={H / 2} r={14} fill="none" stroke={COLORS.muted} strokeOpacity={0.6} strokeDasharray="3 3" />
      <text x={COL_X[7]} y={H / 2 - 24} textAnchor="middle" className="arch__layer" fill={COLORS.muted}>
        ŷ
      </text>
    </g>
  );
});

function Wave({ x, gradientId }: { x: number; gradientId: string }) {
  return (
    <g pointerEvents="none">
      <rect x={x - WAVE_WIDTH} y={4} width={WAVE_WIDTH} height={H - 8} fill={`url(#${gradientId})`} />
      <line x1={x} y1={6} x2={x} y2={H - 6} stroke={COLORS.layers[3]} strokeOpacity={0.9} strokeWidth={2} />
      {TWO_Y.map((y) => (
        <circle key={y} cx={x} cy={y} r={5} fill="#fff" className="arch__pulse" />
      ))}
    </g>
  );
}

function ArchitectureDiagramImpl({ hidden, reps, probability, focusBlock, onFocusBlock, progress }: Props) {
  const uid = useId().replace(/:/g, "");
  const clipId = `arch-clip-${uid}`;
  const gradId = `arch-wave-${uid}`;
  const journeying = progress != null && progress < JOURNEY_END - 1e-3;
  const x = journeying ? waveX(progress) : W;

  return (
    <svg className="arch" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Network architecture with the followed point's activations">
      <defs>
        <clipPath id={clipId}>
          <rect x={0} y={0} width={x + REVEAL_LEAD} height={H} />
        </clipPath>
        <linearGradient id={gradId} x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" stopColor={COLORS.layers[3]} stopOpacity={0} />
          <stop offset="1" stopColor={COLORS.layers[3]} stopOpacity={0.22} />
        </linearGradient>
      </defs>

      {journeying && <DimLayer focusBlock={focusBlock} />}
      <g clipPath={journeying ? `url(#${clipId})` : undefined}>
        <LitLayer hidden={hidden} reps={reps} probability={probability} focusBlock={focusBlock} />
      </g>
      {journeying && <Wave x={x} gradientId={gradId} />}

      {/* Transparent hit targets so blocks stay clickable whether or not they're lit yet. */}
      {onFocusBlock &&
        [0, 1, 2].map((b) => {
          const { left, right } = blockBounds(b);
          return (
            <rect
              key={b}
              x={left + 14}
              y={8}
              width={right - left - 28}
              height={H - 16}
              fill="transparent"
              style={{ cursor: "pointer" }}
              onClick={() => onFocusBlock(b)}
            >
              <title>Zoom into block {b + 1}</title>
            </rect>
          );
        })}
    </svg>
  );
}

export const ArchitectureDiagram = memo(ArchitectureDiagramImpl);
