/**
 * Legend for the two lines drawn on a plane: the dashed best straight line
 * (linear probe) and the solid white curve (the network's own decision boundary).
 */
import type { ReactNode } from "react";

const SWATCH_W = 26;
const SWATCH_H = 8;
const BOUNDARY_COLOR = "rgba(255,255,255,0.85)";

interface LineLegendProps {
  /** Colour of the dashed probe line; omit to hide that entry. */
  probeColor?: string;
  /** Text after the dashed swatch, e.g. accuracy. */
  probeLabel?: ReactNode;
  /** Show the solid "network's decision" entry. */
  showBoundary?: boolean;
}

function Swatch({ color, dashed }: { color: string; dashed: boolean }) {
  return (
    <svg className="legend__swatch" width={SWATCH_W} height={SWATCH_H} aria-hidden>
      <line
        x1={1}
        x2={SWATCH_W - 1}
        y1={SWATCH_H / 2}
        y2={SWATCH_H / 2}
        stroke={color}
        strokeWidth={dashed ? 1.6 : 1.8}
        strokeDasharray={dashed ? "5 4" : undefined}
        strokeLinecap="round"
      />
    </svg>
  );
}

export function LineLegend({ probeColor, probeLabel, showBoundary = true }: LineLegendProps) {
  return (
    <span className="legend">
      {probeColor && (
        <span className="legend__item">
          <Swatch color={probeColor} dashed />
          best straight line{probeLabel ? <> · {probeLabel}</> : null}
        </span>
      )}
      {showBoundary && (
        <span className="legend__item">
          <Swatch color={BOUNDARY_COLOR} dashed={false} />
          network's decision boundary
        </span>
      )}
    </span>
  );
}
