/**
 * The 128 hidden numbers for one point inside one block, sorted from most negative
 * to most positive. `relu` in [0, 1] animates ReLU clipping the negatives to zero.
 */
import { useMemo } from "react";
import { scaleLinear } from "d3";
import { useElementSize } from "../hooks/useElementSize";
import { COLORS } from "../lib/theme";

interface Props {
  pre: Float32Array;
  relu: number;
  color: string;
}

const HEIGHT = 200;

export function NeuronBars({ pre, relu, color }: Props) {
  const { ref, width } = useElementSize<HTMLDivElement>();
  const sorted = useMemo(() => Array.from(pre).sort((a, b) => a - b), [pre]);
  const extent = useMemo(() => Math.max(1e-3, ...sorted.map(Math.abs)), [sorted]);
  const y = scaleLinear().domain([-extent, extent]).range([HEIGHT - 10, 10]);
  const barW = width / sorted.length;
  const off = sorted.filter((v) => v <= 0).length;

  return (
    <div className="bars" ref={ref}>
      {width > 0 && (
        <svg width={width} height={HEIGHT} role="img" aria-label={`128 hidden values; ${off} are negative and ReLU sets them to zero`}>
          <line x1={0} x2={width} y1={y(0)} y2={y(0)} stroke={COLORS.muted} strokeOpacity={0.6} />
          {sorted.map((v, i) => {
            const shown = v < 0 ? v * (1 - relu) : v;
            const top = Math.min(y(0), y(shown));
            return (
              <rect
                key={i}
                x={i * barW + 0.5}
                width={Math.max(0.8, barW - 1)}
                y={top}
                height={Math.max(0.5, Math.abs(y(shown) - y(0)))}
                fill={v < 0 ? COLORS.wrong : color}
                fillOpacity={v < 0 ? 0.25 + 0.6 * (1 - relu) : 0.9}
              />
            );
          })}
          <text x={6} y={HEIGHT - 4} className="bars__note">
            {off} of 128 negative → {relu > 0.5 ? "set to 0 by ReLU" : "about to be clipped"}
          </text>
        </svg>
      )}
    </div>
  );
}
