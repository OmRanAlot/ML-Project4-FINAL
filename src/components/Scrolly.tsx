import type { ReactNode } from "react";
import { useScrollProgress } from "../hooks/useScrollProgress";

interface ScrollyProps {
  steps: ReactNode[];
  /** Renders the sticky graphic for continuous progress t in [0, steps.length]. */
  graphic: (t: number, active: number) => ReactNode;
  id?: string;
}

/** Sticky graphic + scrolling prose steps, in the style of the matrix explorable. */
export function Scrolly({ steps, graphic, id }: ScrollyProps) {
  const { containerRef, t, active } = useScrollProgress(steps.length);
  const within = t - Math.floor(t);
  return (
    <div className="scrolly" ref={containerRef} id={id}>
      <div className="scrolly__graphic">
        <div className="scrolly__sticky">
          {graphic(t, active)}
          <div className="scrolly__progress" aria-hidden>
            {steps.map((_, i) => (
              <span
                key={i}
                className="scrolly__tick"
                style={{ ["--fill" as string]: i < active ? 1 : i === active ? within : 0 }}
              />
            ))}
          </div>
        </div>
      </div>
      <div className="scrolly__steps">
        {steps.map((s, i) => (
          <div key={i} data-step={i} className={`scrolly__step ${i === active ? "is-active" : ""}`}>
            <div className="scrolly__card">{s}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
