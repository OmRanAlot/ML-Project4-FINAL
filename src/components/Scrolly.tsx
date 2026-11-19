import type { CSSProperties, ReactNode } from "react";
import { useScrollProgress } from "../hooks/useScrollProgress";

/** How much scrolling (in viewport heights) each step takes while the section is pinned. */
const VH_PER_STEP = 110;

interface ScrollyProps {
  steps: ReactNode[];
  /** Renders the pinned graphic for continuous progress t in [0, steps.length]. */
  graphic: (t: number, active: number) => ReactNode;
  id?: string;
}

/**
 * Pinned scrollytelling section: the whole stage (graphic + text) sticks to the viewport.
 * Scrolling drives the graphic continuously, while the text stays put and only swaps
 * when the reader crosses into the next step.
 */
export function Scrolly({ steps, graphic, id }: ScrollyProps) {
  const { containerRef, t, active, scrollToStep } = useScrollProgress(steps.length);
  const within = t - active;
  const style = {
    ["--scrolly-steps" as string]: steps.length,
    ["--scrolly-vh-per-step" as string]: `${VH_PER_STEP}vh`,
  } as CSSProperties;

  return (
    <div className="scrolly" ref={containerRef} id={id} style={style}>
      <div className="scrolly__stage">
        <div className="scrolly__graphic">{graphic(t, active)}</div>
        <div className="scrolly__text">
          <div className="scrolly__progress" aria-label="Steps">
            {steps.map((_, i) => (
              <button
                key={i}
                type="button"
                aria-current={i === active ? "step" : undefined}
                aria-label={`Go to step ${i + 1} of ${steps.length}`}
                className="scrolly__tick"
                onClick={() => scrollToStep(i)}
                style={{ ["--fill" as string]: i < active ? 1 : i === active ? within : 0 }}
              />
            ))}
            <span className="scrolly__count">
              {active + 1} / {steps.length}
            </span>
          </div>
          <div className="scrolly__cards">
            {steps.map((s, i) => (
              <div
                key={i}
                className={`scrolly__card ${i === active ? "is-active" : i < active ? "is-past" : ""}`}
                aria-hidden={i !== active}
              >
                {s}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
