import { useEffect, useRef, useState } from "react";

/**
 * Continuous scroll position through a list of step elements.
 * Returns t in [0, stepCount]: the integer part is the active step, the fraction is
 * how far the reader has scrolled through it (measured at the reading line).
 */
export function useScrollProgress(stepCount: number, readingLine = 0.55) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [t, setT] = useState(0);

  useEffect(() => {
    let raf = 0;
    const measure = () => {
      raf = 0;
      const root = containerRef.current;
      if (!root) return;
      const steps = root.querySelectorAll<HTMLElement>("[data-step]");
      const line = window.innerHeight * readingLine;
      let total = 0;
      steps.forEach((el) => {
        const r = el.getBoundingClientRect();
        total += Math.min(1, Math.max(0, (line - r.top) / Math.max(1, r.height)));
      });
      setT(Math.min(stepCount, total));
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(measure);
    };
    measure();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [stepCount, readingLine]);

  return { containerRef, t, active: Math.min(stepCount - 1, Math.floor(t)) };
}
