import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Scroll progress through a pinned (sticky) section.
 *
 * The container is tall; a 100vh stage inside it sticks to the viewport while the reader
 * scrolls through the container's extra height. That extra height is split evenly into
 * `stepCount` steps. Returns t in [0, stepCount]: the integer part is the active step,
 * the fraction is how far the reader has scrolled through it.
 */
export function useScrollProgress(stepCount: number) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [t, setT] = useState(0);

  useEffect(() => {
    let raf = 0;
    const measure = () => {
      raf = 0;
      const root = containerRef.current;
      if (!root) return;
      const rect = root.getBoundingClientRect();
      const scrollable = Math.max(1, rect.height - window.innerHeight);
      const progress = Math.min(1, Math.max(0, -rect.top / scrollable));
      setT(progress * stepCount);
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
  }, [stepCount]);

  /** Smooth-scroll to the start of step `i` (nudged slightly inside it). */
  const scrollToStep = useCallback(
    (i: number) => {
      const root = containerRef.current;
      if (!root) return;
      const rect = root.getBoundingClientRect();
      const scrollable = Math.max(1, rect.height - window.innerHeight);
      const containerTop = window.scrollY + rect.top;
      const target = containerTop + ((i + 0.02) / stepCount) * scrollable;
      window.scrollTo({ top: target, behavior: "smooth" });
    },
    [stepCount],
  );

  return { containerRef, t, active: Math.min(stepCount - 1, Math.floor(t)), scrollToStep };
}
