import { useCallback, useEffect, useRef, useState } from "react";

/** Plays a fractional frame index from 0 to frameCount - 1 at `fps` snapshots per second. */
export function usePlayback(frameCount: number, fps = 14, initial = frameCount - 1) {
  const [frame, setFrameState] = useState(initial);
  const [playing, setPlaying] = useState(false);
  const frameRef = useRef(initial);

  const setFrame = useCallback((f: number) => {
    frameRef.current = f;
    setFrameState(f);
  }, []);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      // Cap the step so returning from a background tab doesn't jump to the end.
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const next = frameRef.current + dt * fps;
      if (next >= frameCount - 1) {
        setFrame(frameCount - 1);
        setPlaying(false);
        return;
      }
      setFrame(next);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, fps, frameCount, setFrame]);

  const toggle = useCallback(() => {
    if (!playing && frameRef.current >= frameCount - 1) setFrame(0);
    setPlaying((p) => !p);
  }, [playing, frameCount, setFrame]);

  const scrub = useCallback(
    (f: number) => {
      setPlaying(false);
      setFrame(f);
    },
    [setFrame],
  );

  return { frame, playing, toggle, scrub };
}
