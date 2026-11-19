import { useEffect, useRef, useState } from "react";
import { Plane2D } from "../components/Plane2D";
import { useNetwork } from "../context/NetworkContext";
import { easeInOut } from "../lib/geometry";

const CYCLE_SECONDS = 9;

/** Depth oscillates input -> h3 -> input, with pauses at both ends. */
function depthAt(seconds: number): number {
  const phase = (seconds % CYCLE_SECONDS) / CYCLE_SECONDS;
  if (phase < 0.15) return 0;
  if (phase < 0.5) return 3 * easeInOut((phase - 0.15) / 0.35);
  if (phase < 0.65) return 3;
  return 3 * (1 - easeInOut((phase - 0.65) / 0.35));
}

export function Hero() {
  const { model } = useNetwork();
  const [seconds, setSeconds] = useState(0);
  const heroRef = useRef<HTMLElement>(null);
  const last = model.frameCount("relu") - 1;

  // Animate only while the hero is on screen (and never for reduced-motion readers).
  useEffect(() => {
    const el = heroRef.current;
    if (!el || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let raf = 0;
    let elapsed = 0;
    let prev = 0;
    const tick = (now: number) => {
      elapsed += Math.min(0.1, (now - (prev || now)) / 1000);
      prev = now;
      setSeconds(elapsed);
      raf = requestAnimationFrame(tick);
    };
    const io = new IntersectionObserver(([entry]) => {
      cancelAnimationFrame(raf);
      prev = 0;
      if (entry.isIntersecting) raf = requestAnimationFrame(tick);
    });
    io.observe(el);
    return () => {
      io.disconnect();
      cancelAnimationFrame(raf);
    };
  }, []);

  const depth = depthAt(seconds);
  return (
    <header className="hero" ref={heroRef}>
      <div className="hero__text">
        <p className="kicker">An interactive explainer · CS375H</p>
        <h1>
          Make it <em>linear</em>.
        </h1>
        <p className="hero__lede">
          How a neural network reshapes data, one layer and one training step at a time, until a plain straight line can
          finish the job.
        </p>
        <p className="byline">Om Rana</p>
        <a className="hero__cue" href="#challenge">
          Start with a challenge ↓
        </a>
      </div>
      <div className="hero__art" aria-hidden>
        <Plane2D
          positions={model.positionsAt("relu", last, depth)}
          labels={model.labels}
          extent={model.extentAt("relu", depth)}
          grid={model.gridAt("relu", last, depth)}
          gridOpacity={0.8}
        />
      </div>
    </header>
  );
}
