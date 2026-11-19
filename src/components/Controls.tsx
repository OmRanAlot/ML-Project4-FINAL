import type { ReactNode } from "react";

interface ScrubberProps {
  frame: number;
  count: number;
  playing: boolean;
  onToggle: () => void;
  onScrub: (f: number) => void;
  label: ReactNode;
  /** What is being played, for screen readers (e.g. "training", "depth"). */
  subject?: string;
}

export function PlayScrubber({ frame, count, playing, onToggle, onScrub, label, subject = "training" }: ScrubberProps) {
  return (
    <div className="scrubber">
      <button className="btn btn--icon" onClick={onToggle} aria-label={playing ? `Pause ${subject}` : `Play ${subject}`}>
        {playing ? (
          <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
            <rect x="3" y="2" width="3.5" height="12" rx="1" fill="currentColor" />
            <rect x="9.5" y="2" width="3.5" height="12" rx="1" fill="currentColor" />
          </svg>
        ) : (
          <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
            <path d="M4 2.5v11l9-5.5z" fill="currentColor" />
          </svg>
        )}
      </button>
      <input
        className="scrubber__range"
        type="range"
        min={0}
        max={count - 1}
        step={0.01}
        value={frame}
        onChange={(e) => onScrub(Number(e.target.value))}
        aria-label={`${subject[0].toUpperCase()}${subject.slice(1)} progress`}
        style={{ ["--pct" as string]: `${(frame / (count - 1)) * 100}%` }}
      />
      <span className="scrubber__label">{label}</span>
    </div>
  );
}

interface SegmentedProps<T extends string | number> {
  options: { value: T; label: ReactNode; color?: string }[];
  value: T;
  onChange: (v: T) => void;
  ariaLabel: string;
}

export function Segmented<T extends string | number>({ options, value, onChange, ariaLabel }: SegmentedProps<T>) {
  return (
    <div className="segmented" role="radiogroup" aria-label={ariaLabel}>
      {options.map((o) => (
        <button
          key={String(o.value)}
          role="radio"
          aria-checked={o.value === value}
          className={`segmented__opt ${o.value === value ? "is-on" : ""}`}
          style={o.color ? { ["--accent" as string]: o.color } : undefined}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Stat({ label, value, tone }: { label: ReactNode; value: ReactNode; tone?: "good" | "bad" | "neutral" }) {
  return (
    <div className={`stat stat--${tone ?? "neutral"}`}>
      <span className="stat__value">{value}</span>
      <span className="stat__label">{label}</span>
    </div>
  );
}

export function Toggle({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="toggle__track" aria-hidden />
      <span>{children}</span>
    </label>
  );
}
