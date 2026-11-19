import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { createModel, type NetworkModel } from "../lib/model";
import { scoreLine } from "../lib/geometry";
import type { NetworkData } from "../lib/types";

interface NetworkContextValue {
  model: NetworkModel;
  /** Index of the point the reader is following through the network. */
  selected: number;
  setSelected: (i: number) => void;
}

const Ctx = createContext<NetworkContextValue | null>(null);

type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; model: NetworkModel };

function isNetworkData(value: unknown): value is NetworkData {
  const v = value as NetworkData;
  return Boolean(
    v && Array.isArray(v.data?.x) && Array.isArray(v.data?.y) && v.relu?.frames?.length && v.linear?.frames?.length,
  );
}

/** A point the best input-space line gets wrong: the most interesting one to follow. */
function pickDefaultPoint(model: NetworkModel): number {
  const last = model.frameCount("relu") - 1;
  const input = model.reps("relu", last)[0];
  const score = scoreLine(model.probeLine("relu", last, 0), input, model.labels);
  const idx = score.wrongMask.findIndex((w, i) => w === 1 && model.labels[i] === 0);
  return idx >= 0 ? idx : 0;
}

export function NetworkProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    let cancelled = false;
    fetch(`${import.meta.env.BASE_URL}data/network.json`)
      .then((res) => {
        if (!res.ok) throw new Error(`Could not load the trained network (HTTP ${res.status}).`);
        return res.json();
      })
      .then((json: unknown) => {
        if (!isNetworkData(json)) throw new Error("The network file is malformed. Re-run `npm run train`.");
        const model = createModel(json);
        if (!cancelled) {
          setSelected(pickDefaultPoint(model));
          setState({ status: "ready", model });
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const value = useMemo(
    () => (state.status === "ready" ? { model: state.model, selected, setSelected } : null),
    [state, selected],
  );

  if (state.status === "loading") {
    return <div className="boot">Loading the trained network…</div>;
  }
  if (state.status === "error" || !value) {
    return <div className="boot boot--error">{state.status === "error" ? state.message : "Unknown error"}</div>;
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useNetwork(): NetworkContextValue {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useNetwork must be used inside <NetworkProvider>");
  return ctx;
}
