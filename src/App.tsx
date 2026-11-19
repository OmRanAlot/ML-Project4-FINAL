import { lazy, Suspense } from "react";
import { NetworkProvider } from "./context/NetworkContext";
import { Hero } from "./sections/Hero";
import { Challenge, TheIdea } from "./sections/Challenge";
import { InsideBlock } from "./sections/InsideBlock";
import { ThroughDepth } from "./sections/ThroughDepth";
import { ThroughTraining } from "./sections/ThroughTraining";
import { WhyRelu } from "./sections/WhyRelu";

// three.js is heavy; load the 3D section on demand.
const Tower = lazy(() => import("./sections/Tower").then((m) => ({ default: m.Tower })));
import { Payoff } from "./sections/Payoff";
import { Recap } from "./sections/Recap";

export function App() {
  return (
    <NetworkProvider>
      <main>
        <Hero />
        <Challenge />
        <TheIdea />
        <InsideBlock />
        <ThroughDepth />
        <ThroughTraining />
        <WhyRelu />
        <Suspense fallback={<div className="boot">Loading the 3D view…</div>}>
          <Tower />
        </Suspense>
        <Payoff />
        <Recap />
      </main>
    </NetworkProvider>
  );
}
