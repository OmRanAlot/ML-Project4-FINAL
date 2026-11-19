import { Tex } from "../components/Tex";
import { useNetwork } from "../context/NetworkContext";

export function Recap() {
  const { model } = useNetwork();
  const { meta } = model.data;
  const frames = model.data.relu.frames;
  return (
    <section className="section recap" id="recap">
      <div className="prose">
        <h2>
          <span className="num">08</span>Recap
        </h2>
        <ol className="takeaways">
          <li>
            <b>A line has a ceiling.</b> On two moons, no straight line gets past ~87%. The limit comes from the shape of the
            data.
          </li>
          <li>
            <b>Layers change the representation.</b> Each block maps every point to a new 2D position via a 128-dimensional
            detour. The data never changes; the coordinates we describe it in do.
          </li>
          <li>
            <b>ReLU is what lets space bend.</b> Without it, any number of layers collapses into one matrix, and grid lines
            stay straight.
          </li>
          <li>
            <b>Depth spreads the work out.</b> Each block makes a modest change. Together they turn a curved boundary into a
            straight one.
          </li>
          <li>
            <b>Training shapes the transformation.</b> Over training, every representation drifts toward being
            easier to separate with a line, until the final simple model is enough.
          </li>
        </ol>

        <h3>Things this page is not saying</h3>
        <ul className="caveats">
          <li>
            <b>Points don't physically travel.</b> The animations glide between before and after, but each block computes
            its output in one step.
          </li>
          <li>
            <b>Hidden layers aren't 2D.</b> Only the bottlenecks <Tex math="h_1, h_2, h_3" /> are. Each of them is produced
            by a 128-dimensional hidden layer we can only show as bars.
          </li>
          <li>
            <b>Real networks aren't usually this gradual.</b> We set this one up to be watchable (see below). Big networks
            transform data in ways that are far harder to see, but the principle is the same.
          </li>
        </ul>

        <details className="behind">
          <summary>Behind the scenes: how this network was built</summary>
          <p>
            <b>Data:</b> {model.n} points from scikit-learn's <code>make_moons</code> (noise {meta.noise}, seed {meta.seed}),
            standardized.
          </p>
          <p>
            <b>Architecture:</b> {meta.architecture}. The final logistic regression is the only place where the classes are
            decided.
          </p>
          <p>
            <b>Making it watchable.</b> Three choices make the transformation gradual in both depth and training time:
          </p>
          <ul>
            <li>
              <b>Near-identity start:</b> each block's second layer is solved by least squares so that, at step 0, the
              block reproduces its input. Training starts from "do nothing" instead of a random scramble.
            </li>
            <li>
              <b>Depth-smoothness penalty:</b> the loss adds <Tex math={String.raw`\lambda \sum_k \lVert h_k - h_{k-1}\rVert^2`} />{" "}
              with <Tex math={`\\lambda = ${meta.smoothLambda}`} />. A sum of squared step sizes is smallest when the work is
              split evenly, so no single block does everything.
            </li>
            <li>
              <b>Small, steady steps:</b> full-batch Adam with learning rate {meta.learningRate}, run for{" "}
              {frames[frames.length - 1].step} steps. Snapshots are saved whenever the points have moved a fixed distance,
              so the replay moves at an even pace.
            </li>
          </ul>
          <p>
            The browser re-runs every forward pass from the {frames.length} saved weight snapshots. That's how it can warp
            grids, draw decision regions and follow any point you click. The no-ReLU twin uses identical settings.
          </p>
        </details>

        <h3>Inspiration</h3>
        <p className="muted">
          3Blue1Brown's <em>Essence of Linear Algebra</em> and neural network series; Chris Olah, "Neural Networks,
          Manifolds, and Topology" (2014); Yizhe Ang's <em>Matrix Explorable</em>; TensorFlow Playground; Karpathy's
          ConvNetJS 2D demo; R2D3's <em>A Visual Introduction to Machine Learning</em>.
        </p>
      </div>
      <footer className="footer">Om Rana · CS375H · Made with React, D3 and three.js</footer>
    </section>
  );
}
