import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles/global.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root element");

// No <StrictMode>: its dev-only double mount races react-three-fiber's <Canvas>
// teardown under React 19 and can leave the 3D scene blank in development.
createRoot(root).render(<App />);
