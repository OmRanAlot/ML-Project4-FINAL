/**
 * The depth tower: x, h1, h2, h3 as four planes in 3D, every point threaded through
 * all four, plus a ring of 128 bars between planes showing the followed point's hidden
 * layer (each 2D -> 2D jump really happens in 128 dimensions).
 */
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { useNetwork } from "../context/NetworkContext";
import { blockActivations, HIDDEN } from "../lib/mlp";
import { clipLine } from "../lib/geometry";
import { COLORS, LAYER_LABELS, formatPct } from "../lib/theme";
import type { Extent, RunKey } from "../lib/types";

const GAP = 2.6;
const RING_RADIUS = 0.42;
const HALF = 1;
const planeX = (k: number) => (k - 1.5) * GAP;

/** Representation coords -> plane-local (z, y) in [-1, 1], plane k sits at x = planeX(k). */
function toWorld(ext: Extent, k: number, x: number, y: number): [number, number, number] {
  return [planeX(k), ((y - ext.cy) / ext.half) * HALF, ((x - ext.cx) / ext.half) * HALF];
}

interface SceneProps {
  run: RunKey;
  frame: number;
  showPaths: boolean;
}

function Points({ run, frame, showPaths }: SceneProps) {
  const { model, selected, setSelected } = useNetwork();
  const mesh = useRef<THREE.InstancedMesh>(null);
  const paths = useRef<THREE.LineSegments>(null);
  const n = model.n;
  const extents = useMemo(() => [0, 1, 2, 3].map((k) => model.extentAt(run, k)), [model, run]);

  const colors = useMemo(() => {
    const arr = new Float32Array(n * 4 * 3);
    const c0 = new THREE.Color(COLORS.class0);
    const c1 = new THREE.Color(COLORS.class1);
    for (let k = 0; k < 4; k++)
      for (let i = 0; i < n; i++) (model.labels[i] ? c1 : c0).toArray(arr, (k * n + i) * 3);
    return arr;
  }, [model, n]);

  const pathGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3 * 2 * 3), 3));
    const col = new Float32Array(n * 3 * 2 * 3);
    for (let i = 0; i < n; i++)
      for (let s = 0; s < 6; s++) new THREE.Color(model.labels[i] ? COLORS.class1 : COLORS.class0).toArray(col, (i * 6 + s) * 3);
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return g;
  }, [model, n]);

  const invalidate = useThree((st) => st.invalidate);
  useEffect(() => () => pathGeom.dispose(), [pathGeom]);

  useLayoutEffect(() => {
    const m = mesh.current;
    if (!m) return;
    const tmp = new THREE.Object3D();
    const layers = [0, 1, 2, 3].map((k) => model.positionsAt(run, frame, k));
    const pos = pathGeom.getAttribute("position") as THREE.BufferAttribute;
    for (let k = 0; k < 4; k++) {
      for (let i = 0; i < n; i++) {
        const w = toWorld(extents[k], k, layers[k][2 * i], layers[k][2 * i + 1]);
        tmp.position.set(...w);
        tmp.scale.setScalar(i === selected ? 2.4 : 1);
        tmp.updateMatrix();
        m.setMatrixAt(k * n + i, tmp.matrix);
        if (k < 3) {
          const nxt = toWorld(extents[k + 1], k + 1, layers[k + 1][2 * i], layers[k + 1][2 * i + 1]);
          pos.setXYZ((i * 3 + k) * 2, ...w);
          pos.setXYZ((i * 3 + k) * 2 + 1, ...nxt);
        }
      }
    }
    m.instanceMatrix.needsUpdate = true;
    pos.needsUpdate = true;
    invalidate();
  }, [model, run, frame, n, extents, selected, pathGeom, invalidate]);

  return (
    <>
      <instancedMesh
        ref={mesh}
        args={[undefined, undefined, n * 4]}
        onClick={(e) => {
          e.stopPropagation();
          if (e.instanceId != null) setSelected(e.instanceId % n);
        }}
      >
        <sphereGeometry args={[0.028, 10, 10]} />
        <meshBasicMaterial toneMapped={false} />
        <instancedBufferAttribute attach="instanceColor" args={[colors, 3]} />
      </instancedMesh>
      <lineSegments ref={paths} geometry={pathGeom} visible={showPaths}>
        <lineBasicMaterial vertexColors transparent opacity={0.07} depthWrite={false} />
      </lineSegments>
    </>
  );
}

function Planes({ run, frame }: Omit<SceneProps, "showPaths">) {
  const { model } = useNetwork();
  return (
    <>
      {[0, 1, 2, 3].map((k) => {
        const ext = model.extentAt(run, k);
        const grid = model.gridAt(run, frame, k);
        const segs: [number, number, number][] = [];
        for (const line of grid) {
          for (let i = 0; i + 3 < line.length; i += 2) {
            const a = toWorld(ext, k, line[i], line[i + 1]);
            const b = toWorld(ext, k, line[i + 2], line[i + 3]);
            if (Math.abs(a[1]) > 1.05 || Math.abs(a[2]) > 1.05 || Math.abs(b[1]) > 1.05 || Math.abs(b[2]) > 1.05) continue;
            segs.push(a, b);
          }
        }
        const x = planeX(k);
        return (
          <group key={k}>
            <mesh position={[x - 0.002, 0, 0]} rotation={[0, Math.PI / 2, 0]}>
              <planeGeometry args={[2 * HALF, 2 * HALF]} />
              <meshBasicMaterial color={COLORS.layers[k]} transparent opacity={0.045} side={THREE.DoubleSide} depthWrite={false} />
            </mesh>
            <Line
              points={[[x, -1, -1], [x, 1, -1], [x, 1, 1], [x, -1, 1], [x, -1, -1]]}
              color={COLORS.layers[k]}
              lineWidth={1.2}
              transparent
              opacity={0.7}
            />
            {segs.length > 0 && <Line points={segs} segments color="#8fa6d8" lineWidth={0.6} transparent opacity={0.28} />}
          </group>
        );
      })}
      <HeadLine run={run} frame={frame} />
    </>
  );
}

/** The final simple model: a straight line in h3. */
function HeadLine({ run, frame }: Omit<SceneProps, "showPaths">) {
  const { model } = useNetwork();
  const net = model.net(run, frame);
  const ext = model.extentAt(run, 3);
  const seg = clipLine({ a: net.head.w0, b: net.head.w1, c: net.head.b }, ext);
  if (!seg) return null;
  return <Line points={seg.map(([x, y]) => toWorld(ext, 3, x, y))} color="#ffffff" lineWidth={2.5} />;
}

/** 128 bars on a ring between planes: the followed point's hidden activations. */
function HiddenRings({ run, frame }: Omit<SceneProps, "showPaths">) {
  const { model, selected } = useNetwork();
  const net = model.net(run, frame);
  const reps = model.reps(run, frame);
  const r = RING_RADIUS;
  return (
    <>
      {[0, 1, 2].map((b) => {
        const src = reps[b];
        const { post, pre } = blockActivations(net, b, src[2 * selected], src[2 * selected + 1]);
        const max = Math.max(1e-6, ...post.map(Math.abs));
        const cx = (planeX(b) + planeX(b + 1)) / 2;
        return (
          <group key={b} position={[cx, 0, 0]}>
            {Array.from({ length: HIDDEN }, (_, j) => {
              const th = (j / HIDDEN) * Math.PI * 2;
              const on = net.useRelu ? pre[j] > 0 : true;
              const h = on ? 0.03 + 0.24 * (Math.abs(post[j]) / max) : 0.015;
              return (
                <mesh key={j} position={[0, Math.sin(th) * r, Math.cos(th) * r]} rotation={[-th, 0, 0]}>
                  <boxGeometry args={[0.014, 0.018, h]} />
                  <meshBasicMaterial color={on ? "#ff7b7b" : "#2c303b"} toneMapped={false} />
                </mesh>
              );
            })}
          </group>
        );
      })}
    </>
  );
}

function SelectedPath({ run, frame }: Omit<SceneProps, "showPaths">) {
  const { model, selected } = useNetwork();
  const pts = [0, 1, 2, 3].map((k) => {
    const p = model.positionsAt(run, frame, k);
    return toWorld(model.extentAt(run, k), k, p[2 * selected], p[2 * selected + 1]);
  });
  return <Line points={pts} color="#ff6b6b" lineWidth={3} />;
}

/** Projects 3D anchor points to screen space every frame and moves DOM labels there. */
function LabelProjector({ anchors, labels }: { anchors: THREE.Vector3[]; labels: React.RefObject<(HTMLDivElement | null)[]> }) {
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ camera, size }) => {
    anchors.forEach((a, i) => {
      const el = labels.current?.[i];
      if (!el) return;
      v.copy(a).project(camera);
      const behind = v.z > 1;
      el.style.transform = `translate(-50%, -50%) translate(${((v.x + 1) / 2) * size.width}px, ${((1 - v.y) / 2) * size.height}px)`;
      el.style.opacity = behind ? "0" : "1";
    });
  });
  return null;
}

const ANCHORS = [
  ...[0, 1, 2, 3].map((k) => new THREE.Vector3(planeX(k), 1.3, 0)),
  ...[0, 1, 2].map((b) => new THREE.Vector3((planeX(b) + planeX(b + 1)) / 2, -RING_RADIUS - 0.4, 0)),
];

export function DepthTower3D({ run, frame, showPaths }: SceneProps) {
  const { model, selected } = useNetwork();
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const fi = Math.round(frame);
  const probe = model.data[run].frames[fi].probe;
  const net = model.net(run, frame);
  const reps = model.reps(run, frame);
  const offCounts = [0, 1, 2].map((b) => {
    const { pre } = blockActivations(net, b, reps[b][2 * selected], reps[b][2 * selected + 1]);
    return net.useRelu ? pre.filter((z) => z <= 0).length : 0;
  });

  return (
    <div className="tower">
      <Canvas frameloop="demand" camera={{ position: [-5.2, 2.4, 6.4], fov: 46 }} dpr={[1, 2]} gl={{ antialias: true }}>
        <color attach="background" args={[COLORS.bg]} />
        <Planes run={run} frame={frame} />
        <Points run={run} frame={frame} showPaths={showPaths} />
        <HiddenRings run={run} frame={frame} />
        <SelectedPath run={run} frame={frame} />
        <OrbitControls enableDamping makeDefault minDistance={3.5} maxDistance={16} target={[0, 0, 0]} />
        <LabelProjector anchors={ANCHORS} labels={labels} />
      </Canvas>
      <div className="tower__labels" aria-hidden>
        {[0, 1, 2, 3].map((k) => (
          <div key={k} className="tower__label" ref={(el) => void (labels.current[k] = el)}>
            <span style={{ color: COLORS.layers[k] }}>{LAYER_LABELS[k]}</span>
            <small>line can reach {formatPct(probe[k])}</small>
          </div>
        ))}
        {offCounts.map((off, b) => (
          <div key={b} className="tower__ring" ref={(el) => void (labels.current[4 + b] = el)}>
            128 hidden · {off} off
          </div>
        ))}
      </div>
    </div>
  );
}
