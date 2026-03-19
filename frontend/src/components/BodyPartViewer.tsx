import { useRef, useState, Suspense } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Sphere, Torus, MeshDistortMaterial, Float, Stars } from "@react-three/drei";
import * as THREE from "three";
import { motion, AnimatePresence } from "framer-motion";
import {
  Heart, Brain, Wind, Droplets, Activity, ZoomIn, ZoomOut,
  RotateCcw, Pause, Play, Maximize2,
} from "lucide-react";

/* ─── Organ parts definition ───────────────────────────── */
const ORGANS = [
  { id: "heart",  label: "Heart",   icon: Heart,    color: "#FF4D6D", glow: "#FF4D6D" },
  { id: "brain",  label: "Brain",   icon: Brain,    color: "#9D6EFF", glow: "#9D6EFF" },
  { id: "lungs",  label: "Lungs",   icon: Wind,     color: "#4DC9FF", glow: "#4DC9FF" },
  { id: "kidney", label: "Kidney",  icon: Droplets,  color: "#FF9A3C", glow: "#FF9A3C" },
  { id: "system", label: "System",  icon: Activity, color: "#00E5A0", glow: "#00E5A0" },
] as const;

type OrganId = typeof ORGANS[number]["id"];

/* ─── 3D Scene Meshes ───────────────────────────────────── */
function HeartMesh({ color }: { color: string }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const torusRef = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (meshRef.current) {
      // heartbeat oscillation
      const beat = 1 + Math.sin(Date.now() * 0.006) * 0.06;
      meshRef.current.scale.setScalar(beat);
      meshRef.current.rotation.y += delta * 0.4;
    }
    if (torusRef.current) {
      torusRef.current.rotation.y += delta * 0.3;
      torusRef.current.rotation.x += delta * 0.15;
    }
  });

  return (
    <group>
      {/* Core sphere (heart body) */}
      <mesh ref={meshRef}>
        <Sphere args={[1.1, 64, 64]}>
          <MeshDistortMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.4}
            distort={0.35}
            speed={3}
            roughness={0.15}
            metalness={0.3}
            transparent
            opacity={0.92}
          />
        </Sphere>
      </mesh>

      {/* Orbital ring */}
      <mesh ref={torusRef}>
        <Torus args={[1.8, 0.04, 16, 120]}>
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.6}
            transparent
            opacity={0.5}
          />
        </Torus>
      </mesh>
    </group>
  );
}

function BrainMesh({ color }: { color: string }) {
  const meshRef = useRef<THREE.Mesh>(null);
  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.35;
      const pulse = 1 + Math.sin(Date.now() * 0.002) * 0.04;
      meshRef.current.scale.setScalar(pulse);
    }
  });
  return (
    <mesh ref={meshRef}>
      <Sphere args={[1.2, 64, 64]}>
        <MeshDistortMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.35}
          distort={0.55}
          speed={1.5}
          roughness={0.2}
          metalness={0.2}
          transparent
          opacity={0.9}
        />
      </Sphere>
    </mesh>
  );
}

function LungsMesh({ color }: { color: string }) {
  const lRef = useRef<THREE.Mesh>(null);
  const rRef = useRef<THREE.Mesh>(null);
  useFrame(() => {
    const breath = 1 + Math.sin(Date.now() * 0.0015) * 0.1;
    lRef.current?.scale.setScalar(breath);
    rRef.current?.scale.setScalar(breath);
  });
  return (
    <group>
      <mesh ref={lRef} position={[-0.9, 0, 0]}>
        <Sphere args={[0.85, 48, 48]}>
          <MeshDistortMaterial color={color} emissive={color} emissiveIntensity={0.4} distort={0.3} speed={2} transparent opacity={0.88} roughness={0.3} />
        </Sphere>
      </mesh>
      <mesh ref={rRef} position={[0.9, 0, 0]}>
        <Sphere args={[0.85, 48, 48]}>
          <MeshDistortMaterial color={color} emissive={color} emissiveIntensity={0.4} distort={0.3} speed={2} transparent opacity={0.88} roughness={0.3} />
        </Sphere>
      </mesh>
    </group>
  );
}

function GenericOrganMesh({ color }: { color: string }) {
  const meshRef = useRef<THREE.Mesh>(null);
  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.5;
      const p = 1 + Math.sin(Date.now() * 0.003) * 0.05;
      meshRef.current.scale.setScalar(p);
    }
  });
  return (
    <mesh ref={meshRef}>
      <Sphere args={[1.1, 64, 64]}>
        <MeshDistortMaterial
          color={color} emissive={color} emissiveIntensity={0.35}
          distort={0.4} speed={2} roughness={0.2} metalness={0.25} transparent opacity={0.9}
        />
      </Sphere>
    </mesh>
  );
}

function Scene({ organ }: { organ: OrganId }) {
  const cfg = ORGANS.find((o) => o.id === organ)!;

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[4, 6, 4]}  intensity={2.5} color={cfg.color} />
      <pointLight position={[-4, -4, -4]} intensity={1}  color="#ffffff" />
      <pointLight position={[0, -4, 4]}  intensity={1.2} color={cfg.color} />
      <Stars radius={60} depth={50} count={1200} factor={2} saturation={0} fade speed={1} />

      <Float speed={2} rotationIntensity={0.15} floatIntensity={0.4}>
        {organ === "heart"  && <HeartMesh  color={cfg.color} />}
        {organ === "brain"  && <BrainMesh  color={cfg.color} />}
        {organ === "lungs"  && <LungsMesh  color={cfg.color} />}
        {organ === "kidney" && <GenericOrganMesh color={cfg.color} />}
        {organ === "system" && <GenericOrganMesh color={cfg.color} />}
      </Float>

      <OrbitControls enablePan={false} enableZoom={true} minDistance={2.5} maxDistance={8} autoRotate={false} />
    </>
  );
}

/* ─── Stats per Organ ───────────────────────────────────── */
const ORGAN_STATS: Record<OrganId, Array<{ label: string; value: string; unit: string; color: string }>> = {
  heart:  [
    { label: "Heart Rate",      value: "72",    unit: "bpm",  color: "#FF4D6D" },
    { label: "Binding Score",   value: "0.94",  unit: "",     color: "#FF8FA3" },
    { label: "Confidence",      value: "97.2",  unit: "%",    color: "#FF4D6D" },
  ],
  brain:  [
    { label: "BBBP Score",      value: "0.81",  unit: "",     color: "#9D6EFF" },
    { label: "Target Affinity", value: "−8.4",  unit: "kcal", color: "#B89FFF" },
    { label: "Confidence",      value: "91.0",  unit: "%",    color: "#9D6EFF" },
  ],
  lungs:  [
    { label: "Cmax (lung)",     value: "2.1",   unit: "µM",   color: "#4DC9FF" },
    { label: "T½",              value: "6.5",   unit: "h",    color: "#80DEFF" },
    { label: "Confidence",      value: "88.5",  unit: "%",    color: "#4DC9FF" },
  ],
  kidney: [
    { label: "Renal Clearance", value: "0.62",  unit: "L/h",  color: "#FF9A3C" },
    { label: "GFR Impact",      value: "Mild",  unit: "",     color: "#FFBA80" },
    { label: "Safety",          value: "Grade B", unit: "",   color: "#FF9A3C" },
  ],
  system: [
    { label: "ADMET Score",     value: "0.88",  unit: "",     color: "#00E5A0" },
    { label: "Bioavailability", value: "73",    unit: "%",    color: "#50FFBF" },
    { label: "Clin. Readiness", value: "Phase 1", unit: "",  color: "#00E5A0" },
  ],
};

/* ─── Main Component ─────────────────────────────────────── */
interface BodyPartViewerProps {
  className?: string;
  title?: string;
}

export default function BodyPartViewer({ className = "", title }: BodyPartViewerProps) {
  const [activeOrgan, setActiveOrgan] = useState<OrganId>("heart");
  const [rotating, setRotating]       = useState(true);
  const cfg = ORGANS.find((o) => o.id === activeOrgan)!;
  const stats = ORGAN_STATS[activeOrgan];

  return (
    <div
      className={`glass-card rounded-3xl overflow-hidden relative ${className}`}
      style={{ minHeight: 480 }}
    >
      {/* Top accent */}
      <div
        className="absolute top-0 left-6 right-6 h-[2px] rounded-full z-10"
        style={{ background: `linear-gradient(90deg, transparent, ${cfg.glow}, transparent)` }}
      />

      {/* Header */}
      <div className="absolute top-4 left-6 right-6 z-10 flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            3D Organ Viewer
          </p>
          <h2 className="text-base font-bold mt-0.5">
            {title ?? cfg.label}
            <span className="ml-2 text-xs font-normal text-muted-foreground">· Drug Action Site</span>
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setRotating((r) => !r)}
            className="h-8 w-8 flex items-center justify-center rounded-xl glass-surface text-muted-foreground hover:text-foreground transition-all"
          >
            {rotating ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </button>
          <div
            className="h-8 px-3 flex items-center gap-1.5 rounded-xl text-xs font-semibold"
            style={{
              background: `${cfg.glow}20`,
              color: cfg.glow,
              border: `1px solid ${cfg.glow}40`,
            }}
          >
            <span className="h-1.5 w-1.5 rounded-full animate-pulse" style={{ background: cfg.glow }} />
            Live
          </div>
        </div>
      </div>

      {/* 3D Canvas */}
      <div style={{ height: 360 }}>
        <Canvas camera={{ position: [0, 0, 5], fov: 45 }} gl={{ antialias: true }} style={{ background: "transparent" }}>
          <Suspense fallback={null}>
            <Scene organ={activeOrgan} />
          </Suspense>
        </Canvas>
      </div>

      {/* Bottom: Organ Picker + Stats */}
      <div className="px-5 pb-5 space-y-4">
        {/* Organ pills */}
        <div className="flex gap-2 flex-wrap">
          {ORGANS.map((o) => {
            const isActive = o.id === activeOrgan;
            return (
              <button
                key={o.id}
                onClick={() => setActiveOrgan(o.id)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all duration-300"
                style={
                  isActive
                    ? {
                        background: `${o.glow}25`,
                        color: o.glow,
                        border: `1px solid ${o.glow}50`,
                        boxShadow: `0 4px 16px ${o.glow}30`,
                      }
                    : {
                        background: "var(--glass-bg)",
                        color: "hsl(var(--muted-foreground))",
                        border: "1px solid var(--glass-border)",
                      }
                }
              >
                <o.icon style={{ width: 12, height: 12 }} />
                {o.label}
              </button>
            );
          })}
        </div>

        {/* Dynamic Stats */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeOrgan}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
            className="grid grid-cols-3 gap-3"
          >
            {stats.map((s) => (
              <div
                key={s.label}
                className="glass-surface rounded-2xl p-3 text-center"
              >
                <p className="text-xl font-bold" style={{ color: s.color }}>
                  {s.value}
                  {s.unit && <span className="text-xs font-normal text-muted-foreground ml-1">{s.unit}</span>}
                </p>
                <p className="text-xs text-muted-foreground mt-0.5">{s.label}</p>
              </div>
            ))}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
