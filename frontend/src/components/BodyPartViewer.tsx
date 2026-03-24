import { useRef, useState, Suspense } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html, Float } from "@react-three/drei";
import * as THREE from "three";
import { motion, AnimatePresence } from "framer-motion";
import {
  Heart, Brain, Wind, Droplets, Activity,
  Pause, Play,
} from "lucide-react";
import { cn } from "@/lib/utils";

/* ─── Organ Node Definitions ──────────────────── */
const ORGAN_NODES = [
  { id: "brain",  label: "Brain",   icon: Brain,    position: [0, 2.85, 0.3] as [number,number,number], color: "#9D6EFF", number: "01" },
  { id: "heart",  label: "Heart",   icon: Heart,    position: [-0.4, 1.2, 0.5] as [number,number,number], color: "#FF4D6D", number: "02" },
  { id: "lungs",  label: "Lungs",   icon: Wind,     position: [0.5, 1.5, 0.4] as [number,number,number], color: "#4DC9FF", number: "03" },
  { id: "kidney", label: "Kidney",  icon: Droplets, position: [-0.5, 0.3, 0.4] as [number,number,number], color: "#FF9A3C", number: "04" },
  { id: "system", label: "System",  icon: Activity, position: [0.4, -0.3, 0.4] as [number,number,number], color: "#00E5A0", number: "05" },
] as const;

type OrganId = typeof ORGAN_NODES[number]["id"];

/* ─── Stats per Organ ──────────────────────────── */
const ORGAN_STATS: Record<OrganId, Array<{ label: string; value: string; unit: string; color: string }>> = {
  heart: [
    { label: "Heart Rate",    value: "72",   unit: "bpm",  color: "#FF4D6D" },
    { label: "Binding Score", value: "0.94", unit: "",     color: "#FF8FA3" },
    { label: "Confidence",    value: "97.2", unit: "%",    color: "#FF4D6D" },
  ],
  brain: [
    { label: "BBBP Score",      value: "0.81", unit: "",     color: "#9D6EFF" },
    { label: "Target Affinity", value: "−8.4", unit: "kcal", color: "#B89FFF" },
    { label: "Confidence",      value: "91.0", unit: "%",    color: "#9D6EFF" },
  ],
  lungs: [
    { label: "Cmax (lung)", value: "2.1",  unit: "µM", color: "#4DC9FF" },
    { label: "T½",          value: "6.5",  unit: "h",  color: "#80DEFF" },
    { label: "Confidence",  value: "88.5", unit: "%",  color: "#4DC9FF" },
  ],
  kidney: [
    { label: "Renal Clearance", value: "0.62",    unit: "L/h", color: "#FF9A3C" },
    { label: "GFR Impact",      value: "Mild",    unit: "",    color: "#FFBA80" },
    { label: "Safety",          value: "Grade B", unit: "",    color: "#FF9A3C" },
  ],
  system: [
    { label: "ADMET Score",     value: "0.88",    unit: "",  color: "#00E5A0" },
    { label: "Bioavailability", value: "73",      unit: "%", color: "#50FFBF" },
    { label: "Clin. Readiness", value: "Phase 1", unit: "",  color: "#00E5A0" },
  ],
};

/* ─── 3D Body Silhouette (Stylized) ────────────── */
function BodySilhouette() {
  const groupRef = useRef<THREE.Group>(null);
  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.15;
    }
  });

  const skinColor = "#E8B4A0";
  const muscleColor = "#C97664";

  return (
    <group ref={groupRef}>
      {/* Head */}
      <mesh position={[0, 3.1, 0]}>
        <sphereGeometry args={[0.42, 32, 32]} />
        <meshStandardMaterial color={skinColor} roughness={0.5} metalness={0.1} />
      </mesh>
      {/* Neck */}
      <mesh position={[0, 2.55, 0]}>
        <cylinderGeometry args={[0.14, 0.16, 0.35, 16]} />
        <meshStandardMaterial color={skinColor} roughness={0.5} />
      </mesh>
      {/* Torso */}
      <mesh position={[0, 1.5, 0]}>
        <cylinderGeometry args={[0.55, 0.42, 1.8, 16]} />
        <meshStandardMaterial color={muscleColor} roughness={0.4} metalness={0.15} transparent opacity={0.85} />
      </mesh>
      {/* Pelvis */}
      <mesh position={[0, 0.4, 0]}>
        <cylinderGeometry args={[0.42, 0.35, 0.4, 16]} />
        <meshStandardMaterial color={muscleColor} roughness={0.4} transparent opacity={0.8} />
      </mesh>
      {/* Left Upper Arm */}
      <mesh position={[-0.75, 2.0, 0]} rotation={[0, 0, 0.2]}>
        <cylinderGeometry args={[0.1, 0.09, 0.8, 12]} />
        <meshStandardMaterial color={skinColor} roughness={0.5} />
      </mesh>
      {/* Left Lower Arm */}
      <mesh position={[-0.95, 1.35, 0]} rotation={[0, 0, 0.15]}>
        <cylinderGeometry args={[0.08, 0.07, 0.75, 12]} />
        <meshStandardMaterial color={skinColor} roughness={0.5} />
      </mesh>
      {/* Right Upper Arm */}
      <mesh position={[0.75, 2.0, 0]} rotation={[0, 0, -0.2]}>
        <cylinderGeometry args={[0.1, 0.09, 0.8, 12]} />
        <meshStandardMaterial color={skinColor} roughness={0.5} />
      </mesh>
      {/* Right Lower Arm */}
      <mesh position={[0.95, 1.35, 0]} rotation={[0, 0, -0.15]}>
        <cylinderGeometry args={[0.08, 0.07, 0.75, 12]} />
        <meshStandardMaterial color={skinColor} roughness={0.5} />
      </mesh>
      {/* Left Upper Leg */}
      <mesh position={[-0.2, -0.35, 0]}>
        <cylinderGeometry args={[0.14, 0.11, 1.1, 12]} />
        <meshStandardMaterial color={muscleColor} roughness={0.4} transparent opacity={0.8} />
      </mesh>
      {/* Left Lower Leg */}
      <mesh position={[-0.22, -1.3, 0]}>
        <cylinderGeometry args={[0.1, 0.07, 1.0, 12]} />
        <meshStandardMaterial color={skinColor} roughness={0.5} />
      </mesh>
      {/* Right Upper Leg */}
      <mesh position={[0.2, -0.35, 0]}>
        <cylinderGeometry args={[0.14, 0.11, 1.1, 12]} />
        <meshStandardMaterial color={muscleColor} roughness={0.4} transparent opacity={0.8} />
      </mesh>
      {/* Right Lower Leg */}
      <mesh position={[0.22, -1.3, 0]}>
        <cylinderGeometry args={[0.1, 0.07, 1.0, 12]} />
        <meshStandardMaterial color={skinColor} roughness={0.5} />
      </mesh>
    </group>
  );
}

/* ─── Organ Node Marker (3D) ───────────────────── */
function OrganMarker({
  node,
  active,
  onClick,
}: {
  node: typeof ORGAN_NODES[number];
  active: boolean;
  onClick: () => void;
}) {
  return (
    <Float speed={2} rotationIntensity={0} floatIntensity={0.3}>
      <mesh position={node.position}>
        <sphereGeometry args={[active ? 0.12 : 0.08, 16, 16]} />
        <meshStandardMaterial
          color={node.color}
          emissive={node.color}
          emissiveIntensity={active ? 0.8 : 0.3}
          transparent
          opacity={active ? 1 : 0.7}
        />
      </mesh>

      {/* Floating HTML label */}
      <Html position={[node.position[0] + 0.35, node.position[1] + 0.15, node.position[2]]} distanceFactor={6}>
        <button
          onClick={onClick}
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold whitespace-nowrap transition-all duration-300 border cursor-pointer select-none",
            active
              ? "bg-[#2B2B2B] text-white border-white/20 shadow-lg scale-110"
              : "bg-white/80 dark:bg-white/10 text-foreground border-black/10 dark:border-white/15 hover:scale-105"
          )}
          style={active ? { boxShadow: `0 4px 20px ${node.color}50` } : {}}
        >
          <span
            className="flex h-4 w-4 items-center justify-center rounded-full text-[8px] font-black text-white"
            style={{ background: node.color }}
          >
            {node.number}
          </span>
          {node.label}
        </button>
      </Html>
    </Float>
  );
}

/* ─── Full Scene ───────────────────────────────── */
function Scene({
  activeOrgan,
  onOrganClick,
}: {
  activeOrgan: OrganId;
  onOrganClick: (id: OrganId) => void;
}) {
  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 8, 5]} intensity={1.5} color="#ffffff" />
      <pointLight position={[-3, -2, 4]} intensity={0.8} color="#FFE4D6" />
      <pointLight position={[3, 4, -2]} intensity={0.6} color="#D6E4FF" />

      <BodySilhouette />

      {ORGAN_NODES.map((node) => (
        <OrganMarker
          key={node.id}
          node={node}
          active={activeOrgan === node.id}
          onClick={() => onOrganClick(node.id)}
        />
      ))}

      <OrbitControls
        enablePan={false}
        enableZoom={true}
        minDistance={4}
        maxDistance={12}
        autoRotate={false}
        maxPolarAngle={Math.PI * 0.85}
        minPolarAngle={Math.PI * 0.15}
      />
    </>
  );
}

/* ─── Main Component ───────────────────────────── */
interface BodyPartViewerProps {
  className?: string;
  title?: string;
}

export default function BodyPartViewer({ className = "", title }: BodyPartViewerProps) {
  const [activeOrgan, setActiveOrgan] = useState<OrganId>("heart");
  const [rotating, setRotating] = useState(true);
  const cfg = ORGAN_NODES.find((o) => o.id === activeOrgan)!;
  const stats = ORGAN_STATS[activeOrgan];

  return (
    <div
      className={cn(
        "rounded-[32px] overflow-hidden relative border border-border/50 bg-card dark:bg-gradient-to-br dark:from-[rgba(15,20,40,0.6)] dark:to-[rgba(8,12,28,0.8)] transition-colors duration-300",
        className
      )}
      style={{ minHeight: 520 }}
    >
      {/* Top accent */}
      <div
        className="absolute top-0 left-6 right-6 h-[2px] rounded-full z-10"
        style={{ background: `linear-gradient(90deg, transparent, ${cfg.color}, transparent)` }}
      />

      {/* Header */}
      <div className="absolute top-4 left-6 right-6 z-10 flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            3D Anatomy Viewer
          </p>
          <h2 className="text-base font-bold mt-0.5">
            {title ?? "Full-Body Overview"}
            <span className="ml-2 text-xs font-normal text-muted-foreground">· Drug Action Sites</span>
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setRotating((r) => !r)}
            className="h-8 w-8 flex items-center justify-center rounded-xl bg-muted/30 dark:bg-white/10 text-muted-foreground hover:text-foreground transition-all border border-border/30"
          >
            {rotating ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </button>
          <div
            className="h-8 px-3 flex items-center gap-1.5 rounded-xl text-xs font-semibold"
            style={{
              background: `${cfg.color}20`,
              color: cfg.color,
              border: `1px solid ${cfg.color}40`,
            }}
          >
            <span className="h-1.5 w-1.5 rounded-full animate-pulse" style={{ background: cfg.color }} />
            Live
          </div>
        </div>
      </div>

      {/* 3D Canvas */}
      <div style={{ height: 400 }}>
        <Canvas camera={{ position: [0, 1.2, 7], fov: 40 }} gl={{ antialias: true }} style={{ background: "transparent" }}>
          <Suspense fallback={null}>
            <Scene activeOrgan={activeOrgan} onOrganClick={(id) => setActiveOrgan(id)} />
          </Suspense>
        </Canvas>
      </div>

      {/* Bottom: Organ Picker + Stats */}
      <div className="px-5 pb-5 space-y-4">
        {/* Organ pills */}
        <div className="flex gap-2 flex-wrap">
          {ORGAN_NODES.map((o) => {
            const isActive = o.id === activeOrgan;
            return (
              <button
                key={o.id}
                onClick={() => setActiveOrgan(o.id)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all duration-300"
                style={
                  isActive
                    ? {
                        background: `${o.color}25`,
                        color: o.color,
                        border: `1px solid ${o.color}50`,
                        boxShadow: `0 4px 16px ${o.color}30`,
                      }
                    : {
                        background: "hsl(var(--muted) / 0.3)",
                        color: "hsl(var(--muted-foreground))",
                        border: "1px solid hsl(var(--border))",
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
                className="bg-muted/20 dark:bg-white/5 border border-border/40 rounded-2xl p-3 text-center"
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
