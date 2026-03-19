import { useRef, useState, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Float, Text, Sphere, Line } from "@react-three/drei";
import { motion } from "framer-motion";
import { Play, Pause, RotateCcw } from "lucide-react";
import * as THREE from "three";

/* ───── Drug molecule (small cluster of spheres) ───── */
function DrugMolecule({ progress, playing }: { progress: number; playing: boolean }) {
    const ref = useRef<THREE.Group>(null!);

    // Approach trajectory: start at x=-6, end at x=0.8
    const startX = -6;
    const endX = 0.8;
    const eased = 1 - Math.pow(1 - Math.min(progress, 1), 3);
    const x = startX + (endX - startX) * eased;
    const wobbleY = Math.sin(progress * 8) * (1 - eased) * 0.4;
    const wobbleZ = Math.cos(progress * 6) * (1 - eased) * 0.3;

    useFrame((_, delta) => {
        if (ref.current && playing) {
            ref.current.rotation.y += delta * 1.5;
        }
    });

    return (
        <group ref={ref} position={[x, wobbleY, wobbleZ]}>
            {/* Core atom */}
            <Sphere args={[0.35, 32, 32]}>
                <meshStandardMaterial color="#39d5e6" emissive="#39d5e6" emissiveIntensity={0.4} roughness={0.2} metalness={0.6} />
            </Sphere>
            {/* Surrounding atoms */}
            {[0, 1, 2, 3].map((i) => {
                const angle = (Math.PI * 2 * i) / 4;
                return (
                    <Sphere key={i} args={[0.18, 16, 16]} position={[Math.cos(angle) * 0.65, Math.sin(angle) * 0.65, 0]}>
                        <meshStandardMaterial color="#4d8ef7" emissive="#4d8ef7" emissiveIntensity={0.3} roughness={0.3} />
                    </Sphere>
                );
            })}
            {/* Bond lines from center to surrounding */}
            {[0, 1, 2, 3].map((i) => {
                const angle = (Math.PI * 2 * i) / 4;
                return (
                    <Line
                        key={`bond-${i}`}
                        points={[[0, 0, 0], [Math.cos(angle) * 0.65, Math.sin(angle) * 0.65, 0]]}
                        color="#39d5e680"
                        lineWidth={1.5}
                    />
                );
            })}
        </group>
    );
}

/* ───── Protein cavity (large wireframe + surface) ───── */
function ProteinCavity({ progress }: { progress: number }) {
    const ref = useRef<THREE.Group>(null!);

    useFrame((_, delta) => {
        if (ref.current) {
            ref.current.rotation.y += delta * 0.15;
        }
    });

    // Glow intensifies when binding
    const glowIntensity = progress > 0.7 ? (progress - 0.7) / 0.3 : 0;

    return (
        <group ref={ref} position={[2.5, 0, 0]}>
            {/* Outer surface */}
            <Sphere args={[2, 32, 32]}>
                <meshStandardMaterial
                    color="#1a2340"
                    transparent
                    opacity={0.15}
                    wireframe
                    side={THREE.DoubleSide}
                />
            </Sphere>
            {/* Inner cavity */}
            <Sphere args={[1.3, 32, 32]}>
                <meshStandardMaterial
                    color="#4d8ef7"
                    transparent
                    opacity={0.08 + glowIntensity * 0.12}
                    side={THREE.BackSide}
                />
            </Sphere>
            {/* Active site marker */}
            <Sphere args={[0.5, 16, 16]} position={[-1.2, 0, 0]}>
                <meshStandardMaterial
                    color="#39d5e6"
                    emissive="#39d5e6"
                    emissiveIntensity={0.2 + glowIntensity * 0.6}
                    transparent
                    opacity={0.3 + glowIntensity * 0.4}
                />
            </Sphere>
            {/* Active site residue labels */}
            {progress > 0.8 && (
                <>
                    <Text position={[-1.2, 0.8, 0]} fontSize={0.15} color="#39d5e6">
                        ASP-145
                    </Text>
                    <Text position={[-0.5, -0.7, 0.5]} fontSize={0.15} color="#4d8ef7">
                        HIS-41
                    </Text>
                    <Text position={[-1.5, -0.3, -0.5]} fontSize={0.15} color="#4d8ef7">
                        GLU-166
                    </Text>
                </>
            )}
        </group>
    );
}

/* ───── Interaction lines (H-bonds, pi-stacking) ───── */
function InteractionLines({ progress }: { progress: number }) {
    if (progress < 0.85) return null;

    const opacity = (progress - 0.85) / 0.15;

    const bonds = [
        { from: [0.8, 0.2, 0] as [number, number, number], to: [1.3, 0.5, 0.3] as [number, number, number], label: "H-bond" },
        { from: [0.8, -0.2, 0] as [number, number, number], to: [1.3, -0.4, -0.2] as [number, number, number], label: "H-bond" },
        { from: [0.8, 0, 0.2] as [number, number, number], to: [1.5, 0, 0.5] as [number, number, number], label: "π-stack" },
    ];

    return (
        <group>
            {bonds.map((bond, i) => (
                <group key={i}>
                    <Line
                        points={[bond.from, bond.to]}
                        color={bond.label === "H-bond" ? "#39d5e6" : "#a855f7"}
                        lineWidth={1.5}
                        dashed
                        dashSize={0.1}
                        gapSize={0.05}
                        transparent
                        opacity={opacity}
                    />
                    <Text
                        position={[
                            (bond.from[0] + bond.to[0]) / 2,
                            (bond.from[1] + bond.to[1]) / 2 + 0.15,
                            (bond.from[2] + bond.to[2]) / 2,
                        ]}
                        fontSize={0.1}
                        color={bond.label === "H-bond" ? "#39d5e6" : "#a855f7"}
                    >
                        {bond.label}
                    </Text>
                </group>
            ))}
        </group>
    );
}

/* ───── Ambient particles ───── */
function AmbientParticles() {
    const count = 100;
    const positions = useMemo(() => {
        const arr = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            arr[i * 3] = (Math.random() - 0.5) * 20;
            arr[i * 3 + 1] = (Math.random() - 0.5) * 12;
            arr[i * 3 + 2] = (Math.random() - 0.5) * 12;
        }
        return arr;
    }, []);

    const ref = useRef<THREE.Points>(null!);

    useFrame((_, delta) => {
        if (ref.current) {
            ref.current.rotation.y += delta * 0.02;
        }
    });

    return (
        <points ref={ref}>
            <bufferGeometry>
                <bufferAttribute attach="attributes-position" args={[positions, 3]} />
            </bufferGeometry>
            <pointsMaterial color="#39d5e6" size={0.04} transparent opacity={0.4} sizeAttenuation />
        </points>
    );
}

/* ───── Scene orchestrator ───── */
function Scene({ progress, playing }: { progress: number; playing: boolean }) {
    return (
        <>
            <ambientLight intensity={0.3} />
            <directionalLight position={[5, 5, 5]} intensity={0.8} color="#ffffff" />
            <pointLight position={[-3, 2, 4]} intensity={0.6} color="#39d5e6" />
            <pointLight position={[3, -2, -4]} intensity={0.4} color="#4d8ef7" />

            <AmbientParticles />
            <DrugMolecule progress={progress} playing={playing} />
            <ProteinCavity progress={progress} />
            <InteractionLines progress={progress} />

            {/* Status text */}
            {progress > 0.95 && (
                <Float speed={2} floatIntensity={0.3}>
                    <Text position={[0, 3, 0]} fontSize={0.25} color="#39d5e6" anchorX="center">
                        BINDING COMPLETE
                    </Text>
                </Float>
            )}

            <OrbitControls enablePan enableZoom enableRotate autoRotate={!playing} autoRotateSpeed={0.3} />
        </>
    );
}

/* ───── Main component ───── */
export default function BindingCinematic() {
    const [playing, setPlaying] = useState(false);
    const [progress, setProgress] = useState(0);
    const [speed, setSpeed] = useState(1);

    // Use animation frame to advance progress
    const lastTimeRef = useRef<number | null>(null);

    const animate = () => {
        if (!playing) return;
        const now = performance.now();
        if (lastTimeRef.current !== null) {
            const dt = (now - lastTimeRef.current) / 1000;
            setProgress((p) => {
                const next = p + dt * 0.15 * speed;
                if (next >= 1.1) {
                    setPlaying(false);
                    return 1.1;
                }
                return next;
            });
        }
        lastTimeRef.current = now;
        requestAnimationFrame(animate);
    };

    const handlePlay = () => {
        if (progress >= 1.1) {
            setProgress(0);
        }
        setPlaying(true);
        lastTimeRef.current = null;
        requestAnimationFrame(animate);
    };

    const handlePause = () => {
        setPlaying(false);
        lastTimeRef.current = null;
    };

    const handleReset = () => {
        setPlaying(false);
        setProgress(0);
        lastTimeRef.current = null;
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="glass-card rounded-2xl p-5 relative overflow-hidden flex flex-col"
        >
            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />

            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Binding Cinematic</h3>
                <div className="flex items-center gap-2">
                    {/* Speed slider */}
                    <label className="text-xs text-muted-foreground">Speed</label>
                    <input
                        type="range"
                        min={0.25}
                        max={3}
                        step={0.25}
                        value={speed}
                        onChange={(e) => setSpeed(Number(e.target.value))}
                        className="w-16 accent-quantum h-1"
                    />
                    <span className="text-xs font-mono text-quantum w-6">{speed}×</span>
                </div>
            </div>

            {/* 3D Canvas */}
            <div className="relative flex-1 min-h-[320px] rounded-xl bg-background/30 ring-1 ring-white/5 overflow-hidden">
                <Canvas camera={{ position: [0, 1, 8], fov: 50 }} dpr={[1, 2]}>
                    <Scene progress={progress} playing={playing} />
                </Canvas>

                {/* Progress bar overlay */}
                <div className="absolute bottom-3 left-3 right-3 h-1.5 rounded-full bg-muted/30 overflow-hidden ring-1 ring-white/5">
                    <div
                        className="h-full rounded-full bg-gradient-to-r from-primary to-quantum transition-all duration-100"
                        style={{ width: `${Math.min(progress / 1.1, 1) * 100}%` }}
                    />
                </div>
            </div>

            {/* Controls */}
            <div className="mt-3 flex items-center justify-center gap-3">
                <button onClick={handleReset} className="p-2 rounded-xl glass-surface hover:ring-1 hover:ring-quantum/20 transition-all">
                    <RotateCcw className="h-4 w-4 text-muted-foreground" />
                </button>
                <button
                    onClick={playing ? handlePause : handlePlay}
                    className="p-3 rounded-xl bg-quantum/10 ring-1 ring-quantum/30 hover:bg-quantum/20 transition-all glow-cyan"
                >
                    {playing ? <Pause className="h-5 w-5 text-quantum" /> : <Play className="h-5 w-5 text-quantum" />}
                </button>

                {/* Phase indicator */}
                <div className="flex items-center gap-1.5 text-xs ml-2">
                    <PhaseIndicator label="Approach" active={progress > 0 && progress < 0.7} done={progress >= 0.7} />
                    <PhaseIndicator label="Snap" active={progress >= 0.7 && progress < 0.85} done={progress >= 0.85} />
                    <PhaseIndicator label="Interact" active={progress >= 0.85} done={progress >= 1.05} />
                </div>
            </div>
        </motion.div>
    );
}

function PhaseIndicator({ label, active, done }: { label: string; active: boolean; done: boolean }) {
    return (
        <span className={`px-2 py-0.5 rounded-lg text-xs font-medium transition-all ${done ? "bg-quantum/15 text-quantum" : active ? "bg-primary/15 text-primary animate-pulse" : "text-muted-foreground"
            }`}>
            {label}
        </span>
    );
}
