import { motion } from "framer-motion";
import { Cpu, Zap } from "lucide-react";
import { useMemo } from "react";
import { cn } from "@/lib/utils";

/** VQE convergence point */
interface VQEIteration {
    iteration: number;
    energy: number; // Hartree
}

/** Generate realistic VQE convergence data */
function generateVQEConvergence(): VQEIteration[] {
    const data: VQEIteration[] = [];
    let energy = -74.5;
    for (let i = 0; i <= 60; i++) {
        energy += (Math.random() - 0.65) * 0.08 * Math.exp(-i * 0.04);
        data.push({ iteration: i, energy: Number(energy.toFixed(6)) });
    }
    return data;
}

const expectationValues = [
    { operator: "⟨H⟩", value: -75.6284, unit: "Ha", desc: "Ground state energy" },
    { operator: "⟨Z₁Z₂⟩", value: 0.4231, unit: "", desc: "Qubit–qubit correlation" },
    { operator: "⟨X₁⟩", value: -0.0042, unit: "", desc: "Single-qubit X expectation" },
    { operator: "⟨N⟩", value: 10.0, unit: "e⁻", desc: "Electron number" },
];

const circuitInfo = {
    qubits: 8,
    depth: 24,
    gates: 156,
    parameters: 32,
    optimizer: "COBYLA",
    ansatz: "UCCSD",
    backend: "Qiskit Aer (statevector)",
};

export default function QuantumChemPanel() {
    const convergence = useMemo(generateVQEConvergence, []);
    const groundState = convergence[convergence.length - 1].energy;

    // Chart SVG
    const W = 460, H = 140, pad = 35;
    const maxIter = convergence[convergence.length - 1].iteration;
    const energyValues = convergence.map(c => c.energy);
    const minE = Math.min(...energyValues);
    const maxE = Math.max(...energyValues);
    const rangeE = maxE - minE || 1;

    const path = convergence.map((d, i) => {
        const x = pad + (d.iteration / maxIter) * (W - pad * 2);
        const y = H - pad - ((d.energy - minE) / (rangeE * 1.1)) * (H - pad * 2);
        return `${i === 0 ? "M" : "L"}${x},${y}`;
    }).join("");

    return (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="liquid-glass rounded-2xl p-5 relative overflow-hidden">
            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Cpu className="h-4 w-4 text-primary" />
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Quantum Chemistry (VQE)</h3>
                </div>
                <span className="px-2 py-0.5 rounded-lg text-xs font-semibold bg-primary/10 text-primary ring-1 ring-primary/30">{circuitInfo.ansatz}</span>
            </div>

            {/* Ground state energy */}
            <div className="text-center mb-4">
                <p className="text-xs text-muted-foreground mb-1">VQE Ground State Energy</p>
                <motion.p initial={{ scale: 0.8 }} animate={{ scale: 1 }} className="text-3xl font-bold font-mono text-primary">
                    {groundState.toFixed(4)} <span className="text-sm text-muted-foreground">Hartree</span>
                </motion.p>
            </div>

            {/* Convergence chart */}
            <div className="mb-4">
                <p className="text-xs text-muted-foreground mb-1">Energy Convergence</p>
                <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg bg-background/20 ring-1 ring-white/5">
                    <defs>
                        <filter id="vqe-glow"><feGaussianBlur stdDeviation="2" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
                        <linearGradient id="vqe-grad" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stopColor="hsl(38,92%,50%)" /><stop offset="100%" stopColor="hsl(142,71%,45%)" /></linearGradient>
                    </defs>
                    <text x={pad} y={12} fontSize="8" fill="hsl(215,20%,55%)">E (Ha)</text>
                    <text x={W - 50} y={H - 5} fontSize="8" fill="hsl(215,20%,55%)">Iteration</text>
                    <motion.path initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 2 }} d={path} fill="none" stroke="url(#vqe-grad)" strokeWidth="1.5" filter="url(#vqe-glow)" />
                </svg>
            </div>

            {/* Expectation values */}
            <div className="mb-4">
                <p className="text-xs text-muted-foreground mb-2">Expectation Values</p>
                <div className="space-y-1.5">
                    {expectationValues.map((ev) => (
                        <div key={ev.operator} className="flex items-center justify-between glass-surface rounded-lg px-3 py-1.5">
                            <div className="flex items-center gap-2">
                                <span className="font-mono text-xs text-primary">{ev.operator}</span>
                                <span className="text-xs text-muted-foreground">{ev.desc}</span>
                            </div>
                            <span className="font-mono text-xs font-semibold">{ev.value.toFixed(4)} {ev.unit}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Circuit info */}
            <div className="grid grid-cols-4 gap-2">
                {[
                    { label: "Qubits", value: circuitInfo.qubits },
                    { label: "Depth", value: circuitInfo.depth },
                    { label: "Gates", value: circuitInfo.gates },
                    { label: "Params", value: circuitInfo.parameters },
                ].map((i) => (
                    <div key={i.label} className="glass-surface rounded-xl p-2 text-center">
                        <p className="text-xs text-muted-foreground">{i.label}</p>
                        <p className="font-mono font-semibold text-sm text-primary">{i.value}</p>
                    </div>
                ))}
            </div>
        </motion.div>
    );
}
