import { Cpu } from "lucide-react";
import { useMemo } from "react";

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
    { operator: "⟨H⟩", value: -75.6284, unit: "Ha", desc: "Ground energy" },
    { operator: "⟨Z₁Z₂⟩", value: 0.4231, unit: "", desc: "Correlation" },
    { operator: "⟨X₁⟩", value: -0.0042, unit: "", desc: "Single-qubit X" },
    { operator: "⟨N⟩", value: 10.0, unit: "e⁻", desc: "Elec. number" },
];

const circuitInfo = {
    qubits: 8,
    depth: 24,
    gates: 156,
    parameters: 32,
    optimizer: "COBYLA",
    ansatz: "UCCSD",
    backend: "Qiskit Aer",
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
        <div className="h-full flex flex-col p-2">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center">
                        <Cpu className="h-3.5 w-3.5 text-foreground" />
                    </div>
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Quantum Chem</h3>
                </div>
                <span className="px-2.5 py-1 rounded-lg text-[10px] font-semibold bg-muted/20 text-foreground border border-border/40">{circuitInfo.ansatz}</span>
            </div>

            {/* Ground state energy */}
            <div className="text-center mb-5">
                <p className="text-xs text-muted-foreground mb-1">VQE Ground State</p>
                <p className="text-3xl font-bold font-mono text-foreground">
                    {groundState.toFixed(4)} <span className="text-sm text-muted-foreground">Ha</span>
                </p>
            </div>

            {/* Convergence chart minified */}
            <div className="mb-5">
                <div className="flex items-center justify-between mb-1.5">
                    <p className="text-[11px] text-muted-foreground">Energy Convergence</p>
                </div>
                <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-xl bg-background border border-border/50">
                    <text x={pad} y={15} fontSize="10" fill="hsl(var(--muted-foreground))">E (Ha)</text>
                    <text x={W - 55} y={H - 8} fontSize="10" fill="hsl(var(--muted-foreground))">Iter</text>
                    <path 
                        d={path} 
                        fill="none" 
                        stroke="hsl(var(--foreground))"
                        strokeWidth="2" 
                    />
                </svg>
            </div>

            {/* Expectation values */}
            <div className="mb-5">
                <p className="text-[11px] text-muted-foreground mb-1.5">Expectation Values</p>
                <div className="grid grid-cols-2 gap-2">
                    {expectationValues.map((ev) => (
                        <div key={ev.operator} className="bg-background border border-border/50 rounded-xl px-3 py-2 text-center">
                            <p className="font-mono text-[10px] text-muted-foreground mb-0.5">{ev.operator}</p>
                            <p className="font-mono text-xs font-bold text-foreground">{ev.value.toFixed(4)}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Circuit info */}
            <div className="grid grid-cols-4 gap-2 border-t border-border/30 pt-4 mt-auto">
                {[
                    { label: "Qubits", value: circuitInfo.qubits },
                    { label: "Depth", value: circuitInfo.depth },
                    { label: "Gates", value: circuitInfo.gates },
                    { label: "Params", value: circuitInfo.parameters },
                ].map((i) => (
                    <div key={i.label} className="text-center">
                        <p className="font-mono font-bold text-sm text-foreground">{i.value}</p>
                        <p className="text-[10px] text-muted-foreground mt-0.5">{i.label}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}
