import { motion } from "framer-motion";
import { useState, useMemo, useCallback, useRef } from "react";
import { Play, Pause, RotateCcw, BarChart3 } from "lucide-react";
import { generateTrajectory, subsample, type MDTrajectory } from "@/lib/mdEngine";
import { cn } from "@/lib/utils";

function scoreColor(s: number) {
    if (s > 0.7) return "text-success";
    if (s > 0.45) return "text-warning";
    return "text-destructive";
}

export default function MDSimulationPanel() {
    const [trajectory, setTrajectory] = useState<MDTrajectory | null>(null);
    const [simulating, setSimulating] = useState(false);
    const [progress, setProgress] = useState(0);
    const frameRef = useRef<number | null>(null);

    const chartData = useMemo(() => (trajectory ? subsample(trajectory.frames) : []), [trajectory]);

    const runSimulation = useCallback(() => {
        setSimulating(true);
        setProgress(0);
        setTrajectory(null);

        const start = performance.now();
        const duration = 2500;

        const step = () => {
            const elapsed = performance.now() - start;
            const p = Math.min(elapsed / duration, 1);
            setProgress(p);
            if (p < 1) {
                frameRef.current = requestAnimationFrame(step);
            } else {
                setTrajectory(generateTrajectory(100, 0.5));
                setSimulating(false);
            }
        };
        frameRef.current = requestAnimationFrame(step);
    }, []);

    const reset = () => {
        if (frameRef.current) cancelAnimationFrame(frameRef.current);
        setTrajectory(null);
        setSimulating(false);
        setProgress(0);
    };

    // Chart dimensions
    const W = 500, H = 150, pad = 30;

    const rmsdPath = useMemo(() => {
        if (!chartData.length) return "";
        const maxT = chartData[chartData.length - 1].time;
        const maxR = Math.max(...chartData.map(d => d.rmsd));
        return chartData.map((d, i) => {
            const x = pad + (d.time / maxT) * (W - pad * 2);
            const y = H - pad - (d.rmsd / (maxR * 1.1)) * (H - pad * 2);
            return `${i === 0 ? "M" : "L"}${x},${y}`;
        }).join("");
    }, [chartData]);

    const energyPath = useMemo(() => {
        if (!chartData.length) return "";
        const maxT = chartData[chartData.length - 1].time;
        const minE = Math.min(...chartData.map(d => d.energy));
        const maxE = Math.max(...chartData.map(d => d.energy));
        const range = maxE - minE || 1;
        return chartData.map((d, i) => {
            const x = pad + (d.time / maxT) * (W - pad * 2);
            const y = H - pad - ((d.energy - minE) / (range * 1.1)) * (H - pad * 2);
            return `${i === 0 ? "M" : "L"}${x},${y}`;
        }).join("");
    }, [chartData]);

    return (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-card rounded-2xl p-5 relative overflow-hidden">
            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-primary" />
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">MD Simulation</h3>
                    {trajectory && <span className="text-xs text-muted-foreground">100 ps · NVT · 300 K</span>}
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={reset} className="p-1.5 rounded-lg glass-surface hover:ring-1 hover:ring-quantum/20 transition-all">
                        <RotateCcw className="h-3.5 w-3.5 text-muted-foreground" />
                    </button>
                    <button
                        onClick={runSimulation}
                        disabled={simulating}
                        className={cn("px-3 py-1.5 rounded-xl text-xs font-semibold transition-all", simulating ? "bg-muted/30 text-muted-foreground" : "bg-primary/10 text-primary ring-1 ring-primary/30 hover:bg-primary/20")}
                    >
                        {simulating ? "Simulating…" : trajectory ? "Re-run" : "Run MD"}
                    </button>
                </div>
            </div>

            {/* Simulation progress */}
            {simulating && (
                <div className="mb-4">
                    <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                        <span>Simulation progress</span>
                        <span className="font-mono">{(progress * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-muted/30 overflow-hidden ring-1 ring-white/5">
                        <motion.div className="h-full rounded-full bg-gradient-to-r from-primary to-quantum" style={{ width: `${progress * 100}%` }} />
                    </div>
                </div>
            )}

            {/* Charts */}
            {trajectory && (
                <div className="space-y-4">
                    {/* RMSD Chart */}
                    <div>
                        <p className="text-xs text-muted-foreground mb-1">RMSD (Å) vs Time (ps)</p>
                        <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg bg-background/20 ring-1 ring-white/5">
                            <defs>
                                <filter id="md-glow"><feGaussianBlur stdDeviation="2" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
                                <linearGradient id="rmsd-grad" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stopColor="hsl(187,79%,54%)" /><stop offset="100%" stopColor="hsl(217,91%,60%)" /></linearGradient>
                            </defs>
                            <text x={pad} y={12} fontSize="8" fill="hsl(215,20%,55%)">RMSD</text>
                            <motion.path initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.5 }} d={rmsdPath} fill="none" stroke="url(#rmsd-grad)" strokeWidth="1.5" filter="url(#md-glow)" />
                        </svg>
                    </div>

                    {/* Energy Chart */}
                    <div>
                        <p className="text-xs text-muted-foreground mb-1">Potential Energy (kJ/mol) vs Time (ps)</p>
                        <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg bg-background/20 ring-1 ring-white/5">
                            <defs><linearGradient id="energy-grad" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stopColor="hsl(142,71%,45%)" /><stop offset="100%" stopColor="hsl(38,92%,50%)" /></linearGradient></defs>
                            <text x={pad} y={12} fontSize="8" fill="hsl(215,20%,55%)">Energy</text>
                            <motion.path initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.5 }} d={energyPath} fill="none" stroke="url(#energy-grad)" strokeWidth="1.5" filter="url(#md-glow)" />
                        </svg>
                    </div>

                    {/* Metrics */}
                    <div className="grid grid-cols-4 gap-2">
                        <MetricCard label="Avg RMSD" value={`${trajectory.avgRMSD.toFixed(2)} Å`} />
                        <MetricCard label="Avg RMSF" value={`${trajectory.avgRMSF.toFixed(2)} Å`} />
                        <MetricCard label="Convergence" value={trajectory.convergence ? "Yes" : "No"} color={trajectory.convergence ? "text-success" : "text-warning"} />
                        <MetricCard label="Stability" value={`${(trajectory.stabilityScore * 100).toFixed(0)}%`} color={scoreColor(trajectory.stabilityScore)} />
                    </div>
                </div>
            )}

            {!trajectory && !simulating && (
                <div className="text-center py-8 text-xs text-muted-foreground">
                    <p>Run a short molecular dynamics simulation to evaluate</p>
                    <p>protein–ligand complex stability (RMSD, RMSF, energy).</p>
                </div>
            )}
        </motion.div>
    );
}

function MetricCard({ label, value, color }: { label: string; value: string; color?: string }) {
    return (
        <div className="glass-surface rounded-xl p-2.5 text-center">
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className={cn("font-mono font-semibold text-sm mt-0.5", color)}>{value}</p>
        </div>
    );
}
