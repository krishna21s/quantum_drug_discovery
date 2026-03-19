import { motion } from "framer-motion";
import { Flame, TrendingDown, Info } from "lucide-react";
import { useMemo } from "react";
import { estimateFreeEnergy, type FreeEnergyResult } from "@/lib/freeEnergyEngine";
import { cn } from "@/lib/utils";

interface FreeEnergyPanelProps {
    result?: FreeEnergyResult;
}

function confidenceColor(c: FreeEnergyResult["confidence"]) {
    if (c === "High") return "text-success bg-success/10 ring-success/30";
    if (c === "Moderate") return "text-warning bg-warning/10 ring-warning/30";
    return "text-destructive bg-destructive/10 ring-destructive/30";
}

export default function FreeEnergyPanel({ result }: FreeEnergyPanelProps) {
    const data = useMemo(() => result ?? estimateFreeEnergy(-8.2, 180.16, 1.24, 63.6, 1, 4), [result]);

    const decomp = data.decomposition;
    const bars = [
        { label: "van der Waals", value: decomp.vanDerWaals, color: "bg-primary" },
        { label: "Electrostatic", value: decomp.electrostatic, color: "bg-quantum" },
        { label: "Polar Solvation", value: decomp.polarSolvation, color: "bg-warning" },
        { label: "Non-polar Solv.", value: decomp.nonPolarSolvation, color: "bg-success" },
        { label: "−TΔS (Entropy)", value: decomp.entropy, color: "bg-destructive" },
    ];

    const maxAbs = Math.max(...bars.map(b => Math.abs(b.value)));

    return (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="glass-card rounded-2xl p-5 relative overflow-hidden">
            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />

            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Flame className="h-4 w-4 text-quantum" />
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Free Energy Estimation</h3>
                </div>
                <span className={cn("px-2 py-0.5 rounded-lg text-xs font-semibold ring-1", confidenceColor(data.confidence))}>
                    {data.confidence} Confidence
                </span>
            </div>

            {/* ΔG display */}
            <div className="text-center mb-5">
                <p className="text-xs text-muted-foreground mb-1">Estimated Binding Free Energy (ΔG)</p>
                <motion.p
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="text-3xl font-bold font-mono text-quantum"
                >
                    {data.deltaG.toFixed(2)} <span className="text-sm text-muted-foreground">kcal/mol</span>
                </motion.p>
            </div>

            {/* Decomposition bars */}
            <div className="space-y-2.5 mb-4">
                {bars.map((bar, i) => {
                    const pct = (Math.abs(bar.value) / maxAbs) * 100;
                    const isNeg = bar.value < 0;
                    return (
                        <motion.div key={bar.label} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 + i * 0.05 }}>
                            <div className="flex items-center justify-between text-xs mb-0.5">
                                <span className="text-muted-foreground">{bar.label}</span>
                                <span className={cn("font-mono font-semibold", isNeg ? "text-success" : "text-warning")}>{bar.value.toFixed(2)}</span>
                            </div>
                            <div className="h-2 rounded-full bg-muted/20 overflow-hidden ring-1 ring-white/5">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${pct}%` }}
                                    transition={{ duration: 0.8, delay: 0.1 + i * 0.05 }}
                                    className={cn("h-full rounded-full", bar.color, isNeg ? "opacity-80" : "opacity-50")}
                                />
                            </div>
                        </motion.div>
                    );
                })}
            </div>

            {/* Summary row */}
            <div className="grid grid-cols-2 gap-3">
                <div className="glass-surface rounded-xl p-2.5 text-center">
                    <p className="text-xs text-muted-foreground">Docking</p>
                    <p className="font-mono font-semibold text-sm text-primary">{data.dockingContribution.toFixed(2)}</p>
                </div>
                <div className="glass-surface rounded-xl p-2.5 text-center">
                    <p className="text-xs text-muted-foreground">Solvation</p>
                    <p className="font-mono font-semibold text-sm text-quantum">{data.solvationContribution.toFixed(2)}</p>
                </div>
            </div>
        </motion.div>
    );
}
