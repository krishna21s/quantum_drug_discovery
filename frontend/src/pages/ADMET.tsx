import AppLayout from "@/components/AppLayout";
import ADMETPanel from "@/components/ADMETPanel";
import { motion } from "framer-motion";
import { Shield, ChevronRight } from "lucide-react";
import { useState, useMemo } from "react";
import { computeADMET, combinedScore, DEMO_MOLECULES } from "@/lib/admetEngine";
import { cn } from "@/lib/utils";

const moleculeNames = Object.keys(DEMO_MOLECULES);

const bindingData: Record<string, number> = {
    Aspirin: -8.2,
    Cetuximab: -9.4,
    Ibuprofen: -6.8,
    Metformin: -5.2,
    Paracetamol: -7.1,
};

const quantumData: Record<string, number> = {
    Aspirin: -75.3,
    Cetuximab: -82.1,
    Ibuprofen: -68.5,
    Metformin: -55.2,
    Paracetamol: -71.8,
};

export default function ADMET() {
    const [selected, setSelected] = useState(moleculeNames[0]);

    const admet = useMemo(() => computeADMET(DEMO_MOLECULES[selected]), [selected]);
    const cs = useMemo(
        () => combinedScore(bindingData[selected], quantumData[selected], admet.scores.overall),
        [selected, admet.scores.overall]
    );

    // Comparison table data
    const allResults = useMemo(
        () =>
            moleculeNames.map((name) => {
                const d = computeADMET(DEMO_MOLECULES[name]);
                return { name, scores: d.scores, cs: combinedScore(bindingData[name], quantumData[name], d.scores.overall) };
            }),
        []
    );

    return (
        <AppLayout>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-8 space-y-6">
                <div>
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <Shield className="h-6 w-6 text-quantum" />
                        ADMET Analysis
                    </h1>
                    <p className="text-muted-foreground mt-1">
                        Absorption, Distribution, Metabolism, Excretion & Toxicity profiling
                    </p>
                </div>

                <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
                    {/* Left: Molecule selector + comparison table */}
                    <div className="lg:col-span-2 space-y-4">
                        {/* Molecule selector */}
                        <div className="glass-card rounded-2xl p-4 relative overflow-hidden">
                            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />
                            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Select Molecule</h3>
                            <div className="flex flex-wrap gap-2">
                                {moleculeNames.map((name) => (
                                    <button
                                        key={name}
                                        onClick={() => setSelected(name)}
                                        className={cn(
                                            "relative rounded-xl px-4 py-2 text-sm font-medium transition-all duration-300",
                                            selected === name
                                                ? "text-quantum"
                                                : "text-muted-foreground hover:text-foreground"
                                        )}
                                    >
                                        {selected === name && (
                                            <motion.div
                                                layoutId="admet-mol-select"
                                                className="absolute inset-0 rounded-xl glass-surface ring-1 ring-quantum/30"
                                                transition={{ type: "spring", stiffness: 350, damping: 30 }}
                                            />
                                        )}
                                        <span className="relative z-10">{name}</span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Comparison table */}
                        <div className="glass-card rounded-2xl overflow-hidden relative">
                            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />
                            <div className="p-4 border-b border-white/5">
                                <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Candidate Comparison</h3>
                            </div>
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b border-white/5">
                                        <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">Molecule</th>
                                        <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">A</th>
                                        <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">D</th>
                                        <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">M</th>
                                        <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">E</th>
                                        <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">T</th>
                                        <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">Overall</th>
                                        <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">Verdict</th>
                                        <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">Combined</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {allResults.map((r, i) => (
                                        <motion.tr
                                            key={r.name}
                                            initial={{ opacity: 0, x: -8 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: i * 0.05 }}
                                            onClick={() => setSelected(r.name)}
                                            className={cn(
                                                "border-b border-white/3 cursor-pointer transition-colors",
                                                selected === r.name ? "bg-quantum/5" : "hover:bg-muted/20"
                                            )}
                                        >
                                            <td className="px-4 py-3 font-medium">{r.name}</td>
                                            <ScoreCell score={r.scores.absorption} />
                                            <ScoreCell score={r.scores.distribution} />
                                            <ScoreCell score={r.scores.metabolism} />
                                            <ScoreCell score={r.scores.excretion} />
                                            <ScoreCell score={r.scores.toxicity} />
                                            <ScoreCell score={r.scores.overall} bold />
                                            <td className="px-4 py-3 text-center">
                                                <span className={cn(
                                                    "inline-block rounded-full px-2 py-0.5 text-xs font-semibold ring-1",
                                                    r.scores.verdict === "Pass" ? "bg-success/10 text-success ring-success/30" :
                                                        r.scores.verdict === "Caution" ? "bg-warning/10 text-warning ring-warning/30" :
                                                            "bg-destructive/10 text-destructive ring-destructive/30"
                                                )}>
                                                    {r.scores.verdict}
                                                </span>
                                            </td>
                                            <ScoreCell score={r.cs} bold />
                                        </motion.tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* Descriptors */}
                        <div className="glass-card rounded-2xl p-4 relative overflow-hidden">
                            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />
                            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                                Molecular Descriptors — {selected}
                            </h3>
                            <div className="grid grid-cols-4 gap-3">
                                {Object.entries(DEMO_MOLECULES[selected]).map(([key, val]) => (
                                    <div key={key} className="glass-surface rounded-xl p-3 text-center">
                                        <p className="text-xs text-muted-foreground capitalize">{key}</p>
                                        <p className="font-mono font-semibold text-sm mt-0.5">{typeof val === "number" ? val.toFixed(2) : val}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Right: ADMET Panel */}
                    <div>
                        <ADMETPanel
                            data={admet}
                            bindingAffinity={bindingData[selected]}
                            quantumEnergy={quantumData[selected]}
                            combinedScore={cs}
                        />
                    </div>
                </div>
            </motion.div>
        </AppLayout>
    );
}

function ScoreCell({ score, bold }: { score: number; bold?: boolean }) {
    const color = score > 0.7 ? "text-success" : score > 0.45 ? "text-warning" : "text-destructive";
    return (
        <td className={cn("px-4 py-3 text-center font-mono text-xs", color, bold && "font-bold")}>
            {(score * 100).toFixed(0)}%
        </td>
    );
}
