import { motion } from "framer-motion";
import { useState } from "react";
import { Shield, ChevronDown, ChevronUp } from "lucide-react";
import ADMETRadarChart from "./ADMETRadarChart";
import type { ADMETDetail } from "@/lib/admetEngine";
import { cn } from "@/lib/utils";

interface ADMETPanelProps {
    data: ADMETDetail;
    bindingAffinity?: number;
    quantumEnergy?: number;
    combinedScore?: number;
}

function scoreColor(s: number) {
    if (s > 0.7) return "text-success";
    if (s > 0.45) return "text-warning";
    return "text-destructive";
}
function scoreBg(s: number) {
    if (s > 0.7) return "bg-success/10 ring-success/30";
    if (s > 0.45) return "bg-warning/10 ring-warning/30";
    return "bg-destructive/10 ring-destructive/30";
}
function scoreGlow(s: number) {
    if (s > 0.7) return "shadow-[0_0_20px_-4px_hsl(142_71%_45%_/_0.3)]";
    if (s > 0.45) return "shadow-[0_0_20px_-4px_hsl(38_92%_50%_/_0.3)]";
    return "shadow-[0_0_20px_-4px_hsl(0_84%_60%_/_0.3)]";
}
function barColor(s: number) {
    if (s > 0.7) return "from-success to-success/60";
    if (s > 0.45) return "from-warning to-warning/60";
    return "from-destructive to-destructive/60";
}

interface ScoreBarProps {
    label: string;
    score: number;
    children?: React.ReactNode;
}

function ScoreSection({ label, score, children }: ScoreBarProps) {
    const [open, setOpen] = useState(false);
    return (
        <div className="space-y-1.5">
            <button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between text-xs group">
                <span className="font-medium group-hover:text-foreground transition-colors">{label}</span>
                <div className="flex items-center gap-2">
                    <span className={cn("font-mono font-semibold text-xs", scoreColor(score))}>{(score * 100).toFixed(0)}%</span>
                    {children && (open ? <ChevronUp className="h-3 w-3 text-muted-foreground" /> : <ChevronDown className="h-3 w-3 text-muted-foreground" />)}
                </div>
            </button>
            <div className="h-1.5 rounded-full bg-muted/40 overflow-hidden ring-1 ring-white/5">
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${score * 100}%` }}
                    transition={{ duration: 1, delay: 0.2, ease: "easeOut" }}
                    className={cn("h-full rounded-full bg-gradient-to-r", barColor(score))}
                />
            </div>
            {open && children && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="glass-surface rounded-xl p-3 text-xs space-y-1">
                    {children}
                </motion.div>
            )}
        </div>
    );
}

function DetailRow({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
    return (
        <div className="flex items-center justify-between">
            <span className="text-muted-foreground">{label}</span>
            <span className={cn("font-mono font-medium", warn ? "text-warning" : "")}>{value}</span>
        </div>
    );
}

export default function ADMETPanel({ data, bindingAffinity, quantumEnergy, combinedScore: cs }: ADMETPanelProps) {
    const { scores } = data;

    return (
        <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="glass-card rounded-2xl p-6 space-y-5 relative overflow-hidden"
        >
            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />

            <div className="flex items-center gap-2">
                <Shield className="h-4 w-4 text-quantum" />
                <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">ADMET Profile</h3>
            </div>

            {/* Radar */}
            <div className="flex justify-center">
                <ADMETRadarChart scores={scores} />
            </div>

            {/* Sub-scores */}
            <div className="space-y-3">
                <ScoreSection label="Absorption" score={scores.absorption}>
                    <DetailRow label="Lipinski violations" value={String(data.absorption.lipinskiViolations)} warn={data.absorption.lipinskiViolations > 1} />
                    <DetailRow label="Bioavailability" value={data.absorption.bioavailability} />
                    <DetailRow label="Solubility" value={data.absorption.solubilityClass} />
                    <DetailRow label="Intestinal absorption" value={data.absorption.intestinalAbsorption} />
                </ScoreSection>

                <ScoreSection label="Distribution" score={scores.distribution}>
                    <DetailRow label="BBB permeant" value={data.distribution.bbbPermeant ? "Yes" : "No"} />
                    <DetailRow label="Volume of distribution" value={data.distribution.vdCategory} />
                    <DetailRow label="Plasma protein binding" value={data.distribution.plasmaProteinBinding} />
                </ScoreSection>

                <ScoreSection label="Metabolism" score={scores.metabolism}>
                    <DetailRow label="CYP450 substrate" value={data.metabolism.cyp450Substrate ? "Yes" : "No"} warn={data.metabolism.cyp450Substrate} />
                    <DetailRow label="CYP450 inhibitor" value={data.metabolism.cyp450Inhibitor ? "Yes" : "No"} warn={data.metabolism.cyp450Inhibitor} />
                    <DetailRow label="Hepatic clearance" value={data.metabolism.hepaticClearance} />
                </ScoreSection>

                <ScoreSection label="Excretion" score={scores.excretion}>
                    <DetailRow label="Renal clearance" value={data.excretion.renalClearance} />
                    <DetailRow label="Half-life" value={data.excretion.estimatedHalfLife} />
                    <DetailRow label="Category" value={data.excretion.halfLifeCategory} />
                </ScoreSection>

                <ScoreSection label="Toxicity (safety)" score={scores.toxicity}>
                    <DetailRow label="hERG inhibition" value={data.toxicity.hergRisk} warn={data.toxicity.hergRisk !== "Low"} />
                    <DetailRow label="Ames mutagenicity" value={data.toxicity.amesMutagenicity} warn={data.toxicity.amesMutagenicity !== "Negative"} />
                    <DetailRow label="Hepatotoxicity" value={data.toxicity.hepatotoxicity} warn={data.toxicity.hepatotoxicity !== "Low"} />
                    <DetailRow label="Cardiotoxicity" value={data.toxicity.cardiotoxicity} warn={data.toxicity.cardiotoxicity !== "Low"} />
                </ScoreSection>
            </div>

            {/* Combined score */}
            {cs !== undefined && (
                <div className="glass-surface rounded-xl p-3 space-y-1">
                    <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Combined Score</span>
                        <span className={cn("font-mono font-bold text-sm", scoreColor(cs))}>{(cs * 100).toFixed(0)}%</span>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs text-center text-muted-foreground">
                        {bindingAffinity !== undefined && <div>Binding: <span className="font-mono text-foreground">{bindingAffinity}</span></div>}
                        {quantumEnergy !== undefined && <div>Quantum: <span className="font-mono text-foreground">{quantumEnergy}</span></div>}
                        <div>ADMET: <span className={cn("font-mono", scoreColor(scores.overall))}>{(scores.overall * 100).toFixed(0)}%</span></div>
                    </div>
                </div>
            )}

            {/* Verdict */}
            <div className={cn(
                "rounded-xl ring-1 p-4 text-center",
                scoreBg(scores.overall),
                scoreGlow(scores.overall)
            )}>
                <p className="text-xs text-muted-foreground mb-1">Drug-likeness Verdict</p>
                <p className={cn("text-lg font-bold", scoreColor(scores.overall))}>{scores.verdict}</p>
            </div>
        </motion.div>
    );
}
