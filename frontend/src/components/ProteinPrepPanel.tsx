import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect, useCallback } from "react";
import { CheckCircle2, Loader2, Settings2 } from "lucide-react";
import { PREP_STEPS, type PrepStep } from "@/lib/mdEngine";
import { cn } from "@/lib/utils";

export default function ProteinPrepPanel() {
    const [running, setRunning] = useState(false);
    const [completed, setCompleted] = useState<string[]>([]);
    const [activeStep, setActiveStep] = useState<string | null>(null);

    const runPipeline = useCallback(async () => {
        setRunning(true);
        setCompleted([]);
        for (const step of PREP_STEPS) {
            setActiveStep(step.id);
            await new Promise((r) => setTimeout(r, step.duration));
            setCompleted((prev) => [...prev, step.id]);
        }
        setActiveStep(null);
        setRunning(false);
    }, []);

    const allDone = completed.length === PREP_STEPS.length;

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-card rounded-2xl p-5 relative overflow-hidden"
        >
            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />

            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Settings2 className="h-4 w-4 text-quantum" />
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Protein Preparation</h3>
                </div>
                <button
                    onClick={runPipeline}
                    disabled={running}
                    className={cn(
                        "px-4 py-1.5 rounded-xl text-xs font-semibold transition-all duration-300",
                        running
                            ? "bg-muted/30 text-muted-foreground cursor-wait"
                            : allDone
                                ? "bg-success/10 text-success ring-1 ring-success/30 hover:bg-success/20"
                                : "bg-quantum/10 text-quantum ring-1 ring-quantum/30 hover:bg-quantum/20"
                    )}
                >
                    {running ? "Processing…" : allDone ? "Re-run" : "Prepare"}
                </button>
            </div>

            <div className="space-y-2">
                {PREP_STEPS.map((step, i) => {
                    const done = completed.includes(step.id);
                    const active = activeStep === step.id;
                    return (
                        <motion.div
                            key={step.id}
                            initial={{ opacity: 0.5 }}
                            animate={{ opacity: done || active ? 1 : 0.5 }}
                            className={cn(
                                "flex items-center gap-3 rounded-xl px-3 py-2.5 transition-all duration-300",
                                active && "glass-surface ring-1 ring-quantum/20",
                                done && "glass-surface"
                            )}
                        >
                            <div className="flex-shrink-0 w-6 h-6 flex items-center justify-center">
                                {done ? (
                                    <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 400 }}>
                                        <CheckCircle2 className="h-5 w-5 text-success" />
                                    </motion.div>
                                ) : active ? (
                                    <Loader2 className="h-5 w-5 text-quantum animate-spin" />
                                ) : (
                                    <div className="h-5 w-5 rounded-full border border-white/10 flex items-center justify-center text-xs text-muted-foreground">{i + 1}</div>
                                )}
                            </div>
                            <div className="flex-1 min-w-0">
                                <p className={cn("text-sm font-medium", done ? "text-foreground" : "text-muted-foreground")}>{step.label}</p>
                                {(done || active) && (
                                    <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-xs text-muted-foreground mt-0.5">{step.detail}</motion.p>
                                )}
                            </div>
                        </motion.div>
                    );
                })}
            </div>

            {allDone && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-4 rounded-xl bg-success/5 ring-1 ring-success/20 p-3 text-center">
                    <p className="text-xs text-success font-semibold">✓ Protein prepared and ready for simulation</p>
                </motion.div>
            )}
        </motion.div>
    );
}
