import { motion } from "framer-motion";

interface QuantumOutputProps {
  groundStateEnergy: number;
  vqcPrediction: string;
  vqcProbability: number;
  bindingAffinity: number;
  drugLikeness: number;
  verdict: "effective" | "moderate" | "weak";
}

const verdictStyles = {
  effective: { bg: "bg-success/10 ring-success/30", text: "text-success", label: "Effective Candidate", glow: "shadow-[0_0_20px_-4px_hsl(142_71%_45%_/_0.3)]" },
  moderate: { bg: "bg-warning/10 ring-warning/30", text: "text-warning", label: "Moderate Candidate", glow: "shadow-[0_0_20px_-4px_hsl(38_92%_50%_/_0.3)]" },
  weak: { bg: "bg-destructive/10 ring-destructive/30", text: "text-destructive", label: "Weak Candidate", glow: "shadow-[0_0_20px_-4px_hsl(0_84%_60%_/_0.3)]" },
};

export default function QuantumOutputPanel({
  groundStateEnergy = -75.3,
  vqcPrediction = "Active Drug",
  vqcProbability = 0.87,
  bindingAffinity = -9.4,
  drugLikeness = 0.82,
  verdict = "effective",
}: Partial<QuantumOutputProps>) {
  const v = verdictStyles[verdict];

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
      className="glass-card rounded-2xl p-6 space-y-5 relative overflow-hidden"
    >
      {/* Top glow line */}
      <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />

      <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Quantum Output</h3>

      {/* Ground state */}
      <div className="rounded-xl glass-surface p-4 font-mono text-sm space-y-1">
        <p className="text-muted-foreground text-xs">Ground State Energy</p>
        <p className="text-2xl font-bold text-quantum neon-text">{groundStateEnergy} <span className="text-sm text-muted-foreground">Hartree</span></p>
      </div>

      {/* VQC */}
      <div className="rounded-xl glass-surface p-4 font-mono text-sm space-y-1">
        <p className="text-muted-foreground text-xs">VQC Prediction</p>
        <p className="text-lg font-bold">{vqcPrediction}</p>
        <div className="flex items-center gap-2 mt-2">
          <div className="flex-1 h-2.5 rounded-full bg-muted/50 overflow-hidden ring-1 ring-white/5">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${vqcProbability * 100}%` }}
              transition={{ duration: 1.2, delay: 0.3, ease: "easeOut" }}
              className="h-full rounded-full bg-gradient-to-r from-primary to-quantum shadow-[0_0_12px_hsl(187_79%_54%_/_0.4)]"
            />
          </div>
          <span className="text-xs text-quantum font-semibold">{(vqcProbability * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Classical outputs */}
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-xl glass-surface p-3 text-center">
          <p className="text-xs text-muted-foreground">Binding Affinity</p>
          <p className="text-lg font-bold font-mono mt-0.5">{bindingAffinity}</p>
          <p className="text-xs text-muted-foreground">kcal/mol</p>
        </div>
        <div className="rounded-xl glass-surface p-3 text-center">
          <p className="text-xs text-muted-foreground">Drug-likeness</p>
          <p className="text-lg font-bold font-mono mt-0.5">{drugLikeness}</p>
          <p className="text-xs text-muted-foreground">QED score</p>
        </div>
      </div>

      {/* Verdict */}
      <div className={`rounded-xl ${v.bg} ring-1 p-4 text-center ${v.glow}`}>
        <p className="text-xs text-muted-foreground mb-1">Final Verdict</p>
        <p className={`text-lg font-bold ${v.text}`}>{v.label}</p>
      </div>
    </motion.div>
  );
}
