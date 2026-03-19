import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";

const diseases = [
  { rank: 1, name: "Lung Cancer", confidence: "100%" },
  { rank: 2, name: "NSBD2", confidence: "100%" },
];

const clinicalTrials = [
  { rank: 1, id: "NCT01788163", match: "100%" },
  { rank: 2, id: "NCT04261725", match: "100%" },
  { rank: 3, id: "NCT04552613", match: "100%" },
  { rank: 4, id: "NCT01523340", match: "100%" },
  { rank: 5, id: "NCT03171636", match: "100%" },
];

export default function DiseasePanel() {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
      className="glass-card rounded-2xl p-5 space-y-5 relative overflow-hidden"
    >
      <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

      {/* Protein Target Header */}
      <div className="flex items-center gap-3">
        <div className="w-12 h-12 rounded-xl glass-surface flex items-center justify-center overflow-hidden">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-quantum/20 to-primary/20 flex items-center justify-center ring-1 ring-quantum/20">
            <span className="text-xs font-mono font-bold text-quantum neon-text">EGFR</span>
          </div>
        </div>
        <div>
          <h3 className="font-semibold">EGFR</h3>
          <p className="text-xs text-muted-foreground">Epidermal Growth Factor Receptor</p>
        </div>
      </div>

      {/* Associated Diseases */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1">
            <span className="text-primary font-bold text-sm">{diseases.length}</span> Associated Diseases
          </h4>
        </div>
        <div className="space-y-2">
          {diseases.map((d) => (
            <div key={d.rank} className="flex items-center justify-between rounded-xl glass-surface px-3 py-2.5 text-sm hover:glow-cyan transition-all duration-300">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-quantum w-4 font-semibold">{d.rank}</span>
                <span>{d.name}</span>
              </div>
              <span className="text-xs font-mono text-muted-foreground">{d.confidence}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Clinical Trials */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1">
            <span className="text-primary font-bold text-sm">12</span> Completed Clinical Trials
          </h4>
        </div>
        <div className="space-y-1.5">
          {clinicalTrials.map((t) => (
            <div key={t.rank} className="flex items-center justify-between rounded-xl glass-surface px-3 py-2 text-sm hover:ring-1 hover:ring-quantum/10 transition-all duration-300">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-quantum w-4 font-semibold">{t.rank}</span>
                <span className="font-mono text-xs">{t.id}</span>
              </div>
              <span className="text-xs font-mono text-muted-foreground">{t.match}</span>
            </div>
          ))}
        </div>
        <button className="mt-3 flex items-center gap-1 text-xs text-quantum hover:underline font-medium transition-colors">
          See more <ArrowRight className="h-3 w-3" />
        </button>
      </div>
    </motion.div>
  );
}
