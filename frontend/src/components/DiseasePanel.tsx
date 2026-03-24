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
    <div className="space-y-5 h-full flex flex-col p-4">
      {/* Protein Target Header */}
      <div className="flex items-center gap-3">
        <div className="w-12 h-12 rounded-xl bg-background border border-border/50 flex items-center justify-center">
          <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
            <span className="text-xs font-mono font-bold text-foreground">EGFR</span>
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
            <div key={d.rank} className="flex items-center justify-between rounded-xl bg-background border border-border/50 px-3 py-2.5 text-sm transition-all duration-300">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-foreground opacity-50 w-4 font-semibold">{d.rank}</span>
                <span className="text-foreground">{d.name}</span>
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
            <div key={t.rank} className="flex items-center justify-between rounded-xl bg-background border border-border/50 px-3 py-2 text-sm transition-all duration-300">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-foreground opacity-50 w-4 font-semibold">{t.rank}</span>
                <span className="font-mono text-xs text-foreground">{t.id}</span>
              </div>
              <span className="text-xs font-mono text-muted-foreground">{t.match}</span>
            </div>
          ))}
        </div>
        <button className="mt-4 flex items-center gap-1 text-xs text-primary hover:underline font-medium transition-colors">
          See more <ArrowRight className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}
