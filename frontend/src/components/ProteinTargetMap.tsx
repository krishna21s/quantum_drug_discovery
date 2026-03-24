import { Activity } from "lucide-react";

interface DrugCategory {
  label: string;
  count: number;
  drugs: string[];
}

const categories: DrugCategory[] = [
  { label: "Approved", count: 3, drugs: ["Cetuximab", "Afatinib", "Osimertinib"] },
  { label: "Investigational", count: 5, drugs: ["IGN311", "Rindopepimut", "Matuzumab", "Canertinib", "Varlitinib"] },
  { label: "Experimental", count: 1, drugs: ["PD-168393"] },
  { label: "Other", count: 3, drugs: ["Gefitinib", "Lapatinib"] },
];

export default function ProteinTargetMap() {
  return (
    <div className="h-full space-y-5 flex flex-col p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center">
            <Activity className="h-3.5 w-3.5 text-foreground" />
          </div>
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Protein Target Map</h3>
        </div>
        <span className="px-2.5 py-1 rounded-lg text-[10px] font-semibold bg-muted/20 text-foreground border border-border/40">EGFR · 1M17</span>
      </div>

      <div className="space-y-3">
        {categories.map((cat) => (
          <div key={cat.label} className="bg-background border border-border/50 rounded-2xl p-4 transition-all duration-300">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-foreground">{cat.label}</span>
              <span className="text-xs font-mono font-bold bg-primary/10 text-foreground px-2 py-0.5 rounded-full">
                {cat.count} drugs
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {cat.drugs.map(drug => (
                <span key={drug} className="text-[11px] font-medium text-muted-foreground bg-muted/30 px-2 py-1 rounded-md border border-border/20">
                  {drug}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
