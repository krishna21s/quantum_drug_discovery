import AppLayout from "@/components/AppLayout";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { Link } from "react-router-dom";
import { Activity, Copy, Check, Zap, Target, Beaker } from "lucide-react";

const candidates = [
  { 
    id: "CAND-001", 
    name: "EGFR-Inhibitor-Alpha", 
    smiles: "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CN5", 
    score: 0.94, 
    affinity: "-9.2 kcal/mol" 
  },
  { 
    id: "CAND-002", 
    name: "EGFR-Inhibitor-Beta", 
    smiles: "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4", 
    score: 0.89, 
    affinity: "-8.7 kcal/mol" 
  },
  { 
    id: "CAND-003", 
    name: "Novel-Kinase-Ligand", 
    smiles: "CN1CCN(CC1)C2=CC=C(C=C2)NC(=O)C3=CC=CC=C3C4=CN=CN=C4", 
    score: 0.82, 
    affinity: "-8.1 kcal/mol" 
  },
  { 
    id: "CAND-004", 
    name: "Pyridine-Derivative-X", 
    smiles: "CC1=NC(=NC=C1)C2=CC=CC=C2NC(=O)C3=CC=C(C=C3)CN4CCN(CC4)C", 
    score: 0.78, 
    affinity: "-7.5 kcal/mol" 
  },
];

export default function Results() {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const copySmiles = (smiles: string, id: string) => {
    navigator.clipboard.writeText(smiles);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <AppLayout>
      <div className="min-h-screen p-8 max-w-5xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="flex flex-col gap-2 border-b border-border pb-6">
          <div className="inline-flex items-center gap-2 text-primary font-semibold text-sm">
            <Activity className="h-4 w-4" /> Experiment Summary
          </div>
          <h1 className="text-3xl font-bold text-foreground">AI-Generated Candidates</h1>
          <p className="text-muted-foreground text-sm max-w-2xl">
            Review top structural predictions from the RL-optimized generation pipeline. 
            Select a candidate below to simulate physical binding poses and quantum energy surfaces.
          </p>
        </div>

        {/* Candidates List */}
        <div className="grid grid-cols-1 gap-5">
          {candidates.map((cand) => (
            <div key={cand.id} className="bg-card border border-border rounded-2xl p-6 transition-all duration-300 hover:shadow-[0_8px_30px_rgb(0,0,0,0.04)] hover:border-foreground/20">
              
              <div className="flex flex-col md:flex-row md:items-start justify-between gap-5">
                
                {/* Info Section */}
                <div className="space-y-4 flex-1">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 bg-muted border border-border rounded-xl flex items-center justify-center text-foreground flex-shrink-0">
                      <Beaker className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="font-bold text-lg text-foreground leading-none">{cand.name}</h3>
                      <p className="text-sm font-mono text-muted-foreground mt-1.5">{cand.id}</p>
                    </div>
                  </div>

                  {/* Metrics Row */}
                  <div className="flex flex-wrap items-center gap-4 text-sm">
                    <div className="flex items-center gap-1.5 bg-muted/50 rounded-lg px-3 py-1.5 border border-border/50">
                      <Target className="h-3.5 w-3.5 text-muted-foreground" />
                      <span className="text-muted-foreground">Vina Affinity:</span>
                      <span className="font-semibold text-foreground">{cand.affinity}</span>
                    </div>
                    <div className="flex items-center gap-1.5 bg-success/10 rounded-lg px-3 py-1.5 border border-success/20">
                      <span className="text-success font-semibold text-xs uppercase tracking-wider">Score</span>
                      <span className="font-bold text-success tabular-nums">{cand.score.toFixed(2)}</span>
                    </div>
                  </div>

                  {/* SMILES Section */}
                  <div className="pt-2">
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-1.5">SMILES string</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-background border border-border rounded-lg px-4 py-2.5 overflow-hidden">
                        <p className="font-mono text-xs text-foreground truncate select-all">{cand.smiles}</p>
                      </div>
                      <Button 
                        variant="outline" 
                        size="icon"
                        className="h-9 w-9 flex-shrink-0 rounded-lg border-border bg-background hover:bg-muted"
                        onClick={() => copySmiles(cand.smiles, cand.id)}
                        title="Copy SMILES"
                      >
                        {copiedId === cand.id ? (
                          <Check className="h-4 w-4 text-success" />
                        ) : (
                          <Copy className="h-4 w-4 text-muted-foreground" />
                        )}
                      </Button>
                    </div>
                  </div>
                </div>

                {/* Right Action Section */}
                <div className="flex md:flex-col justify-end pt-1 md:pt-0">
                  <Link to={`/quantum?smiles=${encodeURIComponent(cand.smiles)}`}>
                    <Button className="rounded-xl font-medium shadow-none group h-12 px-6">
                      <Zap className="h-4 w-4 mr-2" />
                      Simulate in Quantum Lab
                    </Button>
                  </Link>
                </div>
                
              </div>
            </div>
          ))}
        </div>

      </div>
    </AppLayout>
  );
}
