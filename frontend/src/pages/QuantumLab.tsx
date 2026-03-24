import AppLayout from "@/components/AppLayout";
import QuantumCircuitDiagram from "@/components/QuantumCircuitDiagram";
import BindingSimulation from "@/components/BindingSimulation";
import ProteinTargetMap from "@/components/ProteinTargetMap";
import DiseasePanel from "@/components/DiseasePanel";
import QuantumChemPanel from "@/components/QuantumChemPanel";
import { useState, useCallback, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { Sparkles, Loader2, AlertTriangle, Copy, CheckCircle2, Beaker, Network, Zap, Target } from "lucide-react";
import { Button } from "@/components/ui/button";
import { scoreBinding, type BindingScoreResponse } from "@/lib/drugApi";
import { cn } from "@/lib/utils";

const exampleMolecules = [
  { name: "Erlotinib core", smiles: "c1ccc2c(c1)c(ncn2)Nc1cccc(c1)C#C" },
  { name: "Candidate #1", smiles: "C[C@@](CO)(Nc1ncnc2ccc(F)cc12)C1CC1" },
  { name: "Candidate #9", smiles: "c1ccc(Nc2ncnc(N3CCCC3)c2N)cn1" },
  { name: "Aspirin", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O" },
  { name: "Benzene", smiles: "c1ccccc1" },
];

export default function QuantumLab() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initSmiles = searchParams.get("smiles") || "";
  
  const [smiles, setSmiles] = useState(initSmiles);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BindingScoreResponse | null>(null);
  const [copied, setCopied] = useState(false);
  const [hasAutoRun, setHasAutoRun] = useState(false);

  const runScoring = useCallback(async (s: string) => {
    if (!s.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await scoreBinding(s.trim());
      setResult(res);
      setSearchParams(s ? { smiles: s } : {}, { replace: true });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Scoring failed");
    } finally {
      setLoading(false);
    }
  }, [setSearchParams]);

  // Auto-load if passed via URL on mount
  useEffect(() => {
    if (initSmiles && !hasAutoRun) {
      setHasAutoRun(true);
      runScoring(initSmiles);
    }
  }, [initSmiles, hasAutoRun, runScoring]);

  const copySmiles = () => {
    if (result?.canonical_smiles) {
      navigator.clipboard.writeText(result.canonical_smiles);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <AppLayout>
      <div className="min-h-screen p-8 max-w-[1600px] mx-auto space-y-8">
        
        {/* ── Header ── */}
        <div className="flex flex-col gap-2 border-b border-border pb-6">
          <div className="inline-flex items-center gap-2 text-primary font-semibold text-sm">
            <Sparkles className="h-4 w-4" /> Dual Oracle Prediction
          </div>
          <h1 className="text-3xl font-bold text-foreground">Quantum Lab Simulator</h1>
          <p className="text-muted-foreground text-sm max-w-2xl">
            Live binding affinity prediction using XGBoost + 8-qubit QSVR ensemble.
            Simulate binding energy and identify structural flaws instantaneously.
          </p>
        </div>

        {/* ── 3-Column Layout Grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 xl:grid-cols-3 gap-6 items-start">
          
          {/* ── Column 1: Input & Library ── */}
          <div className="space-y-6 flex flex-col">
            
            <div className="bg-card border border-border rounded-2xl p-6 flex flex-col">
              <h2 className="font-semibold text-base text-foreground flex items-center gap-2">
                <Beaker className="h-5 w-5" /> Molecule SMILES
              </h2>
              <p className="text-xs text-muted-foreground mt-1 mb-4">
                Enter a valid SMILES string to predict binding affinity
              </p>
              
              <textarea
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="CC1=C(C=C..."
                className={cn(
                  "w-full rounded-xl border bg-background px-4 py-3 text-sm font-mono focus:outline-none focus:border-foreground focus:ring-1 focus:ring-foreground transition-all resize-none h-28",
                  smiles.trim() ? "border-foreground/30" : "border-border"
                )}
              />
              
              <Button
                onClick={() => runScoring(smiles)}
                disabled={!smiles.trim() || loading}
                className="w-full mt-4 rounded-xl font-semibold h-12 shadow-none border border-transparent"
              >
                {loading ? (
                  <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Scoring...</>
                ) : (
                  <><Zap className="h-4 w-4 mr-2" />Simulate Affinity</>
               )}
              </Button>
            </div>

            <div className="bg-card border border-border rounded-2xl p-6">
              <h2 className="font-semibold text-base mb-4 flex items-center gap-2">
                <Network className="h-5 w-5" /> Pipeline Architecture
              </h2>
              <div className="space-y-3 text-xs text-muted-foreground">
                <div className="flex items-start gap-3">
                  <div className="h-2 w-2 rounded-full bg-foreground mt-1 flex-shrink-0" />
                  <span><strong className="text-foreground font-semibold">XGBoost:</strong> 4273-d fingerprint regression</span>
                </div>
                <div className="flex items-start gap-3">
                  <div className="h-2 w-2 rounded-full bg-primary mt-1 flex-shrink-0" />
                  <span><strong className="text-foreground font-semibold">QSVR:</strong> 8-qubit quantum kernel SVR</span>
                </div>
                <div className="flex items-start gap-3">
                  <div className="h-2 w-2 rounded-full bg-muted-foreground mt-1 flex-shrink-0" />
                  <span><strong className="text-foreground font-semibold">Target:</strong> EGFR (PDB 1M17) domain</span>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-2xl p-6">
              <h2 className="font-semibold text-base mb-1">Testing Library</h2>
              <p className="text-xs text-muted-foreground mb-4">Click any known structure</p>
              <div className="grid grid-cols-1 gap-2">
                {exampleMolecules.map((mol) => (
                  <button
                    key={mol.name}
                    onClick={() => { 
                      setSmiles(mol.smiles); 
                      runScoring(mol.smiles); 
                    }}
                    className={cn(
                      "w-full bg-background border rounded-xl px-4 py-3 text-left transition-colors duration-200 group flex justify-between items-center",
                      smiles === mol.smiles ? "border-foreground bg-accent/5" : "border-border hover:bg-muted/50"
                    )}
                  >
                    <div className="min-w-0 pr-2">
                      <p className="text-sm font-semibold text-foreground truncate">{mol.name}</p>
                      <p className="text-xs font-mono text-muted-foreground mt-0.5 truncate">{mol.smiles}</p>
                    </div>
                    {smiles === mol.smiles && <CheckCircle2 className="h-4 w-4 text-foreground flex-shrink-0" />}
                  </button>
                ))}
              </div>
            </div>

          </div>

          {/* ── Column 2: Oracle Results ── */}
          <div className="space-y-6 flex flex-col">
            
            {loading && (
              <div className="bg-card border border-border rounded-2xl p-10 text-center flex flex-col items-center justify-center h-full min-h-[400px]">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-muted border border-border mb-6">
                  <Loader2 className="h-8 w-8 text-foreground animate-spin" />
                </div>
                <p className="text-xl font-bold text-foreground mb-2">Simulating</p>
                <p className="text-sm text-muted-foreground">Running dual oracle scoring...</p>
              </div>
            )}

            {error && (
              <div className="bg-card border border-destructive/50 rounded-2xl p-10 text-center flex-grow">
                <AlertTriangle className="h-12 w-12 text-destructive mx-auto mb-4" />
                <p className="text-xl font-bold text-destructive mb-2">Failed</p>
                <p className="text-sm text-muted-foreground mb-6">{error}</p>
                <Button variant="outline" onClick={() => runScoring(smiles)} className="rounded-lg">
                  Retry
                </Button>
              </div>
            )}

            {result && !loading && (
              <>
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  <div className="bg-card border border-border rounded-2xl p-6 text-center shadow-sm">
                    <p className="text-[11px] text-muted-foreground font-bold uppercase tracking-widest mb-3">XGBoost Oracle</p>
                    <p className="text-4xl font-black text-foreground tabular-nums tracking-tighter">
                      {result.xgb_pic50?.toFixed(2) ?? "—"}
                    </p>
                    <p className="text-xs text-muted-foreground mt-2 font-medium">pIC₅₀ Score</p>
                  </div>
                  <div className="bg-card border border-border rounded-2xl p-6 text-center shadow-sm">
                    <p className="text-[11px] text-muted-foreground font-bold uppercase tracking-widest mb-3">Quantum SVR</p>
                    <p className="text-4xl font-black text-foreground tabular-nums tracking-tighter">
                      {result.qsvr_pic50?.toFixed(2) ?? "—"}
                    </p>
                    <p className="text-xs text-muted-foreground mt-2 font-medium">pIC₅₀ Score</p>
                  </div>
                </div>

                <div className="bg-card border border-border rounded-2xl p-6">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-bold text-sm text-foreground">Canonical Details</h3>
                    <Button variant="ghost" size="icon" onClick={copySmiles} className="h-7 w-7 text-muted-foreground hover:text-foreground">
                      {copied ? <CheckCircle2 className="h-3.5 w-3.5 text-success" /> : <Copy className="h-3.5 w-3.5" />}
                    </Button>
                  </div>
                  <div className="bg-background border border-border rounded-lg p-3 overflow-hidden mb-4">
                    <p className="font-mono text-xs break-all text-foreground select-all leading-relaxed">
                      {result.canonical_smiles ?? result.smiles}
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-xs mb-4">
                    <div className="bg-background border border-border rounded-md p-2">
                       <span className="text-muted-foreground block mb-0.5">Mode</span>
                       <span className="font-bold">{result.scoring_mode}</span>
                    </div>
                    <div className="bg-background border border-border rounded-md p-2">
                       <span className="text-muted-foreground block mb-0.5">Latency</span>
                       <span className="font-bold">{(result.latency_s * 1000).toFixed(0)} ms</span>
                    </div>
                  </div>
                  <div className="bg-muted/30 border border-border rounded-lg p-3 flex items-start gap-2">
                    <Target className="h-4 w-4 text-foreground mt-0.5 flex-shrink-0" />
                    <p className="text-[11px] leading-relaxed text-muted-foreground">
                      {(result.xgb_pic50 ?? 0) >= 7.0
                        ? "Strong Binder — Demonstrates lead compound quality."
                        : (result.xgb_pic50 ?? 0) >= 6.0
                          ? "Moderate Binder — Potential optimization candidate."
                          : "Weak Binder — Sub-optimal targeting."}
                    </p>
                  </div>
                </div>
                
                <div className="bg-card border border-border rounded-2xl overflow-hidden p-3">
                  <QuantumChemPanel />
                </div>
              </>
            )}

            {!loading && !error && !result && (
              <div className="bg-card border border-border border-dashed rounded-2xl p-10 text-center flex flex-col items-center justify-center min-h-[400px]">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-muted border border-border mb-4">
                  <Sparkles className="h-7 w-7 text-muted-foreground" />
                </div>
                <p className="text-lg font-bold text-foreground mb-2">Load Candidate</p>
                <p className="text-xs text-muted-foreground max-w-[250px] mx-auto leading-relaxed">
                  Provide a SMILES string representing the molecular structure to invoke the algorithms.
                </p>
              </div>
            )}

          </div>

          {/* ── Column 3: Visualizations & Modeling ── */}
          <div className="space-y-6 flex flex-col">
            <div className="bg-card border border-border rounded-2xl overflow-hidden min-h-[250px] p-2">
              <ProteinTargetMap />
            </div>
            
            <div className="bg-card border border-border rounded-2xl overflow-hidden p-2">
              <DiseasePanel />
            </div>
            
            <div className="bg-card border border-border rounded-2xl overflow-hidden p-4">
              <QuantumCircuitDiagram />
            </div>
          </div>

        </div>

        {/* ── Full Width Binding Simulation (Because it's usually wide) ── */}
        <div className="w-full bg-card border border-border rounded-2xl overflow-hidden p-4">
          <BindingSimulation />
        </div>

      </div>
    </AppLayout>
  );
}
