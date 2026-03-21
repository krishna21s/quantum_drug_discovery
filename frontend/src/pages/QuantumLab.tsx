import AppLayout from "@/components/AppLayout";
import QuantumCircuitDiagram from "@/components/QuantumCircuitDiagram";
import QuantumOutputPanel from "@/components/QuantumOutputPanel";
import BindingSimulation from "@/components/BindingSimulation";
import ProteinTargetMap from "@/components/ProteinTargetMap";
import DiseasePanel from "@/components/DiseasePanel";
import QuantumChemPanel from "@/components/QuantumChemPanel";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useCallback } from "react";
import { Sparkles, Loader2, AlertTriangle, Copy, CheckCircle2 } from "lucide-react";
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

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } },
};
const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

export default function QuantumLab() {
  const [smiles, setSmiles] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BindingScoreResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const runScoring = useCallback(async () => {
    if (!smiles.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await scoreBinding(smiles.trim());
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Scoring failed");
    } finally {
      setLoading(false);
    }
  }, [smiles]);

  const copySmiles = () => {
    if (result?.canonical_smiles) {
      navigator.clipboard.writeText(result.canonical_smiles);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  return (
    <AppLayout>
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="p-6 lg:p-8 space-y-6"
      >
        {/* Header */}
        <motion.div variants={item}>
          <div className="flex items-center gap-2 mb-1">
            <span className="stat-pill bg-purple-500/15 text-purple-400 text-[11px] font-semibold">
              <Sparkles className="h-3 w-3" /> Dual Oracle
            </span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight">
            Quantum <span className="gradient-text">Lab</span>
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Live binding affinity prediction · XGBoost + 8-qubit QSVR ensemble
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Input */}
          <motion.div variants={item} className="lg:col-span-1 space-y-4">
            {/* SMILES Input */}
            <div className="glass-card rounded-3xl p-6 relative overflow-hidden">
              <div
                className="absolute top-0 left-6 right-6 h-[2px] rounded-full"
                style={{ background: "linear-gradient(90deg, transparent, hsl(270 80% 65%), transparent)" }}
              />
              <h2 className="font-semibold text-base mb-1">Molecule Input</h2>
              <p className="text-xs text-muted-foreground mb-4">
                Enter SMILES to predict binding affinity (pIC₅₀)
              </p>
              <textarea
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="Enter SMILES, e.g. c1ccc(Nc2ncnc3ccccc23)cn1"
                className={cn(
                  "w-full rounded-2xl border bg-muted/20 backdrop-blur-sm px-4 py-3",
                  "text-sm font-mono focus:outline-none focus:ring-1 transition-all resize-none h-24",
                  smiles.trim()
                    ? "border-purple-500/30 focus:ring-purple-500/40"
                    : "border-white/10 focus:ring-white/20"
                )}
              />
              <Button
                onClick={runScoring}
                disabled={!smiles.trim() || loading}
                className="w-full mt-4 rounded-2xl font-semibold h-11"
                style={{
                  background: "linear-gradient(135deg, hsl(270 80% 55%), hsl(217 91% 60%))",
                  boxShadow: "0 8px 24px -4px hsl(270 80% 55% / 0.4)",
                  border: "none",
                  opacity: !smiles.trim() || loading ? 0.5 : 1,
                }}
              >
                {loading ? (
                  <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Scoring…</>
                ) : (
                  <><Sparkles className="h-4 w-4 mr-2" />Score Binding Affinity</>
                )}
              </Button>
            </div>

            {/* Examples */}
            <div className="glass-card rounded-3xl p-6 relative overflow-hidden">
              <div
                className="absolute top-0 left-6 right-6 h-[2px] rounded-full"
                style={{ background: "linear-gradient(90deg, transparent, hsl(217 91% 60%), transparent)" }}
              />
              <h2 className="font-semibold text-base mb-1">Example Molecules</h2>
              <p className="text-xs text-muted-foreground mb-4">Click to auto-fill</p>
              <div className="space-y-2">
                {exampleMolecules.map((mol, i) => (
                  <motion.button
                    key={mol.name}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.15 + i * 0.05 }}
                    onClick={() => { setSmiles(mol.smiles); setResult(null); setError(null); }}
                    className={cn(
                      "w-full glass-surface rounded-2xl px-4 py-3 text-left transition-all duration-200",
                      "hover:scale-[1.02] hover:ring-1 hover:ring-purple-400/20 group",
                      smiles === mol.smiles && "ring-1 ring-purple-400/40 bg-purple-400/5"
                    )}
                  >
                    <p className="text-sm font-semibold">{mol.name}</p>
                    <p className="text-xs font-mono text-muted-foreground mt-0.5 truncate">{mol.smiles}</p>
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Architecture */}
            <div className="glass-card rounded-3xl p-6 relative overflow-hidden">
              <div className="absolute top-0 left-6 right-6 h-[2px] rounded-full"
                style={{ background: "linear-gradient(90deg, transparent, hsl(187 79% 54%), transparent)" }} />
              <h2 className="font-semibold text-base mb-2">Pipeline Architecture</h2>
              <div className="space-y-2 text-xs text-muted-foreground">
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                  <span><strong className="text-foreground">XGBoost:</strong> 4273-d fingerprint regression</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-purple-400" />
                  <span><strong className="text-foreground">QSVR:</strong> 8-qubit kernel SVR (Nyström, 100 landmarks)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-quantum" />
                  <span><strong className="text-foreground">Target:</strong> EGFR (PDB 1M17) kinase domain</span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Right: Results + Visualizations */}
          <motion.div variants={item} className="lg:col-span-2 space-y-6">
            <AnimatePresence mode="wait">
              {/* Loading */}
              {loading && (
                <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="glass-card rounded-3xl p-12 text-center space-y-4">
                  <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-purple-400/10 ring-1 ring-purple-400/20">
                    <Sparkles className="h-8 w-8 text-purple-400 animate-pulse" />
                  </div>
                  <p className="text-lg font-semibold">Computing Binding Affinity</p>
                  <p className="text-sm text-muted-foreground">Running XGBoost + QSVR dual oracle scoring...</p>
                  <div className="flex justify-center gap-2 mt-4">
                    {["Feature Extraction", "XGBoost", "QSVR Kernel"].map((step, i) => (
                      <motion.div key={step} initial={{ opacity: 0.3 }}
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.3 }}
                        className="px-3 py-1 rounded-full glass-surface text-xs font-medium">
                        {step}
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* Error */}
              {error && (
                <motion.div key="error" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  className="glass-card rounded-3xl p-8 text-center space-y-3 ring-1 ring-destructive/30">
                  <AlertTriangle className="h-10 w-10 text-destructive mx-auto" />
                  <p className="text-lg font-semibold text-destructive">Scoring Failed</p>
                  <p className="text-sm text-muted-foreground">{error}</p>
                  <Button variant="outline" onClick={runScoring} className="rounded-xl mt-2">Retry</Button>
                </motion.div>
              )}

              {/* Result */}
              {result && !loading && (
                <motion.div key="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="space-y-4">
                  {/* Dual Score Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="glass-card rounded-3xl p-6 text-center relative overflow-hidden">
                      <div className="absolute top-0 left-6 right-6 h-[2px] rounded-full"
                        style={{ background: "linear-gradient(90deg, transparent, hsl(187 79% 54%), transparent)" }} />
                      <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wider">XGBoost pIC₅₀</p>
                      <p className="text-5xl font-bold text-quantum mt-2">
                        {result.xgb_pic50?.toFixed(2) ?? "—"}
                      </p>
                      <p className="text-xs text-muted-foreground mt-2">
                        IC₅₀ ≈ {result.xgb_pic50 ? `${(Math.pow(10, -result.xgb_pic50) * 1e9).toFixed(0)} nM` : "—"}
                      </p>
                    </div>
                    <div className="glass-card rounded-3xl p-6 text-center relative overflow-hidden">
                      <div className="absolute top-0 left-6 right-6 h-[2px] rounded-full"
                        style={{ background: "linear-gradient(90deg, transparent, hsl(270 80% 65%), transparent)" }} />
                      <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wider">QSVR pIC₅₀</p>
                      <p className="text-5xl font-bold text-purple-400 mt-2">
                        {result.qsvr_pic50?.toFixed(2) ?? "—"}
                      </p>
                      <p className="text-xs text-muted-foreground mt-2">
                        IC₅₀ ≈ {result.qsvr_pic50 ? `${(Math.pow(10, -result.qsvr_pic50) * 1e9).toFixed(0)} nM` : "—"}
                      </p>
                    </div>
                  </div>

                  {/* Details */}
                  <div className="glass-card rounded-3xl p-6 space-y-3">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold">Scoring Details</h3>
                      <button onClick={copySmiles} className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors">
                        {copied ? <CheckCircle2 className="h-3.5 w-3.5 text-success" /> : <Copy className="h-3.5 w-3.5" />}
                        {copied ? "Copied!" : "Copy SMILES"}
                      </button>
                    </div>
                    <div className="bg-muted/20 rounded-xl p-3">
                      <p className="font-mono text-xs break-all">{result.canonical_smiles ?? result.smiles}</p>
                    </div>
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Scoring mode</span>
                        <span className="font-mono">{result.scoring_mode}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Latency</span>
                        <span className="font-mono">{(result.latency_s * 1000).toFixed(0)}ms</span>
                      </div>
                    </div>
                    {/* Activity interpretation */}
                    <div className="bg-muted/10 rounded-xl p-3 mt-2">
                      <p className="text-xs text-muted-foreground">
                        {(result.xgb_pic50 ?? 0) >= 7.0
                          ? "🟢 Strong predicted binder — lead compound quality (IC₅₀ < 100 nM)"
                          : (result.xgb_pic50 ?? 0) >= 6.0
                            ? "🟡 Moderate predicted activity — optimization candidate"
                            : "🟠 Weak predicted activity — may need structural modification"}
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Empty */}
              {!loading && !error && !result && (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="glass-card rounded-3xl p-12 text-center space-y-4">
                  <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-muted/30 ring-1 ring-white/10">
                    <Sparkles className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <p className="text-lg font-semibold text-muted-foreground">No Prediction Yet</p>
                  <p className="text-sm text-muted-foreground max-w-md mx-auto">
                    Enter a SMILES string and click <strong>Score Binding Affinity</strong> to
                    predict pIC₅₀ using the dual XGBoost + QSVR quantum oracle.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Bottom visualization row */}
            <motion.div variants={item} className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <ProteinTargetMap />
              <div className="space-y-6">
                <DiseasePanel />
                <QuantumChemPanel />
              </div>
            </motion.div>

            <motion.div variants={item}>
              <QuantumCircuitDiagram />
            </motion.div>

            <motion.div variants={item}>
              <BindingSimulation />
            </motion.div>
          </motion.div>
        </div>
      </motion.div>
    </AppLayout>
  );
}
