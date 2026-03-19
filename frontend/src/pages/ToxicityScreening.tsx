import AppLayout from "@/components/AppLayout";
import ToxicityResultPanel from "@/components/ToxicityResultPanel";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useCallback } from "react";
import {
  AlertTriangle, Loader2, Sparkles, FlaskConical,
  ChevronRight, Copy, CheckCircle2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { predictToxicity, type PredictResponse } from "@/lib/toxicityApi";
import { cn } from "@/lib/utils";

const exampleMolecules = [
  { name: "Aspirin", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O", expected: "Safe" },
  { name: "Phenanthrene", smiles: "C1=CC=C2C(=C1)C=CC3=CC=CC=C32", expected: "Toxic" },
  { name: "Bisphenol A", smiles: "CC(c1ccc(O)cc1)(c1ccc(O)cc1)C", expected: "Toxic" },
  { name: "Paracetamol", smiles: "CC(=O)Nc1ccc(O)cc1", expected: "Safe" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O", expected: "Safe" },
];

export default function ToxicityScreening() {
  const [smiles, setSmiles] = useState("");
  const [enableCI, setEnableCI] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const runPrediction = useCallback(async () => {
    if (!smiles.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await predictToxicity(smiles.trim(), enableCI);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, [smiles, enableCI]);

  const handleExampleClick = (exSmiles: string) => {
    setSmiles(exSmiles);
    setResult(null);
    setError(null);
  };

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
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="p-6 lg:p-8 space-y-6 min-h-screen"
      >
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-center gap-2 mb-1">
            <span className="stat-pill bg-destructive/15 text-destructive text-[11px] font-semibold">
              <AlertTriangle className="h-3 w-3" />
              Quantum-AI Pipeline
            </span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight">
            Toxicity <span className="gradient-text">Screening</span>
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Hybrid quantum-classical toxicity prediction · XGBoost + 20-qubit QSVM ensemble
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Input */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-1 space-y-4"
          >
            {/* SMILES Input */}
            <div className="glass-card rounded-3xl p-6 relative overflow-hidden">
              <div
                className="absolute top-0 left-6 right-6 h-[2px] rounded-full"
                style={{ background: "linear-gradient(90deg, transparent, hsl(0 72% 51%), transparent)" }}
              />
              <h2 className="font-semibold text-base mb-1">Molecule Input</h2>
              <p className="text-xs text-muted-foreground mb-4">
                Enter a SMILES string to screen for toxicity
              </p>

              <textarea
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="Enter SMILES string, e.g. CC(=O)OC1=CC=CC=C1C(=O)O"
                className={cn(
                  "w-full rounded-2xl border bg-muted/20 backdrop-blur-sm px-4 py-3",
                  "text-sm font-mono focus:outline-none focus:ring-1 transition-all resize-none h-24",
                  smiles.trim()
                    ? "border-primary/30 focus:ring-primary/40 focus:border-primary"
                    : "border-white/10 focus:ring-white/20"
                )}
              />

              {/* CI Toggle */}
              <label className="flex items-center gap-2 mt-3 cursor-pointer group">
                <input
                  type="checkbox"
                  checked={enableCI}
                  onChange={(e) => setEnableCI(e.target.checked)}
                  className="rounded border-white/20 bg-muted/30 text-quantum focus:ring-quantum/30"
                />
                <span className="text-xs text-muted-foreground group-hover:text-foreground transition-colors">
                  Enable confidence interval (slower, 15-120s)
                </span>
              </label>

              {/* Run Button */}
              <Button
                onClick={runPrediction}
                disabled={!smiles.trim() || loading}
                className="w-full mt-4 rounded-2xl font-semibold h-11"
                style={{
                  background: "linear-gradient(135deg, hsl(0 72% 51%), hsl(350 85% 62%))",
                  boxShadow: "0 8px 24px -4px hsl(0 72% 51% / 0.4)",
                  border: "none",
                  opacity: !smiles.trim() || loading ? 0.5 : 1,
                }}
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing…
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Run Toxicity Analysis
                  </>
                )}
              </Button>
            </div>

            {/* Example Molecules */}
            <div className="glass-card rounded-3xl p-6 relative overflow-hidden">
              <div
                className="absolute top-0 left-6 right-6 h-[2px] rounded-full"
                style={{ background: "linear-gradient(90deg, transparent, hsl(207 100% 50%), transparent)" }}
              />
              <h2 className="font-semibold text-base mb-1">Example Molecules</h2>
              <p className="text-xs text-muted-foreground mb-4">
                Click to auto-fill the input
              </p>
              <div className="space-y-2">
                {exampleMolecules.map((mol, i) => (
                  <motion.button
                    key={mol.name}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.15 + i * 0.05 }}
                    onClick={() => handleExampleClick(mol.smiles)}
                    className={cn(
                      "w-full glass-surface rounded-2xl px-4 py-3 text-left transition-all duration-200",
                      "hover:scale-[1.02] hover:ring-1 hover:ring-primary/20 group",
                      smiles === mol.smiles && "ring-1 ring-quantum/40 bg-quantum/5"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-semibold">{mol.name}</p>
                        <p className="text-xs font-mono text-muted-foreground mt-0.5 truncate max-w-[200px]">
                          {mol.smiles}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <span
                          className={cn(
                            "text-xs font-semibold px-2 py-0.5 rounded-full ring-1",
                            mol.expected === "Safe"
                              ? "bg-success/10 text-success ring-success/30"
                              : "bg-destructive/10 text-destructive ring-destructive/30"
                          )}
                        >
                          {mol.expected}
                        </span>
                        <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/40 group-hover:text-muted-foreground transition-colors" />
                      </div>
                    </div>
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Architecture Info */}
            <div className="glass-card rounded-3xl p-6 relative overflow-hidden">
              <div
                className="absolute top-0 left-6 right-6 h-[2px] rounded-full"
                style={{ background: "linear-gradient(90deg, transparent, hsl(280 80% 65%), transparent)" }}
              />
              <h2 className="font-semibold text-base mb-2">Pipeline Architecture</h2>
              <div className="space-y-2 text-xs text-muted-foreground">
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                  <span><strong className="text-foreground">Classical:</strong> XGBoost on 4278-d multi-fingerprint</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-purple-400" />
                  <span><strong className="text-foreground">Quantum:</strong> 20-qubit QSVM via Nyström kernel</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-quantum" />
                  <span><strong className="text-foreground">Ensemble:</strong> Conservative max-alert policy</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-warning" />
                  <span><strong className="text-foreground">Dataset:</strong> Tox21 NR-AR (500 samples)</span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Right Column: Results */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2"
          >
            <AnimatePresence mode="wait">
              {loading && (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="glass-card rounded-3xl p-12 text-center space-y-4"
                >
                  <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-quantum/10 ring-1 ring-quantum/20">
                    <FlaskConical className="h-8 w-8 text-quantum animate-pulse" />
                  </div>
                  <p className="text-lg font-semibold">Running Hybrid Analysis</p>
                  <p className="text-sm text-muted-foreground max-w-md mx-auto">
                    {enableCI
                      ? "Computing XGBoost prediction + quantum kernel with bootstrap confidence interval..."
                      : "Computing XGBoost prediction + statevector quantum kernel..."
                    }
                  </p>
                  <div className="flex justify-center gap-2 mt-4">
                    {["XGBoost", "Quantum Kernel", "Ensemble"].map((step, i) => (
                      <motion.div
                        key={step}
                        initial={{ opacity: 0.3 }}
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.3 }}
                        className="px-3 py-1 rounded-full glass-surface text-xs font-medium"
                      >
                        {step}
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}

              {error && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="glass-card rounded-3xl p-8 text-center space-y-3 ring-1 ring-destructive/30"
                >
                  <AlertTriangle className="h-10 w-10 text-destructive mx-auto" />
                  <p className="text-lg font-semibold text-destructive">Analysis Failed</p>
                  <p className="text-sm text-muted-foreground max-w-md mx-auto">{error}</p>
                  <Button variant="outline" onClick={runPrediction} className="rounded-xl mt-2">
                    Retry
                  </Button>
                </motion.div>
              )}

              {result && !loading && (
                <motion.div
                  key="result"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <ToxicityResultPanel result={result} />
                </motion.div>
              )}

              {!loading && !error && !result && (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="glass-card rounded-3xl p-12 text-center space-y-4"
                >
                  <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-muted/30 ring-1 ring-white/10">
                    <FlaskConical className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <p className="text-lg font-semibold text-muted-foreground">
                    No Analysis Yet
                  </p>
                  <p className="text-sm text-muted-foreground max-w-md mx-auto">
                    Enter a SMILES string and click <strong>Run Toxicity Analysis</strong> to
                    screen a molecule using the hybrid quantum-classical pipeline.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </motion.div>
    </AppLayout>
  );
}
