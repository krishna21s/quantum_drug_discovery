import AppLayout from "@/components/AppLayout";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Loader2, FlaskConical, ArrowRight, CheckCircle2, XCircle, TrendingUp, Shield, Atom, AlertTriangle, Zap } from "lucide-react";
import { useState, useCallback, useEffect } from "react";
import { refineMolecule, type RefinementResult, type RefinementStep } from "@/lib/refineApi";
import { fetchDBCandidates, type DBCandidate } from "@/lib/dbApi";
import { type Candidate } from "@/lib/drugApi";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useLocation, Link, useNavigate } from "react-router-dom";

function scoreColor(v: number) {
  if (v > 0.7) return "text-emerald-400";
  if (v > 0.45) return "text-amber-400";
  return "text-red-400";
}

function scoreGradient(v: number) {
  if (v > 0.7) return "from-emerald-500 to-green-400";
  if (v > 0.45) return "from-amber-500 to-orange-400";
  return "from-red-500 to-rose-400";
}

export default function Refinement() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [smilesInput, setSmilesInput] = useState("");
  const [maxSteps, setMaxSteps] = useState(5);
  const [preserveScaffold, setPreserveScaffold] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RefinementResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeStep, setActiveStep] = useState(0);

  const location = useLocation();
  const navigate = useNavigate();

  // Pick up initial SMILES from redirect (e.g. from Molecules page "Optimise" button)
  useEffect(() => {
    if (location.state?.initialSmiles) {
      setSmilesInput(location.state.initialSmiles);
      // Clean up state so refresh doesn't hold it forever
      navigate(location.pathname, { replace: true });
    }
  }, [location, navigate]);

  // Load DB candidates
  useEffect(() => {
    fetchDBCandidates()
      .then((res) => setCandidates(res.candidates))
      .catch(() => {});
  }, []);

  const handleRefine = useCallback(async () => {
    if (!smilesInput.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setActiveStep(0);
    try {
      const res = await refineMolecule(smilesInput.trim(), maxSteps, preserveScaffold);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Refinement failed");
    } finally {
      setLoading(false);
    }
  }, [smilesInput, maxSteps, preserveScaffold]);

  const handleCandidateSelect = (val: string) => {
    const id = parseInt(val);
    const c = candidates.find((c) => c.rank === id);
    if (c) setSmilesInput(c.smiles);
  };

  const currentStep: RefinementStep | null = result ? result.trajectory[activeStep] : null;

  return (
    <AppLayout>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-8 space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-purple-400" />
            Lead Optimization
          </h1>
          <p className="text-muted-foreground mt-1">
            Iterative molecular refinement guided by ADMET, binding affinity &amp; toxicity oracles
          </p>
        </div>

        {/* Input Section */}
        <div className="glass-card rounded-2xl p-6 relative overflow-hidden space-y-4">
          <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-purple-500/40 to-transparent" />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* SMILES Input */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">SMILES String</label>
              <textarea
                value={smilesInput}
                onChange={(e) => setSmilesInput(e.target.value)}
                placeholder="Enter or select a SMILES string to optimize..."
                className="w-full rounded-xl border border-white/10 bg-muted/20 backdrop-blur-sm px-4 py-3 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-purple-400/40 focus:border-purple-400 transition-all resize-none h-[80px]"
              />
            </div>

            {/* DB Candidate Selector */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Or Pick From Database</label>
              <Select onValueChange={handleCandidateSelect}>
                <SelectTrigger className="rounded-xl border border-white/10 bg-muted/20 backdrop-blur-sm px-3 py-5 text-sm font-mono">
                  <SelectValue placeholder="-- Select a candidate --" />
                </SelectTrigger>
                <SelectContent className="max-h-60 rounded-xl bg-background border-white/10">
                  {candidates.map((c) => (
                    <SelectItem key={c.rank} value={String(c.rank)} className="font-mono text-xs cursor-pointer focus:bg-purple-500/20">
                      #{c.rank} — pIC₅₀ {c.xgb_pic50.toFixed(2)} — {c.smiles.substring(0, 35)}...
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <div className="flex items-center gap-4 mt-2">
                <label className="flex items-center gap-2 text-xs text-muted-foreground">
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={maxSteps}
                    onChange={(e) => setMaxSteps(Number(e.target.value))}
                    className="w-14 rounded-lg border border-white/10 bg-muted/20 px-2 py-1 text-center text-sm font-mono"
                  />
                  Max Steps
                </label>
                <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer">
                  <input
                    type="checkbox"
                    checked={preserveScaffold}
                    onChange={(e) => setPreserveScaffold(e.target.checked)}
                    className="rounded accent-purple-500"
                  />
                  Preserve Scaffold
                </label>
              </div>
            </div>
          </div>

          <Button
            onClick={handleRefine}
            disabled={!smilesInput.trim() || loading}
            className="rounded-xl px-8 py-5 text-sm font-semibold"
            style={{
              background: "linear-gradient(135deg, hsl(270 70% 55%), hsl(290 80% 60%))",
              border: "none",
              opacity: !smilesInput.trim() || loading ? 0.5 : 1,
            }}
          >
            {loading ? (
              <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Optimizing...</>
            ) : (
              <><Sparkles className="h-4 w-4 mr-2" /> Start Refinement</>
            )}
          </Button>
        </div>

        {/* Loading State */}
        {loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card rounded-2xl p-10 text-center space-y-4">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-purple-500/10 ring-1 ring-purple-500/20">
              <FlaskConical className="h-8 w-8 text-purple-400 animate-pulse" />
            </div>
            <p className="text-lg font-semibold">Running Lead Optimization...</p>
            <p className="text-sm text-muted-foreground">Mutating → Scoring (ADMET + Binding + Toxicity) → Selecting best variant</p>
            <div className="flex items-center justify-center gap-1.5 mt-2">
              {[0, 1, 2, 3, 4].map((i) => (
                <motion.div key={i} className="h-2 w-2 rounded-full bg-purple-400" animate={{ scale: [1, 1.5, 1] }} transition={{ repeat: Infinity, duration: 1, delay: i * 0.15 }} />
              ))}
            </div>
          </motion.div>
        )}

        {/* Error */}
        {error && (
          <div className="glass-card rounded-2xl p-6 text-center ring-1 ring-destructive/30">
            <p className="text-sm text-destructive font-medium">{error}</p>
          </div>
        )}

        {/* Results */}
        <AnimatePresence>
          {result && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
              {/* Summary Bar */}
              <div className="glass-card rounded-2xl p-6 relative overflow-hidden">
                <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-emerald-500/40 to-transparent" />
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Steps</p>
                    <p className="text-2xl font-bold font-mono">{result.total_steps}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Improvement</p>
                    <p className={cn("text-2xl font-bold font-mono", result.total_improvement > 0 ? "text-emerald-400" : "text-amber-400")}>
                      {result.total_improvement > 0 ? "+" : ""}{(result.total_improvement * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Time</p>
                    <p className="text-2xl font-bold font-mono">{result.elapsed_seconds.toFixed(1)}s</p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Status</p>
                    <p className={cn("text-lg font-bold", result.converged ? "text-emerald-400" : "text-amber-400")}>
                      {result.converged ? "Converged" : "Max Steps"}
                    </p>
                  </div>
                </div>
              </div>

              {/* Timeline */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left: Step Timeline */}
                <div className="space-y-2">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Evolution Timeline</h3>
                  {result.trajectory.map((step, i) => (
                    <motion.button
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.08 }}
                      onClick={() => setActiveStep(i)}
                      className={cn(
                        "w-full flex items-center gap-3 p-3 rounded-xl transition-all text-left",
                        activeStep === i
                          ? "glass-surface ring-1 ring-purple-400/30"
                          : "hover:bg-muted/20"
                      )}
                    >
                      <div className={cn(
                        "flex h-8 w-8 items-center justify-center rounded-lg text-xs font-bold shrink-0",
                        step.accepted
                          ? "bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30"
                          : "bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/30"
                      )}>
                        {step.step === 0 ? "O" : step.step}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-mono truncate">{step.smiles.substring(0, 28)}...</p>
                        <p className="text-xs text-muted-foreground">
                          {step.step === 0 ? "Original" : step.accepted ? `Δ +${(step.delta_reward * 100).toFixed(1)}%` : "Converged"}
                        </p>
                      </div>
                      {step.accepted && step.step > 0 && <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />}
                      {!step.accepted && step.step > 0 && <XCircle className="h-4 w-4 text-amber-400 shrink-0" />}
                    </motion.button>
                  ))}
                </div>

                {/* Right: Score Details for selected step */}
                <div className="lg:col-span-2 space-y-4">
                  {currentStep && (
                    <>
                      {/* Step Header */}
                      <div className="glass-card rounded-2xl p-5 relative overflow-hidden">
                        <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-purple-500/40 to-transparent" />
                        <div className="flex items-center justify-between mb-3">
                          <h3 className="text-sm font-semibold">
                            {currentStep.step === 0 ? "Original Molecule" : `Refinement Step ${currentStep.step}`}
                          </h3>
                          {currentStep.step > 0 && (
                            <span className={cn(
                              "text-xs font-semibold px-3 py-1 rounded-full ring-1",
                              currentStep.accepted
                                ? "bg-emerald-500/10 text-emerald-400 ring-emerald-500/30"
                                : "bg-amber-500/10 text-amber-400 ring-amber-500/30"
                            )}>
                              {currentStep.accepted ? "✓ Accepted" : "✗ Converged"}
                            </span>
                          )}
                        </div>
                        <p className="font-mono text-xs text-muted-foreground break-all bg-muted/20 rounded-lg p-3">
                          {currentStep.smiles}
                        </p>
                        {currentStep.variants_evaluated > 0 && (
                          <p className="text-xs text-muted-foreground mt-2">
                            Evaluated <span className="font-semibold text-foreground">{currentStep.variants_evaluated}</span> structural variants
                          </p>
                        )}
                      </div>

                      {/* Score Cards */}
                      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                        {[
                          { label: "Composite", value: currentStep.scores.composite_reward, icon: TrendingUp },
                          { label: "Binding (Ensemble)", value: currentStep.scores.ensemble_pic50, icon: Atom, raw: true },
                          { label: "Binding (XGB)", value: currentStep.scores.xgb_pic50, icon: Atom, raw: true, dim: true },
                          { label: "Binding (QSVR)", value: currentStep.scores.qsvr_pic50, icon: Zap, raw: true, dim: true },
                          { label: "ADMET", value: currentStep.scores.admet_overall, icon: Shield },
                          { label: "Safety (Ensemble)", value: 1 - currentStep.scores.toxicity_ensemble, icon: AlertTriangle },
                          { label: "Safety (XGB)", value: 1 - currentStep.scores.toxicity_xgb, icon: AlertTriangle, dim: true },
                          { label: "Safety (QSVM)", value: 1 - currentStep.scores.toxicity_quantum, icon: Zap, dim: true },
                          { label: "QED", value: currentStep.scores.qed_score, icon: Zap },
                          { label: "Absorption", value: currentStep.scores.admet_absorption, icon: Shield },
                          { label: "Distribution", value: currentStep.scores.admet_distribution, icon: Shield },
                          { label: "Metabolism", value: currentStep.scores.admet_metabolism, icon: Shield },
                        ].map((item) => (
                          <div key={item.label} className={cn("glass-surface rounded-xl p-4 text-center", "dim" in item && item.dim && "opacity-70")}>
                            <item.icon className={cn("h-4 w-4 mx-auto mb-1", item.label.includes("QSVR") || item.label.includes("QSVM") ? "text-purple-400" : "text-muted-foreground")} />
                            <p className="text-xs text-muted-foreground mb-1">{item.label}</p>
                            {"raw" in item && item.raw ? (
                              <p className="font-mono font-bold text-lg">{item.value.toFixed(2)}</p>
                            ) : (
                              <p className={cn("font-mono font-bold text-lg bg-clip-text text-transparent bg-gradient-to-r", scoreGradient(item.value))}>
                                {(item.value * 100).toFixed(0)}%
                              </p>
                            )}
                          </div>
                        ))}
                      </div>

                      {/* Before vs After comparison (only on last step) */}
                      {activeStep === result.trajectory.length - 1 && result.trajectory.length > 1 && (
                        <div className="glass-card rounded-2xl p-5 relative overflow-hidden">
                          <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-emerald-500/40 to-transparent" />
                          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">Original → Final Comparison</h3>
                          <div className="grid grid-cols-3 gap-4 text-center text-sm">
                            <div>
                              <p className="text-xs text-muted-foreground mb-2">Metric</p>
                              {["Composite", "Binding (Ens)", "Binding (QSVR)", "ADMET", "Safety (Ens)", "Safety (QSVM)", "QED"].map((m) => (
                                <p key={m} className="py-1.5 text-xs text-muted-foreground">{m}</p>
                              ))}
                            </div>
                            <div>
                              <p className="text-xs text-muted-foreground mb-2">Original</p>
                              {[
                                result.trajectory[0].scores.composite_reward,
                                result.trajectory[0].scores.ensemble_pic50,
                                result.trajectory[0].scores.qsvr_pic50,
                                result.trajectory[0].scores.admet_overall,
                                1 - result.trajectory[0].scores.toxicity_ensemble,
                                1 - result.trajectory[0].scores.toxicity_quantum,
                                result.trajectory[0].scores.qed_score,
                              ].map((v, i) => (
                                <p key={i} className="py-1.5 font-mono text-xs">
                                  {i === 1 || i === 2 ? v.toFixed(2) : `${(v * 100).toFixed(0)}%`}
                                </p>
                              ))}
                            </div>
                            <div>
                              <p className="text-xs text-muted-foreground mb-2">Final</p>
                              {[
                                currentStep.scores.composite_reward,
                                currentStep.scores.ensemble_pic50,
                                currentStep.scores.qsvr_pic50,
                                currentStep.scores.admet_overall,
                                1 - currentStep.scores.toxicity_ensemble,
                                1 - currentStep.scores.toxicity_quantum,
                                currentStep.scores.qed_score,
                              ].map((v, i) => {
                                const orig = [
                                  result.trajectory[0].scores.composite_reward,
                                  result.trajectory[0].scores.ensemble_pic50,
                                  result.trajectory[0].scores.qsvr_pic50,
                                  result.trajectory[0].scores.admet_overall,
                                  1 - result.trajectory[0].scores.toxicity_ensemble,
                                  1 - result.trajectory[0].scores.toxicity_quantum,
                                  result.trajectory[0].scores.qed_score,
                                ][i];
                                const improved = v > orig;
                                return (
                                  <p key={i} className={cn("py-1.5 font-mono text-xs font-bold", improved ? "text-emerald-400" : v < orig ? "text-red-400" : "text-foreground")}>
                                    {i === 1 || i === 2 ? v.toFixed(2) : `${(v * 100).toFixed(0)}%`}
                                    {improved && " ↑"}
                                    {v < orig && " ↓"}
                                  </p>
                                );
                              })}
                            </div>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </AppLayout>
  );
}
