import AppLayout from "@/components/AppLayout";
import { useExperiment } from "@/context/ExperimentContext";
import { saveExperiment } from "@/lib/experimentDbApi";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  FlaskConical, Save, CheckCircle2, Loader2, AlertTriangle,
  Atom, Shield, Activity, Clock, Thermometer, Target, Database, Sparkles
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Candidate } from "@/lib/drugApi";

export default function ExperimentResults() {
  const { session, hasSession } = useExperiment();
  const navigate = useNavigate();
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedId, setSavedId] = useState<number | null>(null);
  const [selected, setSelected] = useState<Candidate | null>(null);

  const handleSave = async () => {
    if (!session) return;
    setSaving(true);
    setSaveError(null);
    try {
      const res = await saveExperiment({
        pdb_id: session.config.pdb_id || "unknown",
        target_name: session.result.target,
        temperature: session.config.temperature || 1.0,
        n_candidates: session.config.n_candidates || 20,
        stress_factors: session.config.stress_factors || [],
        docking_engine: session.config.docking_engine || "autodock_vina",
        vqe_optimizer: session.config.vqe_optimizer || "COBYLA",
        vqe_max_iterations: session.config.vqe_max_iterations || 100,
        run_admet: session.config.run_admet ?? true,
        generation_time_s: session.result.generation_time_s,
        n_sampled: session.result.n_sampled,
        n_valid: session.result.n_valid,
        candidates_json: session.result.candidates,
      });
      setSaved(true);
      setSavedId(res.id);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  if (!hasSession || !session) {
    return (
      <AppLayout>
        <div className="min-h-screen p-8 max-w-[1200px] mx-auto flex items-center justify-center">
          <div className="glass-card rounded-3xl p-12 text-center space-y-4 max-w-md">
            <FlaskConical className="h-12 w-12 text-muted-foreground mx-auto" />
            <h2 className="text-xl font-bold">No Active Experiment</h2>
            <p className="text-sm text-muted-foreground">
              Run a new experiment first to see results here.
            </p>
            <Link to="/experiment">
              <Button className="rounded-xl mt-2">Run New Experiment</Button>
            </Link>
          </div>
        </div>
      </AppLayout>
    );
  }

  const { config, result, timestamp } = session;
  const candidates: Candidate[] = result.candidates;

  // Auto-select first candidate if not selected
  if (!selected && candidates.length > 0) {
    setSelected(candidates[0]);
  }

  return (
    <AppLayout>
      <div className="min-h-screen p-6 lg:p-8 max-w-[1400px] mx-auto space-y-6">

        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border pb-6">
          <div>
            <div className="inline-flex items-center gap-2 text-primary font-semibold text-sm mb-1">
              <FlaskConical className="h-4 w-4" /> Experiment Results
            </div>
            <h1 className="text-2xl font-bold">
              {result.target} — {candidates.length} Candidates
            </h1>
            <p className="text-xs text-muted-foreground mt-1">
              Generated {new Date(timestamp).toLocaleString()} · {result.generation_time_s.toFixed(1)}s
            </p>
          </div>
          <div className="flex gap-3">
            <Link to="/molecules">
              <Button variant="outline" className="rounded-xl gap-2">
                <Atom className="h-4 w-4" /> View in Molecules
              </Button>
            </Link>
            <Button
              onClick={handleSave}
              disabled={saving || saved}
              className={cn("rounded-xl gap-2", saved && "bg-emerald-600 hover:bg-emerald-600")}
            >
              {saving ? (
                <><Loader2 className="h-4 w-4 animate-spin" /> Saving...</>
              ) : saved ? (
                <><CheckCircle2 className="h-4 w-4" /> Saved (ID: {savedId})</>
              ) : (
                <><Database className="h-4 w-4" /> Save to Database</>
              )}
            </Button>
          </div>
        </div>

        {saveError && (
          <div className="bg-destructive/10 border border-destructive/30 rounded-xl p-3 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-destructive" />
            <span className="text-sm text-destructive">{saveError}</span>
          </div>
        )}

        {/* Config Summary */}
        <div className="glass-card rounded-2xl p-5">
          <h3 className="font-semibold text-sm mb-3 flex items-center gap-2">
            <Target className="h-4 w-4" /> Experiment Configuration
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3">
            {[
              { label: "Target", value: config.pdb_id || "—", icon: Target },
              { label: "Temperature", value: String(config.temperature || 1.0), icon: Thermometer },
              { label: "Candidates", value: String(config.n_candidates || 20), icon: Atom },
              { label: "Docking", value: config.docking_engine || "vina", icon: Activity },
              { label: "VQE", value: config.vqe_optimizer || "COBYLA", icon: Shield },
              { label: "Iterations", value: String(config.vqe_max_iterations || 100), icon: Clock },
              { label: "ADMET", value: config.run_admet !== false ? "Yes" : "No", icon: Shield },
              { label: "Time", value: `${result.generation_time_s.toFixed(1)}s`, icon: Clock },
            ].map((item) => (
              <div key={item.label} className="bg-background border border-border rounded-xl p-3 text-center">
                <item.icon className="h-3.5 w-3.5 mx-auto text-muted-foreground mb-1" />
                <p className="text-[10px] text-muted-foreground uppercase">{item.label}</p>
                <p className="font-bold text-sm mt-0.5 truncate">{item.value}</p>
              </div>
            ))}
          </div>
          {config.stress_factors && config.stress_factors.length > 0 && (
            <div className="mt-3 flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Stress:</span>
              {config.stress_factors.map((s) => (
                <span key={s} className="px-2 py-0.5 rounded-lg text-[10px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
                  {s}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Candidate Table */}
          <div className="lg:col-span-2 glass-card rounded-2xl overflow-hidden">
            <div className="p-4 border-b border-border">
            <h3 className="font-semibold text-sm">All Candidates ({candidates.length})</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-muted-foreground uppercase border-b border-border">
                  <th className="text-left p-3">#</th>
                  <th className="text-left p-3">SMILES</th>
                  <th className="text-center p-3">XGB pIC₅₀</th>
                  <th className="text-center p-3">QSVR pIC₅₀</th>
                  <th className="text-center p-3">QED</th>
                  <th className="text-center p-3">MW</th>
                  <th className="text-center p-3">LogP</th>
                  <th className="text-center p-3">Docking</th>
                  <th className="text-center p-3">ADMET</th>
                  <th className="text-center p-3">Lipinski</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((c, i) => (
                  <tr 
                    key={i} 
                    onClick={() => setSelected(c)}
                    className={cn(
                      "border-b border-border/50 transition-colors cursor-pointer",
                      selected?.rank === c.rank ? "bg-primary/10 ring-1 ring-primary/20" : "hover:bg-muted/20"
                    )}
                  >
                    <td className="p-3 font-bold text-muted-foreground">{c.rank || i + 1}</td>
                    <td className="p-3 font-mono text-xs max-w-[200px] truncate">{c.smiles}</td>
                    <td className="p-3 text-center">
                      <span className={cn("font-bold", (c.xgb_pic50 ?? 0) >= 7 ? "text-emerald-400" : (c.xgb_pic50 ?? 0) >= 6 ? "text-amber-400" : "text-red-400")}>
                        {c.xgb_pic50?.toFixed(2) ?? "—"}
                      </span>
                    </td>
                    <td className="p-3 text-center">
                      <span className={cn("font-bold", (c.quantum_pic50 ?? 0) >= 7 ? "text-emerald-400" : (c.quantum_pic50 ?? 0) >= 6 ? "text-amber-400" : "text-red-400")}>
                        {c.quantum_pic50?.toFixed(2) ?? "—"}
                      </span>
                    </td>
                    <td className="p-3 text-center font-semibold">{(c.qed * 100).toFixed(0)}%</td>
                    <td className="p-3 text-center text-muted-foreground">{c.mw.toFixed(0)}</td>
                    <td className="p-3 text-center text-muted-foreground">{c.logp.toFixed(1)}</td>
                    <td className="p-3 text-center">
                      <span className={cn("font-semibold", (c.docking_score ?? 0) < -7 ? "text-emerald-400" : "text-muted-foreground")}>
                        {c.docking_score?.toFixed(1) ?? "—"}
                      </span>
                    </td>
                    <td className="p-3 text-center">
                      {c.admet ? (
                        <span className={cn(
                          "px-2 py-0.5 rounded-full text-[10px] font-bold",
                          c.admet.verdict === "Promising" ? "bg-emerald-500/10 text-emerald-400" :
                          c.admet.verdict === "Acceptable" ? "bg-amber-500/10 text-amber-400" :
                          "bg-red-500/10 text-red-400"
                        )}>
                          {c.admet.overall.toFixed(0)}% {c.admet.verdict}
                        </span>
                      ) : "—"}
                    </td>
                    <td className="p-3 text-center">
                      {c.lipinski_pass ? (
                        <CheckCircle2 className="h-4 w-4 text-emerald-400 mx-auto" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-amber-400 mx-auto" />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

          {/* Right Panel: Selected Candidate Detail */}
          <div>
            <AnimatePresence mode="wait">
              {selected && (
                <motion.div
                  key={selected.rank}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="glass-card rounded-3xl p-6 space-y-4 sticky top-6"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <h2 className="font-bold text-lg">Candidate #{selected.rank || "?"}</h2>
                      {selected.lipinski_pass && (
                        <span className="text-[10px] bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/30 px-2 py-0.5 rounded-full font-bold tracking-wide uppercase">
                          Lipinski ✓
                        </span>
                      )}
                    </div>
                    <Button 
                      size="sm" 
                      variant="outline" 
                      onClick={() => navigate('/refinement', { state: { initialSmiles: selected.smiles } })}
                      className="h-8 px-3 gap-1.5 text-xs text-primary border-primary/30 hover:bg-primary/10 hover:text-primary hover:border-primary/60 transition-colors shadow-sm"
                    >
                      <Sparkles className="h-3.5 w-3.5" />
                      Optimise
                    </Button>
                  </div>

                  <div className="bg-white dark:bg-white/5 rounded-2xl p-2 flex justify-center items-center border border-border overflow-hidden relative group">
                    <img 
                      src={`/api/image/render?smiles=${encodeURIComponent(selected.smiles)}`}
                      alt={`2D structure of ${selected.smiles}`}
                      className="w-full object-contain rounded-xl dark:invert transition-transform duration-300 group-hover:scale-105"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                        (e.target as HTMLImageElement).parentElement!.innerHTML = '<div class="p-6 text-center text-xs text-muted-foreground">Unable to render 2D structure</div>';
                      }}
                    />
                  </div>

                  <div className="bg-muted/20 rounded-xl p-3">
                    <p className="font-mono text-xs break-all leading-relaxed" title={selected.smiles}>{selected.smiles}</p>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="glass-surface rounded-xl p-3 text-center">
                      <p className="text-xs text-muted-foreground">XGB pIC₅₀</p>
                      <p className="text-xl font-bold text-primary mt-1">{selected.xgb_pic50?.toFixed(2) ?? "—"}</p>
                    </div>
                    <div className="glass-surface rounded-xl p-3 text-center">
                      <p className="text-xs text-muted-foreground">QSVR pIC₅₀</p>
                      <p className="text-xl font-bold text-purple-400 mt-1">
                        {selected.quantum_pic50?.toFixed(2) ?? "—"}
                      </p>
                    </div>
                    <div className="glass-surface rounded-xl p-3 text-center">
                      <p className="text-xs text-muted-foreground">QED</p>
                      <p className="text-xl font-bold mt-1">{(selected.qed * 100).toFixed(0)}%</p>
                    </div>
                    <div className="glass-surface rounded-xl p-3 text-center">
                      <p className="text-xs text-muted-foreground">SA Score</p>
                      <p className="text-xl font-bold mt-1">{selected.sa_score.toFixed(1)}</p>
                    </div>
                  </div>

                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Molecular Weight</span>
                      <span className="font-mono">{selected.mw.toFixed(1)} Da</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">LogP</span>
                      <span className="font-mono">{selected.logp.toFixed(2)}</span>
                    </div>
                    {selected.tpsa !== null && selected.tpsa !== undefined && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">TPSA</span>
                        <span className="font-mono">{selected.tpsa.toFixed(1)} Å²</span>
                      </div>
                    )}
                    {selected.docking_score != null && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Docking Score</span>
                        <span className={cn("font-mono font-semibold", selected.docking_score <= -8 ? "text-emerald-400" : selected.docking_score <= -6 ? "text-primary" : "text-amber-400")}>
                          {selected.docking_score.toFixed(2)} kcal/mol
                        </span>
                      </div>
                    )}
                  </div>

                  {/* ADMET Panel */}
                  {selected.admet && (
                    <div className="bg-muted/10 rounded-xl p-3 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold">ADMET Profile</span>
                        <span className={cn(
                          "text-xs font-semibold px-2 py-0.5 rounded-full ring-1",
                          selected.admet.verdict === "Promising" ? "bg-emerald-500/10 text-emerald-400 ring-emerald-500/30" :
                          selected.admet.verdict === "Acceptable" ? "bg-amber-500/10 text-amber-400 ring-amber-500/30" :
                          "bg-red-500/10 text-red-400 ring-red-500/30"
                        )}>
                          {selected.admet.verdict}
                        </span>
                      </div>
                      <div className="flex justify-between text-xs pt-1 border-t border-border/50">
                        <span className="font-semibold">Overall Safety</span>
                        <span className="font-mono font-semibold">{(selected.admet.overall * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
