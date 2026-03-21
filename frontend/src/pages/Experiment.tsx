import AppLayout from "@/components/AppLayout";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { FlaskConical, Upload, Pencil, Search, Sparkles, ChevronRight, Zap, CheckCircle2, Loader2 } from "lucide-react";

const steps = [
  { id: 1, title: "Select Protein Target", description: "Enter PDB ID or upload protein file" },
  { id: 2, title: "Choose Molecule Source", description: "Database, upload, draw, SMILES, or AI-generate" },
  { id: 3, title: "Configure Analysis", description: "Set quantum and AI parameters" },
  { id: 4, title: "Run Experiment", description: "Execute quantum-AI analysis pipeline" },
];

const moleculeSources = [
  { id: "database", label: "Drug Database", icon: Search, desc: "Search existing compounds" },
  { id: "upload", label: "Upload File", icon: Upload, desc: "SDF, MOL2, PDB files" },
  { id: "draw", label: "Draw Molecule", icon: Pencil, desc: "Molecular editor" },
  { id: "ai", label: "AI Generate", icon: Sparkles, desc: "RL-optimized EGFR candidates" },
];

const proteinTargets = [
  { pdb: "6LU7", name: "SARS-CoV-2 Main Protease", disease: "COVID-19" },
  { pdb: "1M17", name: "EGFR Kinase Domain", disease: "Cancer" },
  { pdb: "1HHP", name: "HIV-1 Protease", disease: "HIV/AIDS" },
  { pdb: "1ZG4", name: "Beta-Lactamase", disease: "Antibiotic Resistance" },
  { pdb: "3ERT", name: "Estrogen Receptor", disease: "Breast Cancer" },
];

const pipelineStages = [
  { id: 1, label: "Molecular Docking", detail: "AutoDock Vina binding pose search" },
  { id: 2, label: "Quantum Energy Estimation", detail: "VQE ground state calculation" },
  { id: 3, label: "VQC Prediction", detail: "Variational circuit drug-activity inference" },
  { id: 4, label: "Binding Simulation", detail: "Protein–ligand interaction scoring" },
];

export default function Experiment() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedProtein, setSelectedProtein] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [pipelineStage, setPipelineStage] = useState(0);
  const [stageProgress, setStageProgress] = useState(0);

  const launchExperiment = useCallback(() => {
    // If AI Generate is selected, go directly to real candidates
    if (selectedSource === "ai") {
      navigate("/molecules");
      return;
    }
    setIsRunning(true);
    setPipelineStage(0);
    setStageProgress(0);
  }, [selectedSource, navigate]);

  useEffect(() => {
    if (!isRunning) return;
    if (pipelineStage >= pipelineStages.length) {
      const timeout = setTimeout(() => navigate("/results"), 600);
      return () => clearTimeout(timeout);
    }

    const interval = setInterval(() => {
      setStageProgress((prev) => {
        if (prev >= 100) {
          setPipelineStage((s) => s + 1);
          return 0;
        }
        return prev + Math.random() * 12 + 3;
      });
    }, 200);
    return () => clearInterval(interval);
  }, [isRunning, pipelineStage, navigate]);

  return (
    <AppLayout>
      <div className="p-8 space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <FlaskConical className="h-6 w-6 text-quantum" />
            New Drug Discovery Experiment
          </h1>
          <p className="text-muted-foreground mt-1">Step-by-step guided experiment setup</p>
        </motion.div>

        {/* Step indicator */}
        <div className="flex items-center gap-2">
          {steps.map((step, i) => (
            <div key={step.id} className="flex items-center gap-2">
              <button
                onClick={() => setCurrentStep(step.id)}
                className={`relative flex h-9 w-9 items-center justify-center rounded-full text-sm font-semibold transition-all duration-300 ${step.id === currentStep
                    ? "bg-primary text-primary-foreground shadow-[0_0_15px_-3px_hsl(217_91%_60%_/_0.5)]"
                    : step.id < currentStep
                      ? "bg-quantum/15 text-quantum ring-1 ring-quantum/30"
                      : "bg-muted/50 text-muted-foreground"
                  }`}
              >
                {step.id === currentStep && (
                  <motion.div
                    layoutId="step-active"
                    className="absolute inset-0 rounded-full ring-2 ring-primary/30"
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  />
                )}
                <span className="relative z-10">{step.id}</span>
              </button>
              <span className={`text-sm hidden md:inline ${step.id === currentStep ? "text-foreground font-medium" : "text-muted-foreground"}`}>
                {step.title}
              </span>
              {i < steps.length - 1 && <ChevronRight className="h-4 w-4 text-muted-foreground" />}
            </div>
          ))}
        </div>

        {/* Step content */}
        <motion.div key={currentStep} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.4 }} className="space-y-4">
          {currentStep === 1 && (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">Select Protein Target</h2>
              <div className="liquid-glass rounded-2xl p-4">
                <label className="text-sm text-muted-foreground">Enter PDB ID</label>
                <div className="mt-2 flex gap-2">
                  <input
                    type="text"
                    placeholder="e.g., 6LU7"
                    className="flex-1 rounded-xl border border-white/10 bg-muted/20 backdrop-blur-sm px-4 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30 font-mono transition-all"
                  />
                  <Button variant="outline" className="rounded-xl">Load</Button>
                </div>
              </div>
              <p className="text-sm text-muted-foreground">Or select from library:</p>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {proteinTargets.map((p, i) => (
                  <motion.button
                    key={p.pdb}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.06 }}
                    onClick={() => setSelectedProtein(p.pdb)}
                    className={`liquid-glass rounded-2xl p-4 text-left transition-all duration-300 ${selectedProtein === p.pdb ? "glow-cyan ring-1 ring-quantum/40" : "hover:ring-1 hover:ring-primary/20"
                      }`}
                  >
                    <p className="font-mono text-sm font-semibold text-quantum">{p.pdb}</p>
                    <p className="mt-1 text-sm font-medium">{p.name}</p>
                    <p className="text-xs text-muted-foreground">{p.disease}</p>
                  </motion.button>
                ))}
              </div>
            </div>
          )}

          {currentStep === 2 && (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">Choose Molecule Source</h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {moleculeSources.map((s, i) => (
                  <motion.button
                    key={s.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.08 }}
                    onClick={() => setSelectedSource(s.id)}
                    className={`liquid-glass flex items-center gap-4 rounded-2xl p-5 text-left transition-all duration-300 ${selectedSource === s.id ? "glow-cyan ring-1 ring-quantum/40" : "hover:ring-1 hover:ring-primary/20"
                      }`}
                  >
                    <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 ring-1 ring-primary/20">
                      <s.icon className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <p className="font-semibold">{s.label}</p>
                      <p className="text-sm text-muted-foreground">{s.desc}</p>
                    </div>
                  </motion.button>
                ))}
              </div>
              {selectedSource === "ai" && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="liquid-glass rounded-2xl ring-1 ring-quantum/20 p-5 space-y-4">
                  <h3 className="font-semibold flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-quantum" /> RL-Optimized Drug Candidates
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    50 EGFR-targeted candidates pre-generated via 500-episode RL fine-tuning
                    with dual-oracle scoring (XGBoost + 8-qubit QSVR).
                  </p>
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="glass-surface rounded-xl p-3">
                      <p className="text-xs text-muted-foreground">Target</p>
                      <p className="font-mono font-semibold text-quantum mt-1">EGFR</p>
                    </div>
                    <div className="glass-surface rounded-xl p-3">
                      <p className="text-xs text-muted-foreground">Candidates</p>
                      <p className="font-mono font-semibold mt-1">50</p>
                    </div>
                    <div className="glass-surface rounded-xl p-3">
                      <p className="text-xs text-muted-foreground">Top pIC₅₀</p>
                      <p className="font-mono font-semibold text-success mt-1">7.20</p>
                    </div>
                  </div>
                  <Button variant="hero" className="w-full rounded-xl" onClick={() => navigate("/molecules")}>
                    <FlaskConical className="h-4 w-4 mr-2" /> View All Candidates
                  </Button>
                </motion.div>
              )}
            </div>
          )}

          {currentStep === 3 && (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">Configure Analysis</h2>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="liquid-glass rounded-2xl p-5 space-y-3 relative overflow-hidden">
                  <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent" />
                  <h3 className="font-semibold text-sm">Quantum Parameters</h3>
                  <div>
                    <label className="text-xs text-muted-foreground">VQE Optimizer</label>
                    <select className="mt-1 w-full rounded-xl border border-white/10 bg-muted/20 backdrop-blur-sm px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30 transition-all">
                      <option>COBYLA</option>
                      <option>SPSA</option>
                      <option>L-BFGS-B</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-muted-foreground">Max iterations</label>
                    <input type="number" defaultValue={100} className="mt-1 w-full rounded-xl border border-white/10 bg-muted/20 backdrop-blur-sm px-3 py-2 text-sm font-mono focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30 transition-all" />
                  </div>
                </div>
                <div className="liquid-glass rounded-2xl p-5 space-y-3 relative overflow-hidden">
                  <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/30 to-transparent" />
                  <h3 className="font-semibold text-sm">AI Configuration</h3>
                  <div>
                    <label className="text-xs text-muted-foreground">Docking engine</label>
                    <select className="mt-1 w-full rounded-xl border border-white/10 bg-muted/20 backdrop-blur-sm px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30 transition-all">
                      <option>AutoDock Vina</option>
                      <option>DiffDock</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-muted-foreground">ADMET Prediction</label>
                    <select className="mt-1 w-full rounded-xl border border-white/10 bg-muted/20 backdrop-blur-sm px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30 transition-all">
                      <option>Enabled</option>
                      <option>Disabled</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>
          )}

          {currentStep === 4 && (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">{isRunning ? "Running Pipeline" : "Ready to Run"}</h2>
              <AnimatePresence mode="wait">
                {!isRunning ? (
                  <motion.div key="ready" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="liquid-glass rounded-2xl p-8 text-center space-y-4">
                    <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-quantum/10 ring-1 ring-quantum/20">
                      <Sparkles className="h-8 w-8 text-quantum animate-pulse-glow" />
                    </div>
                    <p className="text-lg font-semibold">Experiment configured</p>
                    <p className="text-sm text-muted-foreground max-w-md mx-auto">
                      {selectedProtein ? `Target: ${selectedProtein}` : "No target selected"} · {selectedSource ? `Source: ${selectedSource}` : "No source selected"}
                    </p>
                    <p className="text-sm text-muted-foreground max-w-md mx-auto">
                      The quantum-AI pipeline will perform molecular docking, VQE energy estimation, VQC prediction, and binding simulation.
                    </p>
                    <Button variant="hero" size="xl" onClick={launchExperiment} className="rounded-xl">
                      <Zap className="h-5 w-5" />
                      Launch Experiment
                    </Button>
                  </motion.div>
                ) : (
                  <motion.div key="running" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="liquid-glass rounded-2xl p-6 space-y-5">
                    {pipelineStages.map((stage, i) => {
                      const isDone = i < pipelineStage;
                      const isActive = i === pipelineStage;
                      const isPending = i > pipelineStage;
                      return (
                        <div key={stage.id} className="space-y-2">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              {isDone ? (
                                <CheckCircle2 className="h-5 w-5 text-quantum drop-shadow-[0_0_6px_hsl(187_79%_54%_/_0.5)]" />
                              ) : isActive ? (
                                <Loader2 className="h-5 w-5 text-primary animate-spin" />
                              ) : (
                                <div className="h-5 w-5 rounded-full border border-muted-foreground/30" />
                              )}
                              <span className={`text-sm font-medium ${isPending ? "text-muted-foreground" : ""}`}>{stage.label}</span>
                            </div>
                            <span className="text-xs text-muted-foreground font-mono">{isDone ? "100%" : isActive ? `${Math.min(Math.round(stageProgress), 100)}%` : ""}</span>
                          </div>
                          <div className="h-2 rounded-full bg-muted/30 overflow-hidden ring-1 ring-white/5">
                            <motion.div
                              className="h-full rounded-full bg-gradient-to-r from-primary to-quantum"
                              style={{ width: `${isDone ? 100 : isActive ? Math.min(stageProgress, 100) : 0}%` }}
                              transition={{ duration: 0.2 }}
                            />
                          </div>
                          {isActive && <p className="text-xs text-muted-foreground pl-7">{stage.detail}</p>}
                        </div>
                      );
                    })}
                    {pipelineStage >= pipelineStages.length && (
                      <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center text-sm font-medium text-quantum neon-text">
                        ✓ Pipeline complete — loading results…
                      </motion.p>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </motion.div>

        {/* Navigation */}
        <div className="flex justify-between pt-4">
          <Button variant="outline" disabled={currentStep === 1} onClick={() => setCurrentStep((s) => s - 1)} className="rounded-xl">
            Previous
          </Button>
          <Button variant="default" disabled={currentStep === 4} onClick={() => setCurrentStep((s) => s + 1)} className="rounded-xl">
            Next <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </AppLayout>
  );
}
