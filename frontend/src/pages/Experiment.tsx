import AppLayout from "@/components/AppLayout";
import { Button } from "@/components/ui/button";
import { useState, useEffect, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  FlaskConical,
  ChevronRight,
  Zap,
  CheckCircle2,
  Loader2,
  Activity,
  Thermometer,
  ShieldAlert,
  GitBranch,
  AlertCircle,
  Sparkles,
  Bot,
} from "lucide-react";
import { generateCandidates, type GenerateResponse } from "@/lib/drugApi";
import { autoConfigureExperiment, type AutoConfigResponse } from "@/lib/experimentApi";
import { useExperiment } from "@/context/ExperimentContext";

const steps = [
  { id: 1, title: "Select Target", description: "Choose protein target" },
  { id: 2, title: "Configure & Stress", description: "Quantum & disease params" },
  { id: 3, title: "Run Experiment", description: "Execute AI pipeline" },
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
  { id: 2, label: "Target Stress Simulation", detail: "Applying structural perturbations" },
  { id: 3, label: "Quantum Energy Estimation", detail: "VQE ground state calculation" },
  { id: 4, label: "VQC Prediction", detail: "Variational circuit drug-activity" },
  { id: 5, label: "ADMET Screening", detail: "Safety & druglikeness profiling" },
];

const stressModifiers = [
  { id: "mutation", label: "Point Mutation", icon: GitBranch },
  { id: "folding", label: "Folding Stress", icon: Activity },
  { id: "thermal", label: "Thermal Instability", icon: Thermometer },
  { id: "binding", label: "Binding Site Deformation", icon: ShieldAlert },
];

export default function Experiment() {
  const navigate = useNavigate();
  const { saveSession } = useExperiment();
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedProtein, setSelectedProtein] = useState<string | null>(null);
  const [customPdb, setCustomPdb] = useState("");
  const [stressFactors, setStressFactors] = useState<string[]>([]);

  // Generation parameters (Step 2) — ALL now state-controlled
  const [nCandidates, setNCandidates] = useState(20);
  const [temperature, setTemperature] = useState(1.0);
  const [vqeOptimizer, setVqeOptimizer] = useState("COBYLA");
  const [vqeMaxIterations, setVqeMaxIterations] = useState(100);
  const [dockingEngine, setDockingEngine] = useState("autodock_vina");
  const [runAdmet, setRunAdmet] = useState(true);

  // Auto-configure state
  const [autoConfiguring, setAutoConfiguring] = useState(false);
  const [autoConfigResult, setAutoConfigResult] = useState<AutoConfigResponse | null>(null);

  // Pipeline execution state (Step 3)
  const [isRunning, setIsRunning] = useState(false);
  const [pipelineStage, setPipelineStage] = useState(0);
  const [stageProgress, setStageProgress] = useState(0);
  const [runError, setRunError] = useState<string | null>(null);

  // Ref to hold the API result so the animation can check it
  const apiResultRef = useRef<GenerateResponse | null>(null);
  const apiErrorRef = useRef<string | null>(null);
  const apiDoneRef = useRef(false);

  const toggleStress = (id: string) => {
    setStressFactors((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const handleLoadTarget = () => {
    const trimmed = customPdb.trim().toUpperCase();
    if (trimmed.length >= 3) {
      setSelectedProtein(trimmed);
    }
  };

  // ── LLM Auto-Configure ──────────────────────────────────
  const handleAutoConfig = useCallback(async () => {
    if (!selectedProtein) return;
    setAutoConfiguring(true);
    setAutoConfigResult(null);
    try {
      const config = await autoConfigureExperiment(selectedProtein);
      // Apply all recommended params
      setTemperature(config.temperature);
      setNCandidates(config.n_candidates);
      setVqeOptimizer(config.vqe_optimizer);
      setVqeMaxIterations(config.vqe_max_iterations);
      setDockingEngine(config.docking_engine);
      setStressFactors(config.stress_factors);
      setRunAdmet(config.run_admet);
      setAutoConfigResult(config);
    } catch (err) {
      console.error("Auto-configure failed:", err);
    } finally {
      setAutoConfiguring(false);
    }
  }, [selectedProtein]);

  // ── Launch Experiment ───────────────────────────────────
  const launchExperiment = useCallback(() => {
    if (!selectedProtein) return;
    setIsRunning(true);
    setPipelineStage(0);
    setStageProgress(0);
    setRunError(null);
    apiResultRef.current = null;
    apiErrorRef.current = null;
    apiDoneRef.current = false;

    // Fire the actual API call with ALL params
    generateCandidates({
      pdb_id: selectedProtein,
      n_candidates: nCandidates,
      temperature,
      stress_factors: stressFactors,
      docking_engine: dockingEngine,
      run_admet: runAdmet,
      vqe_optimizer: vqeOptimizer,
      vqe_max_iterations: vqeMaxIterations,
    })
      .then((result) => {
        apiResultRef.current = result;
        apiDoneRef.current = true;
      })
      .catch((err) => {
        apiErrorRef.current =
          err instanceof Error ? err.message : "Generation failed";
        apiDoneRef.current = true;
      });
  }, [selectedProtein, nCandidates, temperature, stressFactors, dockingEngine, runAdmet, vqeOptimizer, vqeMaxIterations]);

  // Pipeline animation effect
  useEffect(() => {
    if (!isRunning) return;

    if (pipelineStage >= pipelineStages.length) {
      const poll = setInterval(() => {
        if (apiDoneRef.current) {
          clearInterval(poll);
          if (apiErrorRef.current) {
            setRunError(apiErrorRef.current);
            setIsRunning(false);
          } else if (apiResultRef.current) {
            // Save to session context for cross-page access
            saveSession(
              {
                pdb_id: selectedProtein!,
                n_candidates: nCandidates,
                temperature,
                stress_factors: stressFactors,
                docking_engine: dockingEngine,
                run_admet: runAdmet,
                vqe_optimizer: vqeOptimizer,
                vqe_max_iterations: vqeMaxIterations,
              },
              apiResultRef.current
            );
            navigate("/molecules", {
              state: {
                genResult: apiResultRef.current,
                fromExperiment: true,
              },
            });
          }
        }
      }, 200);
      return () => clearInterval(poll);
    }

    const interval = setInterval(() => {
      setStageProgress((prev) => {
        if (prev >= 100) {
          setPipelineStage((s) => s + 1);
          return 0;
        }
        return prev + Math.random() * 15 + 5;
      });
    }, 150);
    return () => clearInterval(interval);
  }, [isRunning, pipelineStage, navigate]);

  return (
    <AppLayout>
      <div className="p-8 space-y-8 max-w-6xl mx-auto">
        <div className="flex flex-col gap-2">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <FlaskConical className="h-6 w-6 text-foreground" />
            New Drug Discovery Experiment
          </h1>
          <p className="text-muted-foreground text-sm">Configure targeting and simulate extreme condition stress states.</p>
        </div>

        {/* Minimal Step Indicator */}
        <div className="flex flex-wrap items-center gap-2 pb-4 border-b border-border">
          {steps.map((step, i) => (
            <div key={step.id} className="flex items-center gap-3">
              <div
                className={`flex h-8 w-8 items-center justify-center rounded-md font-mono text-xs font-semibold transition-colors duration-200 ${
                  step.id === currentStep
                    ? "bg-foreground text-background"
                    : step.id < currentStep
                    ? "bg-muted text-foreground border border-border"
                    : "bg-transparent text-muted-foreground border border-border/50"
                }`}
              >
                {step.id}
              </div>
              <span className={`text-sm ${step.id === currentStep ? "text-foreground font-semibold" : "text-muted-foreground"}`}>
                {step.title}
              </span>
              {i < steps.length - 1 && <ChevronRight className="h-4 w-4 text-muted-foreground/50 mx-2" />}
            </div>
          ))}
        </div>

        {/* Content Area */}
        <div className="min-h-[400px]">
          {currentStep === 1 && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
              <h2 className="text-lg font-semibold">Select Protein Target</h2>
              
              <div className="border border-border bg-card p-5 rounded-xl space-y-3">
                <label className="text-sm font-medium">Enter PDB ID</label>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={customPdb}
                    onChange={(e) => setCustomPdb(e.target.value)}
                    placeholder="e.g., 6LU7"
                    className="flex-1 rounded-lg border border-border bg-background px-4 py-2 text-sm font-mono focus:border-foreground focus:outline-none focus:ring-1 focus:ring-foreground transition-all"
                    onKeyDown={(e) => e.key === "Enter" && handleLoadTarget()}
                  />
                  <Button variant="outline" className="rounded-lg" onClick={handleLoadTarget}>Load Target</Button>
                </div>
              </div>

              <div className="space-y-3">
                <p className="text-sm font-medium">Or select from clinical library:</p>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {proteinTargets.map((p) => (
                    <button
                      key={p.pdb}
                      onClick={() => setSelectedProtein(p.pdb)}
                      className={`text-left p-5 rounded-xl border transition-colors duration-200 ${
                        selectedProtein === p.pdb 
                          ? "border-foreground bg-accent/5" 
                          : "border-border bg-card hover:bg-muted/50"
                      }`}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <span className="font-mono text-sm font-bold">{p.pdb}</span>
                        {selectedProtein === p.pdb && <CheckCircle2 className="h-4 w-4 text-foreground" />}
                      </div>
                      <p className="text-sm font-medium text-foreground">{p.name}</p>
                      <p className="text-xs text-muted-foreground mt-1">{p.disease}</p>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {currentStep === 2 && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Configure Analysis & Target Stress</h2>
                
                {/* Auto Set Button */}
                <Button
                  onClick={handleAutoConfig}
                  disabled={!selectedProtein || autoConfiguring}
                  className="rounded-lg gap-2 font-semibold"
                  style={{
                    background: autoConfiguring
                      ? "hsl(var(--muted))"
                      : "linear-gradient(135deg, hsl(270 70% 55%), hsl(200 85% 50%))",
                    border: "none",
                    boxShadow: autoConfiguring ? "none" : "0 4px 16px -4px hsl(270 70% 55% / 0.4)",
                  }}
                >
                  {autoConfiguring ? (
                    <><Loader2 className="h-4 w-4 animate-spin" /> AI Analyzing...</>
                  ) : (
                    <><Bot className="h-4 w-4" /> Auto Set (AI)</>
                  )}
                </Button>
              </div>

              {/* AI Reasoning Toast */}
              {autoConfigResult && (
                <div className="border border-purple-500/30 bg-purple-500/5 rounded-xl p-4 flex items-start gap-3 animate-in fade-in slide-in-from-top-2 duration-300">
                  <Bot className="h-5 w-5 text-purple-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-semibold text-purple-300">AI Recommendation</p>
                    <p className="text-xs text-muted-foreground mt-1">{autoConfigResult.reasoning}</p>
                  </div>
                  <button onClick={() => setAutoConfigResult(null)} className="text-muted-foreground hover:text-foreground ml-auto text-xs">✕</button>
                </div>
              )}
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border border-border bg-card p-5 rounded-xl space-y-4">
                  <h3 className="font-medium text-sm">Quantum Parameters</h3>
                  <div className="space-y-3">
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">VQE Optimizer</label>
                      <select
                        value={vqeOptimizer}
                        onChange={(e) => setVqeOptimizer(e.target.value)}
                        className="mt-1.5 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:border-foreground focus:outline-none focus:ring-1 focus:ring-foreground transition-colors"
                      >
                        <option value="COBYLA">COBYLA</option>
                        <option value="SPSA">SPSA</option>
                        <option value="L-BFGS-B">L-BFGS-B</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">Max Iterations</label>
                      <input
                        type="number"
                        value={vqeMaxIterations}
                        onChange={(e) => setVqeMaxIterations(Math.max(10, Math.min(1000, parseInt(e.target.value) || 100)))}
                        min={10}
                        max={1000}
                        className="mt-1.5 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm font-mono focus:border-foreground focus:outline-none focus:ring-1 focus:ring-foreground transition-colors"
                      />
                    </div>
                  </div>
                </div>

                <div className="border border-border bg-card p-5 rounded-xl space-y-4">
                  <h3 className="font-medium text-sm">AI Configuration</h3>
                  <div className="space-y-3">
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">Number of Candidates</label>
                      <input
                        type="number"
                        value={nCandidates}
                        onChange={(e) => setNCandidates(Math.max(1, Math.min(100, parseInt(e.target.value) || 20)))}
                        min={1}
                        max={100}
                        className="mt-1.5 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm font-mono focus:border-foreground focus:outline-none focus:ring-1 focus:ring-foreground transition-colors"
                      />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">Docking Engine</label>
                      <select
                        value={dockingEngine}
                        onChange={(e) => setDockingEngine(e.target.value)}
                        className="mt-1.5 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:border-foreground focus:outline-none focus:ring-1 focus:ring-foreground transition-colors"
                      >
                        <option value="autodock_vina">AutoDock Vina</option>
                        <option value="gnina">GNINA (CNN-Based)</option>
                        <option value="none">None</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">
                        Temperature ({temperature.toFixed(1)})
                      </label>
                      <input
                        type="range"
                        min={0.5}
                        max={2.0}
                        step={0.1}
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        className="mt-1.5 w-full accent-foreground"
                      />
                      <div className="flex justify-between text-[10px] text-muted-foreground">
                        <span>Conservative</span>
                        <span>Creative</span>
                      </div>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">ADMET Prediction</label>
                      <select
                        value={runAdmet ? "enabled" : "disabled"}
                        onChange={(e) => setRunAdmet(e.target.value === "enabled")}
                        className="mt-1.5 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:border-foreground focus:outline-none focus:ring-1 focus:ring-foreground transition-colors"
                      >
                        <option value="enabled">Enabled (Full Panel)</option>
                        <option value="disabled">Disabled</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>

              {/* Novelty Feature: Disease Exploration / Structural Stress */}
              <div className="border border-border bg-card rounded-xl p-6">
                <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-6">
                  <div>
                    <h3 className="font-semibold text-foreground flex items-center gap-2">
                      <Activity className="h-4 w-4" /> 
                      Disease Exploration & Structural Stress
                    </h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      Simulate extreme protein variations to discover robust inhibitors.
                    </p>
                  </div>
                  <span className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-foreground text-background whitespace-nowrap">
                    Novelty Feature
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
                  {stressModifiers.map((stress) => {
                    const isActive = stressFactors.includes(stress.id);
                    return (
                      <button
                        key={stress.id}
                        onClick={() => toggleStress(stress.id)}
                        className={`flex flex-col items-center gap-2 p-4 rounded-lg border text-center transition-colors duration-200 ${
                          isActive
                            ? "border-foreground bg-accent/5 text-foreground"
                            : "border-border bg-background hover:bg-muted/50 text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        <stress.icon className={`h-5 w-5 ${isActive ? "text-foreground" : "text-muted-foreground"}`} />
                        <span className="text-xs font-medium">{stress.label}</span>
                      </button>
                    );
                  })}
                </div>

                <div className="pt-5 border-t border-border bg-muted/20 rounded-b-xl -mx-6 -mb-6 px-6 pb-6">
                  <h4 className="text-xs font-semibold text-foreground mb-3 uppercase tracking-wider flex items-center gap-2">
                    <Zap className="h-3 w-3" /> Research Impact
                  </h4>
                  <div className="grid gap-2 sm:grid-cols-3">
                    <div className="text-xs text-muted-foreground flex gap-2">
                      <div className="h-1.5 w-1.5 rounded-full bg-foreground mt-1.5 shrink-0" />
                      Helps in unknown disease exploration
                    </div>
                    <div className="text-xs text-muted-foreground flex gap-2">
                      <div className="h-1.5 w-1.5 rounded-full bg-foreground mt-1.5 shrink-0" />
                      Useful for rare genetic disorders
                    </div>
                    <div className="text-xs text-muted-foreground flex gap-2">
                      <div className="h-1.5 w-1.5 rounded-full bg-foreground mt-1.5 shrink-0" />
                      Enables drug resistance prediction
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {currentStep === 3 && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
              <h2 className="text-lg font-semibold">{isRunning ? "Running Pipeline" : "Ready to Execute"}</h2>
              
              {/* Error display */}
              {runError && (
                <div className="border border-destructive/30 bg-destructive/10 rounded-xl p-4 flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-semibold text-destructive">Pipeline Failed</p>
                    <p className="text-xs text-destructive/80 mt-1">{runError}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-3 rounded-lg"
                      onClick={() => {
                        setRunError(null);
                        launchExperiment();
                      }}
                    >
                      Retry
                    </Button>
                  </div>
                </div>
              )}

              {!isRunning && !runError && (
                <div className="border border-border bg-card p-10 rounded-xl text-center flex flex-col items-center justify-center min-h-[300px]">
                  <div className="h-16 w-16 bg-muted rounded-2xl flex items-center justify-center mb-6 border border-border">
                    <Zap className="h-8 w-8 text-foreground" />
                  </div>
                  <h3 className="text-xl font-bold mb-2">Experiment Setup Complete</h3>
                  <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 text-sm text-muted-foreground mb-2">
                    <span>Target: <strong className="text-foreground">{selectedProtein || "None"}</strong></span>
                    <span>•</span>
                    <span>Candidates: <strong className="text-foreground">{nCandidates}</strong></span>
                    <span>•</span>
                    <span>Temperature: <strong className="text-foreground">{temperature.toFixed(1)}</strong></span>
                    <span>•</span>
                    <span>Docking: <strong className="text-foreground">{dockingEngine === "autodock_vina" ? "AutoDock Vina" : dockingEngine === "gnina" ? "GNINA" : "None"}</strong></span>
                  </div>
                  <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 text-sm text-muted-foreground mb-2">
                    <span>VQE: <strong className="text-foreground">{vqeOptimizer}</strong> ({vqeMaxIterations} iter)</span>
                    <span>•</span>
                    <span>ADMET: <strong className="text-foreground">{runAdmet ? "Enabled" : "Disabled"}</strong></span>
                  </div>
                  {stressFactors.length > 0 && (
                    <p className="text-xs text-muted-foreground mb-6">
                      Stress Factors: {stressFactors.map((s) => stressModifiers.find((m) => m.id === s)?.label).join(", ")}
                    </p>
                  )}
                  <Button size="lg" onClick={launchExperiment} className="rounded-lg h-12 px-8 text-base">
                    <Sparkles className="h-5 w-5 mr-2" /> Launch Experiment
                  </Button>
                </div>
              )}

              {isRunning && (
                <div className="border border-border bg-card p-8 rounded-xl space-y-8">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-mono text-sm text-muted-foreground">STATUS: EXECUTING</span>
                    <span className="font-mono text-sm font-bold">
                      {pipelineStage >= pipelineStages.length
                        ? "Waiting for results..."
                        : `${Math.round((pipelineStage / pipelineStages.length) * 100)}%`}
                    </span>
                  </div>
                  
                  <div className="space-y-6">
                    {pipelineStages.map((stage, i) => {
                      const isDone = i < pipelineStage;
                      const isActive = i === pipelineStage;
                      const isPending = i > pipelineStage;
                      
                      return (
                        <div key={stage.id} className={`flex gap-4 ${isPending ? 'opacity-40' : 'opacity-100'} transition-opacity duration-300`}>
                          <div className="mt-0.5">
                            {isDone ? (
                              <CheckCircle2 className="h-5 w-5 text-foreground" />
                            ) : isActive ? (
                              <Loader2 className="h-5 w-5 text-foreground animate-spin" />
                            ) : (
                              <div className="h-5 w-5 rounded-full border border-border" />
                            )}
                          </div>
                          <div className="flex-1 space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="text-sm font-semibold">{stage.label}</span>
                              {isActive && <span className="font-mono text-xs">{Math.min(Math.round(stageProgress), 100)}%</span>}
                            </div>
                            <p className="text-xs text-muted-foreground">{stage.detail}</p>
                            
                            {isActive && (
                              <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden mt-2">
                                <div 
                                  className="h-full bg-foreground transition-all duration-150 ease-linear"
                                  style={{ width: `${Math.min(stageProgress, 100)}%` }}
                                />
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  
                  {pipelineStage >= pipelineStages.length && (
                    <div className="pt-4 border-t border-border text-center">
                      <span className="text-sm font-bold text-foreground inline-flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" /> Finalizing results...
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="flex justify-between pt-6 border-t border-border">
          <Button 
            variant="outline" 
            disabled={currentStep === 1 || isRunning} 
            onClick={() => setCurrentStep((s) => s - 1)} 
            className="rounded-lg px-6"
          >
            Previous
          </Button>
          <Button 
            disabled={currentStep === 3 || isRunning || (currentStep === 1 && !selectedProtein)} 
            onClick={() => setCurrentStep((s) => s + 1)} 
            className="rounded-lg px-6"
          >
            Continue
          </Button>
        </div>
      </div>
    </AppLayout>
  );
}
