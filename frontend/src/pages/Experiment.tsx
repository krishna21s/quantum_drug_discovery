import AppLayout from "@/components/AppLayout";
import { Button } from "@/components/ui/button";
import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { FlaskConical, ChevronRight, Zap, CheckCircle2, Loader2, Activity, Thermometer, ShieldAlert, GitBranch } from "lucide-react";

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
];

const stressModifiers = [
  { id: "mutation", label: "Point Mutation", icon: GitBranch },
  { id: "folding", label: "Folding Stress", icon: Activity },
  { id: "thermal", label: "Thermal Instability", icon: Thermometer },
  { id: "binding", label: "Binding Site Deformation", icon: ShieldAlert },
];

export default function Experiment() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedProtein, setSelectedProtein] = useState<string | null>(null);
  const [stressFactors, setStressFactors] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [pipelineStage, setPipelineStage] = useState(0);
  const [stageProgress, setStageProgress] = useState(0);

  const toggleStress = (id: string) => {
    setStressFactors(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const launchExperiment = useCallback(() => {
    setIsRunning(true);
    setPipelineStage(0);
    setStageProgress(0);
  }, []);

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
                    placeholder="e.g., 6LU7"
                    className="flex-1 rounded-lg border border-border bg-background px-4 py-2 text-sm font-mono focus:border-foreground focus:outline-none focus:ring-1 focus:ring-foreground transition-all"
                  />
                  <Button variant="outline" className="rounded-lg">Load Target</Button>
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
              <h2 className="text-lg font-semibold">Configure Analysis & Target Stress</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border border-border bg-card p-5 rounded-xl space-y-4">
                  <h3 className="font-medium text-sm">Quantum Parameters</h3>
                  <div className="space-y-3">
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">VQE Optimizer</label>
                      <select className="mt-1.5 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:border-foreground focus:outline-none focus:ring-1 focus:ring-foreground transition-colors">
                        <option>COBYLA</option>
                        <option>SPSA</option>
                        <option>L-BFGS-B</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">Max Iterations</label>
                      <input type="number" defaultValue={100} className="mt-1.5 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm font-mono focus:border-foreground focus:outline-none focus:ring-1 focus:ring-foreground transition-colors" />
                    </div>
                  </div>
                </div>

                <div className="border border-border bg-card p-5 rounded-xl space-y-4">
                  <h3 className="font-medium text-sm">AI Configuration</h3>
                  <div className="space-y-3">
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">Docking Engine</label>
                      <select className="mt-1.5 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:border-foreground focus:outline-none focus:ring-1 focus:ring-foreground transition-colors">
                        <option>AutoDock Vina</option>
                        <option>DiffDock</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">ADMET Prediction</label>
                      <select className="mt-1.5 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:border-foreground focus:outline-none focus:ring-1 focus:ring-foreground transition-colors">
                        <option>Enabled (Full Panel)</option>
                        <option>Disabled</option>
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
              
              {!isRunning ? (
                <div className="border border-border bg-card p-10 rounded-xl text-center flex flex-col items-center justify-center min-h-[300px]">
                  <div className="h-16 w-16 bg-muted rounded-2xl flex items-center justify-center mb-6 border border-border">
                    <Zap className="h-8 w-8 text-foreground" />
                  </div>
                  <h3 className="text-xl font-bold mb-2">Experiment Setup Complete</h3>
                  <div className="flex flex-wrap justify-center gap-2 text-sm text-muted-foreground mb-8">
                    <span>Target: {selectedProtein || "None"}</span>
                    {stressFactors.length > 0 && (
                      <>
                        <span>•</span>
                        <span>Stress Factors: {stressFactors.length}</span>
                      </>
                    )}
                  </div>
                  <Button size="lg" onClick={launchExperiment} className="rounded-lg h-12 px-8 text-base">
                    Launch Experiment
                  </Button>
                </div>
              ) : (
                <div className="border border-border bg-card p-8 rounded-xl space-y-8">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-mono text-sm text-muted-foreground">STATUS: EXECUTING</span>
                    <span className="font-mono text-sm font-bold">{Math.round((pipelineStage / pipelineStages.length) * 100)}%</span>
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
                        <CheckCircle2 className="h-4 w-4" /> Pipeline Complete. Redirecting...
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
