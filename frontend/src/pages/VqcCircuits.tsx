import AppLayout from "@/components/AppLayout";
import { useState, useCallback } from "react";
import { CircuitBoard, Loader2, AlertTriangle, Atom } from "lucide-react";
import { Button } from "@/components/ui/button";
import { generateCircuit, type CircuitResponse, type GateInfo } from "@/lib/vqcApi";
import { useExperiment } from "@/context/ExperimentContext";
import { cn } from "@/lib/utils";

const EXAMPLE_MOLECULES = [
  { name: "Aspirin", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O" },
  { name: "Ibuprofen", smiles: "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" },
  { name: "Caffeine", smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" },
  { name: "Paracetamol", smiles: "CC(=O)NC1=CC=C(O)C=C1" },
];

// ── SVG Circuit Renderer ──
function VqcSvg({ gates, nQubits, depth }: { gates: GateInfo[]; nQubits: number; depth: number }) {
  const gateW = 44;
  const gateH = 30;
  const rowH = 52;
  const startX = 60;
  const startY = 35;
  const colSpacing = gateW + 18;
  const svgW = startX + depth * colSpacing + 30;
  const svgH = startY + nQubits * rowH + 10;

  return (
    <svg width="100%" viewBox={`0 0 ${svgW} ${svgH}`} className="overflow-visible" style={{ minWidth: 500 }}>
      {Array.from({ length: nQubits }).map((_, i) => (
        <g key={`q${i}`}>
          <text x={8} y={startY + i * rowH + 4} fill="hsl(var(--muted-foreground))" fontSize="10" fontFamily="monospace" fontWeight="600">q[{i}]</text>
          <line x1={startX - 8} y1={startY + i * rowH} x2={startX + depth * colSpacing} y2={startY + i * rowH} stroke="hsl(var(--border))" strokeWidth={1} strokeOpacity={0.4} />
        </g>
      ))}
      {gates.map((g, idx) => {
        const x = startX + g.col * colSpacing;
        const y = startY + g.qubit * rowH;

        if (g.type === "CZ" && g.target != null) {
          const ty = startY + g.target * rowH;
          return (
            <g key={idx}>
              <line x1={x + gateW / 2} y1={y} x2={x + gateW / 2} y2={ty} stroke="hsl(var(--foreground))" strokeWidth={1.2} />
              <circle cx={x + gateW / 2} cy={y} r={3.5} fill="hsl(var(--foreground))" />
              <circle cx={x + gateW / 2} cy={ty} r={3.5} fill="hsl(var(--foreground))" />
            </g>
          );
        }

        const isM = g.type === "M";
        const fillColor = isM ? "hsl(var(--muted)/0.5)" : g.type === "H" ? "hsl(260 60% 50% / 0.15)" : "hsl(var(--primary)/0.12)";
        const strokeColor = isM ? "hsl(var(--border))" : g.type === "H" ? "hsl(260 60% 55%)" : "hsl(var(--primary))";

        return (
          <g key={idx}>
            <rect x={x} y={y - gateH / 2} width={gateW} height={gateH} rx={5} fill={fillColor} stroke={strokeColor} strokeWidth={0.8} />
            <text x={x + gateW / 2} y={y + 3} textAnchor="middle" fill="hsl(var(--foreground))" fontSize="9" fontFamily="monospace" fontWeight="700">
              {g.type}
            </text>
            {g.angle != null && (
              <text x={x + gateW / 2} y={y + gateH / 2 + 10} textAnchor="middle" fill="hsl(var(--muted-foreground))" fontSize="6" fontFamily="monospace">
                {g.angle.toFixed(2)}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

export default function VqcPage() {
  const { session } = useExperiment();
  const [smiles, setSmiles] = useState("");
  const [loading, setLoading] = useState(false);
  const [circuit, setCircuit] = useState<CircuitResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runCircuit = useCallback(async (s: string) => {
    if (!s.trim()) return;
    setLoading(true);
    setError(null);
    setCircuit(null);
    try {
      const res = await generateCircuit(s.trim());
      setCircuit(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Circuit generation failed");
    } finally {
      setLoading(false);
    }
  }, []);

  // Build candidate list from session for quick-select
  const sessionCandidates = (session?.result?.candidates ?? []).slice(0, 6);

  return (
    <AppLayout>
      <div className="min-h-screen p-6 lg:p-8 max-w-[1400px] mx-auto space-y-6">
        <div className="border-b border-border pb-6">
          <div className="inline-flex items-center gap-2 text-primary font-semibold text-sm mb-1">
            <CircuitBoard className="h-4 w-4" /> Quantum Circuits
          </div>
          <h1 className="text-2xl font-bold">VQC Circuit Diagrams</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Generate the real QSVR data-reuploading variational quantum circuit for any SMILES molecule.
            Circuit uses molecule-specific angles derived from RDKit descriptors.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Input Panel */}
          <div className="glass-card rounded-2xl p-5 space-y-4">
            <h3 className="font-semibold text-sm flex items-center gap-2">
              <Atom className="h-4 w-4" /> Molecule Input
            </h3>
            <textarea
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="Enter SMILES string..."
              className="w-full rounded-xl border border-border bg-background px-4 py-3 text-sm font-mono focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary/30 transition-all resize-none h-24"
            />
            <Button
              onClick={() => runCircuit(smiles)}
              disabled={!smiles.trim() || loading}
              className="w-full rounded-xl font-semibold h-10"
            >
              {loading ? (
                <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Generating...</>
              ) : (
                <><CircuitBoard className="h-4 w-4 mr-2" />Generate Circuit</>
              )}
            </Button>

            {/* Example molecules */}
            <div className="space-y-1.5">
              <p className="text-[10px] text-muted-foreground uppercase font-semibold">Examples</p>
              {EXAMPLE_MOLECULES.map((mol) => (
                <button
                  key={mol.name}
                  onClick={() => { setSmiles(mol.smiles); runCircuit(mol.smiles); }}
                  className={cn(
                    "w-full text-left px-3 py-2 rounded-lg border text-xs transition-colors",
                    smiles === mol.smiles ? "border-primary bg-primary/5" : "border-border hover:bg-muted/30"
                  )}
                >
                  <p className="font-semibold">{mol.name}</p>
                  <p className="font-mono text-[10px] text-muted-foreground truncate">{mol.smiles}</p>
                </button>
              ))}
            </div>

            {/* Session candidates */}
            {sessionCandidates.length > 0 && (
              <div className="space-y-1.5 pt-2 border-t border-border">
                <p className="text-[10px] text-muted-foreground uppercase font-semibold">From Current Experiment</p>
                {sessionCandidates.map((c, i) => (
                  <button
                    key={i}
                    onClick={() => { setSmiles(c.smiles); runCircuit(c.smiles); }}
                    className={cn(
                      "w-full text-left px-3 py-2 rounded-lg border text-xs transition-colors",
                      smiles === c.smiles ? "border-primary bg-primary/5" : "border-border hover:bg-muted/30"
                    )}
                  >
                    <p className="font-semibold">Candidate #{c.rank || i + 1}</p>
                    <p className="font-mono text-[10px] text-muted-foreground truncate">{c.smiles}</p>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-3 space-y-5">
            {loading && (
              <div className="glass-card rounded-2xl p-12 text-center">
                <Loader2 className="h-8 w-8 text-primary animate-spin mx-auto mb-3" />
                <p className="font-semibold">Generating quantum circuit...</p>
                <p className="text-xs text-muted-foreground mt-1">Computing RDKit features → building data-reuploading circuit</p>
              </div>
            )}
            {error && (
              <div className="glass-card rounded-2xl p-8 text-center ring-1 ring-destructive/30">
                <AlertTriangle className="h-8 w-8 text-destructive mx-auto mb-2" />
                <p className="text-destructive font-semibold">{error}</p>
              </div>
            )}
            {circuit && !loading && (
              <>
                {/* Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {[
                    { label: "Qubits", value: circuit.n_qubits },
                    { label: "Total Gates", value: circuit.total_gates },
                    { label: "Circuit Depth", value: circuit.circuit_depth },
                    { label: "Trainable Params", value: circuit.total_parameters },
                  ].map((item) => (
                    <div key={item.label} className="glass-card rounded-xl p-4 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase">{item.label}</p>
                      <p className="text-2xl font-black mt-1">{item.value}</p>
                    </div>
                  ))}
                </div>

                {/* Circuit SVG */}
                <div className="glass-card rounded-2xl p-5 overflow-x-auto">
                  <h4 className="text-xs text-muted-foreground font-semibold mb-3 uppercase">Circuit Diagram — {circuit.smiles.slice(0, 30)}{circuit.smiles.length > 30 ? "..." : ""}</h4>
                  <VqcSvg gates={circuit.gates} nQubits={circuit.n_qubits} depth={circuit.circuit_depth} />
                </div>

                {/* Gate Counts + Feature Vector */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <div className="glass-card rounded-2xl p-5">
                    <p className="text-xs text-muted-foreground font-semibold mb-3">Gate Type Counts</p>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(circuit.gate_type_counts).map(([type, count]) => (
                        <span key={type} className="px-3 py-1.5 rounded-lg text-xs font-mono font-bold bg-primary/10 border border-primary/30">
                          {type}: {count}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="glass-card rounded-2xl p-5">
                    <p className="text-xs text-muted-foreground font-semibold mb-3">Feature Vector (Arctan Normalized)</p>
                    <div className="grid grid-cols-4 gap-1.5">
                      {circuit.feature_vector.map((v, i) => (
                        <span key={i} className="px-2 py-1.5 rounded text-[10px] font-mono bg-background border border-border text-center">
                          x[{i}]={v.toFixed(4)}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </>
            )}
            {!circuit && !loading && !error && (
              <div className="glass-card rounded-2xl p-12 text-center border-dashed border-2 border-border">
                <CircuitBoard className="h-10 w-10 text-muted-foreground mx-auto mb-4" />
                <p className="font-semibold text-lg">Enter a SMILES String</p>
                <p className="text-xs text-muted-foreground mt-2 max-w-md mx-auto">
                  The real QSVR data-reuploading circuit will be generated with molecule-specific
                  rotation angles derived from RDKit molecular descriptors (MW, LogP, TPSA, HBD, HBA, etc.)
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
