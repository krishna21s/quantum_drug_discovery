/**
 * Simulation Lab API — Aggregation endpoint client
 */

const API_BASE = "http://localhost:8000/api/quantum-lab";

export interface EGFRSummary {
  target_name: string;
  pdb_id: string;
  diseases: string[];
  approved_drugs: string[];
  investigational_drugs: string[];
  clinical_trials_count: number;
}

export interface BindingSummary {
  oracle_loaded: boolean;
  scoring_mode: string | null;
  sample_xgb_pic50: number | null;
  sample_qsvr_pic50: number | null;
  sample_smiles: string | null;
}

export interface ToxicitySummary {
  service_loaded: boolean;
  model_type: string;
  features_count: number;
}

export interface ADMETSummary {
  service_loaded: boolean;
  categories: string[];
}

export interface VQESummary {
  ground_state_energy: number;
  unit: string;
  qubits: number;
  circuit_depth: number;
  gate_count: number;
  parameters: number;
  optimizer: string;
  ansatz: string;
  backend: string;
  convergence_iterations: number;
}

export interface VQCSummary {
  qubit_count: number;
  gate_types: string[];
  total_gates: number;
  circuit_depth: number;
  measurement_qubits: number;
}

export interface LabSummary {
  egfr: EGFRSummary;
  binding: BindingSummary;
  toxicity: ToxicitySummary;
  admet: ADMETSummary;
  vqe: VQESummary;
  vqc: VQCSummary;
  latency_ms: number;
}

export async function fetchLabSummary(): Promise<LabSummary> {
  const res = await fetch(`${API_BASE}/summary`);
  if (!res.ok) throw new Error(`Lab summary failed: ${res.status}`);
  return res.json();
}
