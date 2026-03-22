/**
 * Lead Optimization API Client
 * ==============================
 * Fetch-based client for the /api/refine endpoint.
 */

const API_BASE = "/api";

export interface MoleculeScores {
  smiles: string;
  // Binding — dual oracle
  xgb_pic50: number;
  qsvr_pic50: number;
  ensemble_pic50: number;    // 0.55*XGB + 0.45*QSVR
  // ADMET
  admet_overall: number;
  admet_absorption: number;
  admet_distribution: number;
  admet_metabolism: number;
  admet_excretion: number;
  admet_verdict: string;
  // Toxicity — dual oracle
  toxicity_xgb: number;      // classical XGB
  toxicity_quantum: number;  // quantum SVM
  toxicity_ensemble: number; // avg(XGB, QSVM)
  toxicity_prob: number;
  // Drug-likeness
  qed_score: number;
  sa_score: number;
  mw: number;
  composite_reward: number;
}

export interface RefinementStep {
  step: number;
  smiles: string;
  scores: MoleculeScores;
  delta_reward: number;
  variants_evaluated: number;
  accepted: boolean;
  mutation_type: string;
}

export interface RefinementResult {
  original_smiles: string;
  final_smiles: string;
  trajectory: RefinementStep[];
  total_steps: number;
  total_improvement: number;
  elapsed_seconds: number;
  converged: boolean;
}

export async function refineMolecule(
  smiles: string,
  maxSteps: number = 5,
  preserveScaffold: boolean = true,
): Promise<RefinementResult> {
  const res = await fetch(`${API_BASE}/refine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      smiles,
      max_steps: maxSteps,
      variants_per_step: 15,
      preserve_scaffold: preserveScaffold,
    }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Refinement failed: ${res.status}`);
  }
  return res.json() as Promise<RefinementResult>;
}
