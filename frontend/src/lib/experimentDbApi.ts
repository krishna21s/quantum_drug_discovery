/**
 * Experiment DB API — Persistent Storage
 */

const API_BASE = "http://localhost:8000/api/experiments";

export interface SaveExperimentRequest {
  pdb_id: string;
  target_name?: string;
  temperature: number;
  n_candidates: number;
  stress_factors: string[];
  docking_engine: string;
  vqe_optimizer: string;
  vqe_max_iterations: number;
  run_admet: boolean;
  generation_time_s: number;
  n_sampled: number;
  n_valid: number;
  candidates_json: unknown[];
}

export interface ExperimentSummary {
  id: number;
  pdb_id: string;
  target_name: string | null;
  temperature: number;
  n_candidates: number;
  docking_engine: string;
  vqe_optimizer: string;
  generation_time_s: number | null;
  created_at: string;
  candidate_count: number;
}

export interface ExperimentDetail extends ExperimentSummary {
  stress_factors: string[] | null;
  vqe_max_iterations: number;
  run_admet: boolean;
  n_sampled: number | null;
  n_valid: number | null;
  candidates_json: unknown[];
}

export async function saveExperiment(data: SaveExperimentRequest): Promise<{ id: number; message: string }> {
  const res = await fetch(`${API_BASE}/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Save failed: ${res.status}`);
  return res.json();
}

export async function listExperiments(): Promise<ExperimentSummary[]> {
  const res = await fetch(API_BASE);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

export async function getExperiment(id: number): Promise<ExperimentDetail> {
  const res = await fetch(`${API_BASE}/${id}`);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}
