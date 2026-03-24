/**
 * Experiment API Client
 * ======================
 * Client for the experiment auto-configure endpoint (LLM-powered).
 */

const API_BASE = "/api/experiment";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Experiment API error: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export interface AutoConfigResponse {
  pdb_id: string;
  protein_name: string;
  disease: string;
  temperature: number;
  n_candidates: number;
  vqe_optimizer: string;
  vqe_max_iterations: number;
  docking_engine: string;
  stress_factors: string[];
  run_admet: boolean;
  reasoning: string;
}

/** Use LLM to auto-configure experiment parameters based on protein target. */
export async function autoConfigureExperiment(pdb_id: string): Promise<AutoConfigResponse> {
  const res = await fetch(`${API_BASE}/auto-configure`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pdb_id }),
  });
  return handleResponse<AutoConfigResponse>(res);
}
