/**
 * Toxicity Prediction API Client
 * ================================
 * Typed fetch-based client for the FastAPI toxicity prediction backend.
 */

const API_BASE = "/api";

// ── Types ──────────────────────────────────────────────────────────

export interface TimingInfo {
  xgb_ms: number;
  quantum_s: number;
  total_s: number;
  shot_s: number | null;
}

export interface ConfidenceInterval {
  probability: number;
  std: number;
  ci_lower: number;
  ci_upper: number;
  n_bootstrap: number;
}

export interface PredictResponse {
  smiles: string;
  canonical_smiles: string | null;
  classical_probability: number;
  quantum_probability: number;
  ensemble_probability: number;
  baseline_score: number;
  verdict: string;
  confidence: number;
  timings: TimingInfo;
  quantum_cached: boolean;
  ci: ConfidenceInterval | null;
  mode: string;
}

export interface BatchPredictResponse {
  predictions: PredictResponse[];
  summary: Record<string, unknown>;
  total_time_s: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: string[];
  pipeline_ready: boolean;
  checkpoint_dir: string;
}

// ── API Functions ──────────────────────────────────────────────────

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `API error: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function predictToxicity(
  smiles: string,
  enableCI = false,
  nBootstrap = 5,
): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      smiles,
      enable_ci: enableCI,
      n_bootstrap: nBootstrap,
    }),
  });
  return handleResponse<PredictResponse>(res);
}

export async function predictBatch(
  smilesList: string[],
): Promise<BatchPredictResponse> {
  const res = await fetch(`${API_BASE}/predict/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles_list: smilesList }),
  });
  return handleResponse<BatchPredictResponse>(res);
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  return handleResponse<HealthResponse>(res);
}

// ── Organ Impact Prediction ────────────────────────────────────────

export interface OrganEffect {
  name: string;
  reason: string;
  confidence: number;
}

export interface OrganImpactResponse {
  smiles: string;
  canonical_smiles: string;
  target_organs: OrganEffect[];
  side_effect_organs: OrganEffect[];
  drug_class: string;
  mechanism_summary: string;
}

export async function predictOrganImpact(
  smiles: string,
): Promise<OrganImpactResponse> {
  const res = await fetch(`${API_BASE}/predict/organ-impact`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles }),
  });
  return handleResponse<OrganImpactResponse>(res);
}
