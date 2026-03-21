/**
 * Drug Discovery API Client
 * ===========================
 * Typed fetch-based client for candidates and binding affinity endpoints.
 */

const API_BASE = "/api";

// ── Types ──────────────────────────────────────────────────────────

export interface Candidate {
  rank: number;
  smiles: string;
  xgb_pic50: number;
  quantum_pic50: number | null;
  qed: number;
  sa_score: number;
  mw: number;
  logp: number;
  lipinski_pass: boolean;
  tpsa: number | null;
  is_novel: boolean | null;
  scoring_mode: string | null;
}

export interface CandidatesResponse {
  target: string;
  n_rl_episodes: number;
  total_generated: number;
  final_reward: number;
  total_time_min: number;
  candidates: Candidate[];
}

export interface BindingScoreResponse {
  smiles: string;
  canonical_smiles: string | null;
  xgb_pic50: number | null;
  qsvr_pic50: number | null;
  scoring_mode: string;
  latency_s: number;
  error: string | null;
}

export interface BatchBindingResponse {
  predictions: BindingScoreResponse[];
  summary: Record<string, unknown>;
  total_time_s: number;
}

export interface GenerateRequest {
  pdb_id?: string;
  n_candidates?: number;
  temperature?: number;
  max_mw?: number;
}

export interface GenerateResponse {
  target: string;
  n_requested: number;
  n_sampled: number;
  n_valid: number;
  temperature: number;
  generation_time_s: number;
  candidates: Candidate[];
}

// ── Helpers ────────────────────────────────────────────────────────

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `API error: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── API Functions ──────────────────────────────────────────────────

export async function fetchCandidates(): Promise<CandidatesResponse> {
  const res = await fetch(`${API_BASE}/candidates`);
  return handleResponse<CandidatesResponse>(res);
}

export async function fetchCandidateByRank(rank: number): Promise<Candidate> {
  const res = await fetch(`${API_BASE}/candidates/${rank}`);
  return handleResponse<Candidate>(res);
}

export async function scoreBinding(smiles: string): Promise<BindingScoreResponse> {
  const res = await fetch(`${API_BASE}/binding/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles }),
  });
  return handleResponse<BindingScoreResponse>(res);
}

export async function scoreBindingBatch(
  smilesList: string[],
): Promise<BatchBindingResponse> {
  const res = await fetch(`${API_BASE}/binding/score/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles_list: smilesList }),
  });
  return handleResponse<BatchBindingResponse>(res);
}

export async function generateCandidates(
  opts: GenerateRequest = {},
): Promise<GenerateResponse> {
  const res = await fetch(`${API_BASE}/candidates/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
  return handleResponse<GenerateResponse>(res);
}
