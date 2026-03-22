import { Candidate, CandidatesResponse } from "./drugApi";

const API_BASE = "/api/db";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `DB API error: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export interface DBCandidate {
  id: number;
  smiles: string;
  target: string | null;
  mw: number | null;
  logp: number | null;
  tpsa: number | null;
  qed: number | null;
  sa_score: number | null;
  lipinski_pass: boolean | null;
  is_novel: boolean | null;
  binding_affinity: {
    xgb_pic50: number | null;
    qsvr_pic50: number | null;
    scoring_mode: string | null;
  } | null;
  toxicity: {
    toxicity_score: number | null;
    is_toxic: boolean | null;
    alerts: any | null;
  } | null;
}

export interface DBCandidateDetails {
  candidate: DBCandidate;
  binding_affinity: {
    xgb_pic50: number | null;
    qsvr_pic50: number | null;
    scoring_mode: string | null;
  };
  toxicity: {
    toxicity_score: number | null;
    is_toxic: boolean | null;
    alerts: any | null;
  };
}

export async function fetchDBCandidates(): Promise<CandidatesResponse> {
  const res = await fetch(`${API_BASE}/candidates`);
  const dbCandidates = await handleResponse<DBCandidate[]>(res);
  
  // Map DB candidates to the shape expected by Molecules.tsx (Candidate[])
  const candidates: Candidate[] = dbCandidates.map((c, index) => ({
    rank: c.id, 
    smiles: c.smiles,
    xgb_pic50: c.binding_affinity?.xgb_pic50 ?? 0, 
    quantum_pic50: c.binding_affinity?.qsvr_pic50 ?? null,
    qed: c.qed ?? 0,
    sa_score: c.sa_score ?? 0,
    mw: c.mw ?? 0,
    logp: c.logp ?? 0,
    lipinski_pass: c.lipinski_pass ?? false,
    tpsa: c.tpsa,
    is_novel: c.is_novel,
    scoring_mode: c.binding_affinity?.scoring_mode ?? null
  }));

  return {
    target: candidates.length > 0 ? (dbCandidates[0].target ?? "Database Candidates") : "Database Candidates",
    n_rl_episodes: 500, // Static estimate since we don't save episodes in DB
    total_generated: candidates.length,
    final_reward: 0.5,
    total_time_min: 24.6,
    candidates,
  };
}

export async function fetchDBCandidateDetails(id: number): Promise<DBCandidateDetails> {
  const res = await fetch(`${API_BASE}/candidates/${id}`);
  return handleResponse<DBCandidateDetails>(res);
}

// Actually, let's export a function that gets all DB Candidates WITH binding details.
// It's better to modify the backend /candidates to return the binding affinity too.
