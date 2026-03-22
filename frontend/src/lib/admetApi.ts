/**
 * ADMET API Client
 * ================
 * Fetch-based client for interacting with the ADMET prediction endpoints.
 */

const API_BASE = "/api/admet";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `ADMET API error: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export interface ADMETData {
  absorption: number;
  distribution: number;
  metabolism: number;
  excretion: number;
  overall: number;
  verdict: string;
  candidate_id: number | null;
}

export interface ADMETBatchResult {
  processed: number;
  skipped: number;
  failed: number;
  message: string;
}

/** Generate ADMET predictions for a single SMILES string. */
export async function generateADMET(smiles: string): Promise<ADMETData> {
  const res = await fetch(`${API_BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles }),
  });
  return handleResponse<ADMETData>(res);
}

/** Fill missing ADMET data for all candidates in the database. */
export async function fillMissingADMET(): Promise<ADMETBatchResult> {
  const res = await fetch(`${API_BASE}/fill-missing`, {
    method: "POST",
  });
  return handleResponse<ADMETBatchResult>(res);
}

/** Get ADMET data for a specific candidate by ID. */
export async function getADMETByCandidate(candidateId: number): Promise<ADMETData> {
  const res = await fetch(`${API_BASE}/${candidateId}`);
  return handleResponse<ADMETData>(res);
}
