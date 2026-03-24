/**
 * VQC Circuit API — Real quantum circuit generation
 */

const API_BASE = "http://localhost:8000/api/vqc";

export interface GateInfo {
  type: string;
  qubit: number;
  col: number;
  target?: number;
  angle?: number;
  label?: string;
}

export interface CircuitResponse {
  smiles: string;
  n_qubits: number;
  n_layers: number;
  circuit_depth: number;
  total_gates: number;
  total_parameters: number;
  gates: GateInfo[];
  feature_vector: number[];
  gate_type_counts: Record<string, number>;
  molecular_properties: Record<string, number>;
}

export async function generateCircuit(smiles: string, n_qubits = 8, n_layers = 2): Promise<CircuitResponse> {
  const res = await fetch(`${API_BASE}/circuit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles, n_qubits, n_layers }),
  });
  if (!res.ok) throw new Error(`Circuit generation failed: ${res.status}`);
  return res.json();
}
