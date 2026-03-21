"""Score the 50 RL candidates with both XGB and QSVR oracles."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle.quantum_oracle import QuantumOracle

# Load candidates
script_dir = os.path.dirname(os.path.abspath(__file__))
cand_path = os.path.join(script_dir, "checkpoints", "final_candidates.json")
with open(cand_path) as f:
    data = json.load(f)

oracle = QuantumOracle()
print(f"\nScoring {len(data['candidates'])} candidates...\n")

for i, cand in enumerate(data["candidates"]):
    result = oracle.score(cand["smiles"])
    cand["quantum_pic50"] = result["pic50"]
    cand["scoring_mode"] = result["mode"]
    xgb = cand["xgb_pic50"]
    qsvr = result["pic50"]
    if i < 10:
        print(f"  #{cand['rank']:2d}  XGB={xgb:.2f}  QSVR={qsvr:.2f}  Δ={qsvr-xgb:+.2f}  {cand['smiles'][:50]}")

# Save updated candidates
with open(cand_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"\n  ✓ Saved {len(data['candidates'])} candidates with QSVR scores")

# Summary stats
qsvr_scores = [c["quantum_pic50"] for c in data["candidates"] if c["quantum_pic50"] is not None]
print(f"  QSVR range: {min(qsvr_scores):.2f} – {max(qsvr_scores):.2f}  mean={sum(qsvr_scores)/len(qsvr_scores):.2f}")
