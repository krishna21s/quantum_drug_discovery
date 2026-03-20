"""Quick validity test — writes results to file for clean output."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.char_rnn import CharRNN
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "rnn_pretrained.pt")
rnn = CharRNN.load(ckpt, "cpu")

samples = rnn.sample(200, temperature=1.0, device="cpu")

valid = []
for s in samples:
    mol = Chem.MolFromSmiles(s)
    if mol is not None:
        valid.append(Chem.MolToSmiles(mol, canonical=True))

unique = set(valid)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validity_results.txt")
with open(out, "w") as f:
    f.write("VALIDITY TEST RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write(f"Sampled:  {len(samples)}\n")
    f.write(f"Valid:    {len(valid)}/{len(samples)} = {len(valid)/len(samples)*100:.1f}%\n")
    f.write(f"Unique:   {len(unique)}/{len(valid)} = {len(unique)/max(len(valid),1)*100:.1f}%\n")
    f.write("=" * 50 + "\n\n")
    f.write("Top 15 valid molecules:\n")
    for i, s in enumerate(valid[:15], 1):
        f.write(f"  {i:2d}. {s}\n")

print(f"Results written to {out}")
