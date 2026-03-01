"""
GNN Training Script — Construction V2
========================================
Trains a Graph Isomorphism Network (GIN) on Tox21 NR-AR for:
  1. Binary toxicity classification
  2. Dense molecular embeddings (128-d) for quantum kernel input

Uses the V2 GraphService for SMILES → graph conversion.

Usage:
    cd construction_v2
    ..\\venv\\Scripts\\python.exe training/train_gnn.py

Outputs (in ./checkpoints/):
    gnn_model.pt              — Trained GIN weights
    gnn_projector.pkl         — PCA projector (128-d → 20-d for quantum)
    gnn_embeddings_train.npy  — Cached train embeddings
    gnn_training_report.json  — AUC, Brier, loss curves, comparison vs XGB

Requirements:
    pip install torch torch_geometric

NOTE: If torch_geometric is not installed, this script will give
      clear instructions on how to install it for your platform.
"""

import os
import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd

# Ensure construction_v2 root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHECKPOINT_DIR,
    TOX21_URL,
    TOX21_ENDPOINT,
    RANDOM_STATE,
    GNN_EMBEDDING_DIM,
    GNN_HIDDEN_DIM,
    GNN_NUM_LAYERS,
    GNN_PROJECTION_DIM,
    REFERENCE_MOLECULES,
)

warnings.filterwarnings("ignore")

# ================================================================
# CHECK DEPENDENCIES
# ================================================================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
except ImportError:
    print("=" * 65)
    print(" ❌ PyTorch not found!")
    print("    Install: pip install torch")
    print("=" * 65)
    sys.exit(1)

try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
except ImportError:
    print("=" * 65)
    print(" ❌ torch_geometric not found!")
    print("    Install for your platform:")
    print("    pip install torch_geometric")
    print(
        "    pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html"
    )
    print("=" * 65)
    sys.exit(1)

from services.graph_service import GraphService
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# ================================================================
# CONFIGURATION
# ================================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 100
PATIENCE = 15  # Early stopping patience
DROPOUT = 0.3
NUM_WORKERS = 0  # 0 for Windows compatibility

print("=" * 65)
print(" 🚀 GNN (GIN) Training Pipeline — Construction V2")
print(
    f"    Architecture: GIN  |  Layers: {GNN_NUM_LAYERS}  |  Hidden: {GNN_HIDDEN_DIM}"
)
print(
    f"    Embedding: {GNN_EMBEDDING_DIM}-d  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}"
)
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Device: {device}")


# ================================================================
# 1. GIN MODEL DEFINITION
# ================================================================
class MLP(nn.Module):
    """2-layer MLP used inside each GIN layer."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=DROPOUT):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(self.lin2(x))
        return x


class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) for molecular property prediction.

    Produces:
      - embedding: dense 128-d molecular representation (from pooling)
      - logit: binary classification score (toxicity)
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=GNN_HIDDEN_DIM,
        embed_dim=GNN_EMBEDDING_DIM,
        num_layers=GNN_NUM_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GIN convolution layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(hidden_dim, hidden_dim * 2, hidden_dim, dropout)
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Embedding head (graph-level)
        self.embed_head = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection
        x = F.relu(self.input_proj(x))

        # Message passing
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling → graph-level embedding
        graph_repr = global_add_pool(x, batch)

        # Embedding
        embedding = self.embed_head(graph_repr)

        # Classification
        logit = self.classifier(embedding).squeeze(-1)

        return logit, embedding

    def get_embedding(self, data):
        """Get embedding only (for inference)."""
        self.eval()
        with torch.no_grad():
            _, embedding = self.forward(data)
        return embedding.cpu().numpy()


# ================================================================
# 2. LOAD DATA
# ================================================================
print("\n[1/6] Loading Tox21 NR-AR dataset...")
df = pd.read_csv(TOX21_URL).dropna(subset=[TOX21_ENDPOINT])
print(f"      Total molecules: {len(df)}")

toxic_count = (df[TOX21_ENDPOINT] == 1).sum()
safe_count = (df[TOX21_ENDPOINT] == 0).sum()
print(f"      Toxic: {toxic_count}  |  Safe: {safe_count}")

# Stratified split
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df[TOX21_ENDPOINT],
)
train_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=train_df[TOX21_ENDPOINT],
)
print(f"      Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")


# ================================================================
# 3. CONVERT SMILES TO GRAPHS
# ================================================================
print("\n[2/6] Converting SMILES to graphs via GraphService...")
graph_svc = GraphService()
t0 = time.time()


def build_dataset(dataframe, label_col=TOX21_ENDPOINT):
    """Convert a DataFrame of SMILES+labels to a list of PyG Data objects."""
    graphs = []
    skipped = 0
    for _, row in dataframe.iterrows():
        g = graph_svc.smiles_to_graph(row["smiles"])
        if g is None:
            skipped += 1
            continue
        g.y = torch.tensor([float(row[label_col])], dtype=torch.float)
        graphs.append(g)
    if skipped > 0:
        print(f"      Skipped {skipped} invalid SMILES")
    return graphs


train_graphs = build_dataset(train_df)
val_graphs = build_dataset(val_df)
test_graphs = build_dataset(test_df)
print(f"      Graphs built in {time.time() - t0:.1f}s")
print(
    f"      Train: {len(train_graphs)}  |  Val: {len(val_graphs)}  |  Test: {len(test_graphs)}"
)

# Compute class weight for imbalanced data
n_pos = sum(1 for g in train_graphs if g.y.item() == 1.0)
n_neg = len(train_graphs) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
print(f"      pos_weight: {pos_weight.item():.2f}")

# Feature dim from first graph
atom_feat_dim = train_graphs[0].x.shape[1]
print(f"      Atom feature dim: {atom_feat_dim}")

# DataLoaders
train_loader = DataLoader(
    train_graphs, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)


# ================================================================
# 4. TRAIN MODEL
# ================================================================
print(f"\n[3/6] Training GIN model ({EPOCHS} epochs, patience={PATIENCE})...")

model = GINEncoder(in_dim=atom_feat_dim).to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, verbose=False
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

total_params = sum(p.numel() for p in model.parameters())
print(f"      Model parameters: {total_params:,}")

best_val_auc = 0.0
best_epoch = 0
patience_counter = 0
history = {"train_loss": [], "val_auc": [], "val_loss": []}

for epoch in range(1, EPOCHS + 1):
    # --- TRAIN ---
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = criterion(logits, batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    train_loss = total_loss / len(train_graphs)
    history["train_loss"].append(train_loss)

    # --- VALIDATE ---
    model.eval()
    val_preds = []
    val_labels = []
    val_loss_total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            probs = torch.sigmoid(logits)
            val_preds.extend(probs.cpu().numpy())
            val_labels.extend(batch.y.squeeze().cpu().numpy())
            val_loss_total += (
                criterion(logits, batch.y.squeeze()).item() * batch.num_graphs
            )

    val_loss = val_loss_total / len(val_graphs)
    history["val_loss"].append(val_loss)

    try:
        val_auc = roc_auc_score(val_labels, val_preds)
    except Exception:
        val_auc = 0.5

    history["val_auc"].append(val_auc)
    scheduler.step(val_auc)

    current_lr = optimizer.param_groups[0]["lr"]

    if epoch % 5 == 0 or epoch == 1:
        print(
            f"      Epoch {epoch:3d}/{EPOCHS}  "
            f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_AUC={val_auc:.4f}  lr={current_lr:.1e}"
        )

    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), str(CHECKPOINT_DIR / "gnn_model.pt"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n      Early stopping at epoch {epoch} (best: {best_epoch})")
            break

print(f"\n      Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")


# ================================================================
# 5. EVALUATE ON TEST SET
# ================================================================
print("\n[4/6] Evaluating on test set...")

# Load best model
model.load_state_dict(
    torch.load(str(CHECKPOINT_DIR / "gnn_model.pt"), map_location=device)
)
model.eval()

test_preds = []
test_labels = []
test_embeddings = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits, embeddings = model(batch)
        probs = torch.sigmoid(logits)
        test_preds.extend(probs.cpu().numpy())
        test_labels.extend(batch.y.squeeze().cpu().numpy())
        test_embeddings.extend(embeddings.cpu().numpy())

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)
test_embeddings = np.array(test_embeddings)

test_auc = roc_auc_score(test_labels, test_preds)
test_brier = brier_score_loss(test_labels, test_preds)
test_pred_binary = (test_preds >= 0.5).astype(int)
report = classification_report(
    test_labels.astype(int), test_pred_binary, output_dict=True
)

print(f"      Test ROC-AUC : {test_auc:.4f}")
print(f"      Test Brier   : {test_brier:.4f}")
if "1" in report:
    print(f"      Precision (toxic): {report['1']['precision']:.3f}")
    print(f"      Recall    (toxic): {report['1']['recall']:.3f}")
    print(f"      F1        (toxic): {report['1']['f1-score']:.3f}")
elif "1.0" in report:
    print(f"      Precision (toxic): {report['1.0']['precision']:.3f}")
    print(f"      Recall    (toxic): {report['1.0']['recall']:.3f}")
    print(f"      F1        (toxic): {report['1.0']['f1-score']:.3f}")


# ================================================================
# 6. EXPORT EMBEDDINGS & PCA PROJECTOR
# ================================================================
print("\n[5/6] Exporting embeddings and PCA projector...")

# Generate all training embeddings
train_embeddings = []
model.eval()
train_loader_full = DataLoader(
    train_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

with torch.no_grad():
    for batch in train_loader_full:
        batch = batch.to(device)
        _, emb = model(batch)
        train_embeddings.extend(emb.cpu().numpy())

train_embeddings = np.array(train_embeddings)
np.save(str(CHECKPOINT_DIR / "gnn_embeddings_train.npy"), train_embeddings)
print(f"      Train embeddings: {train_embeddings.shape}")

# Fit PCA projector (128-d → 20-d for quantum kernel)
pca = PCA(n_components=GNN_PROJECTION_DIM, random_state=RANDOM_STATE)
pca.fit(train_embeddings)
explained = sum(pca.explained_variance_ratio_)
print(
    f"      PCA {GNN_EMBEDDING_DIM}-d → {GNN_PROJECTION_DIM}-d  ({explained:.1%} variance explained)"
)

with open(str(CHECKPOINT_DIR / "gnn_projector.pkl"), "wb") as f:
    pickle.dump(pca, f)

# Reference molecule predictions
print("\n  ── Reference Molecule Validation ──")
for name, (smiles, true_label) in REFERENCE_MOLECULES.items():
    g = graph_svc.smiles_to_graph(smiles)
    if g is None:
        print(f"  ✗  {name:<28} → INVALID SMILES")
        continue
    g.y = torch.tensor([0.0], dtype=torch.float)
    batch = next(iter(DataLoader([g], batch_size=1)))
    batch = batch.to(device)
    with torch.no_grad():
        logit, _ = model(batch)
        prob = torch.sigmoid(logit).item()
    status = "✓" if (prob > 0.5) == bool(true_label) else "✗"
    print(
        f"  {status}  {name:<28} → {prob:.1%}  (true={'Toxic' if true_label else 'Safe '})"
    )


# ================================================================
# 7. SAVE REPORT
# ================================================================
print("\n[6/6] Saving training report...")

ckpt = str(CHECKPOINT_DIR)
report_data = {
    "model": "GIN (Graph Isomorphism Network)",
    "architecture": {
        "num_layers": GNN_NUM_LAYERS,
        "hidden_dim": GNN_HIDDEN_DIM,
        "embedding_dim": GNN_EMBEDDING_DIM,
        "dropout": DROPOUT,
        "total_params": total_params,
    },
    "training": {
        "epochs_run": best_epoch,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_val_auc": round(best_val_auc, 4),
    },
    "test_metrics": {
        "roc_auc": round(test_auc, 4),
        "brier": round(test_brier, 4),
        "classification_report": report,
    },
    "projection": {
        "method": "PCA",
        "from_dim": GNN_EMBEDDING_DIM,
        "to_dim": GNN_PROJECTION_DIM,
        "variance_explained": round(explained, 4),
    },
    "data": {
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "test_graphs": len(test_graphs),
        "atom_feature_dim": atom_feat_dim,
    },
    "device": str(device),
    "checkpoints": [
        f"{ckpt}/gnn_model.pt",
        f"{ckpt}/gnn_projector.pkl",
        f"{ckpt}/gnn_embeddings_train.npy",
    ],
}

with open(f"{ckpt}/gnn_training_report.json", "w") as f:
    json.dump(report_data, f, indent=2)

print(f"\n  Saved: {ckpt}/gnn_model.pt")
print(f"  Saved: {ckpt}/gnn_projector.pkl")
print(f"  Saved: {ckpt}/gnn_embeddings_train.npy")
print(f"  Saved: {ckpt}/gnn_training_report.json")

print("\n" + "=" * 65)
print(f" ✅ GNN Training Complete!")
print(f"    Test AUC: {test_auc:.4f}  |  Brier: {test_brier:.4f}")
print(
    f"    Embedding: {GNN_EMBEDDING_DIM}-d → PCA → {GNN_PROJECTION_DIM}-d for quantum"
)
print(f"\n    To enable in app_v2:")
print(f"      1. Set ENABLE_GNN = True in config.py")
print(f"      2. Restart streamlit")
print("=" * 65)
