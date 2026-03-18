"""
GNN Training Script V3 — Balanced + Multi-Task + Focal Loss (Tuned)
=============================================================
Trains a Graph Isomorphism Network (GIN) on Tox21 with:
  1. WeightedRandomSampler → balanced mini-batches
  2. Focal Loss → down-weights easy negatives, focuses on hard positives
  3. Multi-task learning → all 12 Tox21 endpoints as auxiliary supervision
  4. Dense molecular embeddings (128-d) for quantum kernel input

FIXES over V1:
  - Toxic recall 48% → should be 65-85%+
  - Reference toxics (Phenanthrene, BPA) flagged correctly

Usage:
    cd construction_v2
    ..\\venv\\Scripts\\python.exe training/train_gnn.py

Outputs (in ./checkpoints/):
    gnn_model.pt              — Trained GIN weights
    gnn_projector.pkl         — PCA projector (128-d → 20-d for quantum)
    gnn_embeddings_train.npy  — Cached train embeddings
    gnn_training_report.json  — AUC, Brier, loss curves, per-endpoint metrics
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
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from torch.utils.data import WeightedRandomSampler
except ImportError:
    print("=" * 65)
    print(" ❌ PyTorch not found!")
    print("    Install: pip install torch")
    print("=" * 65)
    sys.exit(1)

try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GINConv, global_add_pool
except ImportError:
    print("=" * 65)
    print(" ❌ torch_geometric not found!")
    print("    Install: pip install torch_geometric")
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
# ALL 12 TOX21 ENDPOINTS (multi-task)
# ================================================================
TOX21_ENDPOINTS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]
PRIMARY_ENDPOINT_IDX = TOX21_ENDPOINTS.index(TOX21_ENDPOINT)  # NR-AR

# ================================================================
# CONFIGURATION (V3 — tuned balance)
# ================================================================
BATCH_SIZE = 64
LEARNING_RATE = 5e-4  # Lower LR for stability
WEIGHT_DECAY = 1e-4  # Slightly stronger regularization
EPOCHS = 150
PATIENCE = 25  # More patience with cosine schedule
DROPOUT = 0.2  # Less dropout → learn more from sparse positives
NUM_WORKERS = 0  # 0 for Windows compatibility
FOCAL_GAMMA = 2.0  # Focal loss focusing parameter
FOCAL_ALPHA = 0.5  # Neutral alpha — sampler handles balance, not loss
OVERSAMPLE_RATIO = 3  # Gentler oversample (was 5x, too aggressive)

print("=" * 65)
print(" 🚀 GNN (GIN) Training Pipeline V3 — Balanced + Multi-Task (Tuned)")
print(
    f"    Architecture: GIN  |  Layers: {GNN_NUM_LAYERS}  |  Hidden: {GNN_HIDDEN_DIM}"
)
print(
    f"    Embedding: {GNN_EMBEDDING_DIM}-d  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}"
)
print(
    f"    Focal Loss: γ={FOCAL_GAMMA}, α={FOCAL_ALPHA} (neutral — sampler handles balance)"
)
print(f"    Multi-Task: {len(TOX21_ENDPOINTS)} Tox21 endpoints")
print(f"    Oversampling: toxic {OVERSAMPLE_RATIO}x  |  LR: {LEARNING_RATE}")
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Device: {device}")


# ================================================================
# 1. FOCAL LOSS
# ================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.

    FL(p) = -α (1-p)^γ log(p)        for y=1
    FL(p) = -(1-α) p^γ log(1-p)      for y=0

    When γ=0, this degrades to standard weighted BCE.
    When γ>0, easy negatives are down-weighted, focusing training
    on hard positives (toxic molecules that the model struggles with).
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Focal weighting
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


# ================================================================
# 2. GIN MODEL WITH MULTI-TASK HEAD
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
    Graph Isomorphism Network (GIN) with multi-task heads.

    Produces:
      - embedding: dense 128-d molecular representation
      - primary_logit: NR-AR toxicity prediction
      - auxiliary_logits: 11 additional Tox21 endpoint predictions
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=GNN_HIDDEN_DIM,
        embed_dim=GNN_EMBEDDING_DIM,
        num_layers=GNN_NUM_LAYERS,
        num_tasks=12,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_tasks = num_tasks

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

        # Primary classification head (NR-AR)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

        # Auxiliary multi-task heads (other 11 endpoints)
        self.aux_heads = nn.ModuleList(
            [nn.Linear(embed_dim, 1) for _ in range(num_tasks - 1)]
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

        # Shared embedding
        embedding = self.embed_head(graph_repr)

        # Primary task (NR-AR)
        primary_logit = self.classifier(embedding).squeeze(-1)

        # Auxiliary tasks
        aux_logits = [head(embedding).squeeze(-1) for head in self.aux_heads]

        return primary_logit, aux_logits, embedding

    def get_embedding(self, data):
        """Get embedding only (for inference)."""
        self.eval()
        with torch.no_grad():
            _, _, embedding = self.forward(data)
        return embedding.cpu().numpy()


# ================================================================
# 3. LOAD DATA
# ================================================================
print("\n[1/6] Loading Tox21 dataset (all 12 endpoints)...")
df = pd.read_csv(TOX21_URL)

# Keep molecules that have NR-AR label at minimum
df = df.dropna(subset=[TOX21_ENDPOINT])
print(f"      Total molecules with {TOX21_ENDPOINT}: {len(df)}")

# Extract multi-task labels (NaN = missing = will be masked in loss)
for ep in TOX21_ENDPOINTS:
    if ep not in df.columns:
        df[ep] = np.nan

toxic_count = (df[TOX21_ENDPOINT] == 1).sum()
safe_count = (df[TOX21_ENDPOINT] == 0).sum()
print(
    f"      {TOX21_ENDPOINT} — Toxic: {toxic_count}  |  Safe: {safe_count}  |  Ratio: 1:{safe_count // toxic_count}"
)

# Show multi-task coverage
for ep in TOX21_ENDPOINTS:
    n_labeled = df[ep].notna().sum()
    n_pos = (df[ep] == 1).sum()
    print(f"        {ep:<16} → {n_labeled:5d} labeled ({n_pos} positive)")

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
print(f"\n      Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")


# ================================================================
# 4. CONVERT SMILES TO GRAPHS (with multi-task labels)
# ================================================================
print("\n[2/6] Converting SMILES to graphs via GraphService...")
graph_svc = GraphService()
t0 = time.time()


def build_dataset(dataframe, endpoints=TOX21_ENDPOINTS):
    """Convert DataFrame to PyG Data objects with multi-task labels."""
    graphs = []
    skipped = 0
    for _, row in dataframe.iterrows():
        g = graph_svc.smiles_to_graph(row["smiles"])
        if g is None:
            skipped += 1
            continue

        # Multi-task labels: shape (num_tasks,), NaN → -1 (masked)
        labels = []
        for ep in endpoints:
            val = row.get(ep, np.nan)
            labels.append(float(val) if pd.notna(val) else -1.0)

        g.y = torch.tensor(labels, dtype=torch.float)
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

atom_feat_dim = train_graphs[0].x.shape[1]
print(f"      Atom feature dim: {atom_feat_dim}")


# ================================================================
# 5. BALANCED SAMPLING (WeightedRandomSampler)
# ================================================================
print(f"\n      Creating weighted sampler (toxic {OVERSAMPLE_RATIO}x)...")

# Primary endpoint labels
primary_labels = [g.y[PRIMARY_ENDPOINT_IDX].item() for g in train_graphs]
n_pos = sum(1 for l in primary_labels if l == 1.0)
n_neg = sum(1 for l in primary_labels if l == 0.0)
print(f"      Primary labels: {n_pos} toxic / {n_neg} safe")

# Sample weights: oversample toxic by OVERSAMPLE_RATIO
sample_weights = []
for label in primary_labels:
    if label == 1.0:
        sample_weights.append(OVERSAMPLE_RATIO)
    else:
        sample_weights.append(1.0)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_graphs),
    replacement=True,
)

train_loader = DataLoader(
    train_graphs,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    val_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)


# ================================================================
# 6. TRAIN MODEL
# ================================================================
print(f"\n[3/6] Training GIN model ({EPOCHS} epochs, patience={PATIENCE})...")
print(f"      Focal Loss: γ={FOCAL_GAMMA}, α={FOCAL_ALPHA}")
print(f"      Multi-Task: {len(TOX21_ENDPOINTS)} endpoints")

model = GINEncoder(in_dim=atom_feat_dim, num_tasks=len(TOX21_ENDPOINTS)).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# Cosine annealing with warm restarts — avoids premature LR drops
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
aux_criterion = nn.BCEWithLogitsLoss(reduction="none")

total_params = sum(p.numel() for p in model.parameters())
print(f"      Model parameters: {total_params:,}")

best_val_auc = 0.0
best_epoch = 0
patience_counter = 0
history = {"train_loss": [], "val_auc": [], "val_loss": []}


def compute_multitask_loss(primary_logit, aux_logits, labels):
    """
    Compute combined primary focal loss + auxiliary BCE loss.

    labels: (batch, num_tasks) — -1 means masked (skip).
    Primary task weight = 1.0, auxiliary tasks weight = 0.3 each.
    """
    # Primary task (NR-AR) with focal loss
    primary_labels = labels[:, PRIMARY_ENDPOINT_IDX]
    valid_primary = primary_labels >= 0
    if valid_primary.sum() > 0:
        loss_primary = focal_loss(
            primary_logit[valid_primary],
            primary_labels[valid_primary],
        )
    else:
        loss_primary = torch.tensor(0.0, device=device)

    # Auxiliary tasks
    loss_aux = torch.tensor(0.0, device=device)
    aux_count = 0
    aux_idx = 0
    for t in range(len(TOX21_ENDPOINTS)):
        if t == PRIMARY_ENDPOINT_IDX:
            continue
        task_labels = labels[:, t]
        valid = task_labels >= 0
        if valid.sum() > 0:
            task_loss = aux_criterion(
                aux_logits[aux_idx][valid],
                task_labels[valid],
            ).mean()
            loss_aux = loss_aux + task_loss
            aux_count += 1
        aux_idx += 1

    if aux_count > 0:
        loss_aux = loss_aux / aux_count

    # Combined: primary (weight=1.0) + auxiliary (weight=0.3)
    return loss_primary + 0.3 * loss_aux


for epoch in range(1, EPOCHS + 1):
    # --- TRAIN ---
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        primary_logit, aux_logits, _ = model(batch)

        # Multi-task labels: batch.y is (batch_size, num_tasks)
        labels = batch.y.view(-1, len(TOX21_ENDPOINTS))
        loss = compute_multitask_loss(primary_logit, aux_logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    train_loss = total_loss / max(n_batches, 1)
    history["train_loss"].append(train_loss)

    # --- VALIDATE ---
    model.eval()
    val_preds = []
    val_labels = []
    val_loss_total = 0
    val_n = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            primary_logit, aux_logits, _ = model(batch)
            probs = torch.sigmoid(primary_logit)

            labels = batch.y.view(-1, len(TOX21_ENDPOINTS))
            primary_labels = labels[:, PRIMARY_ENDPOINT_IDX]
            valid = primary_labels >= 0

            if valid.sum() > 0:
                val_preds.extend(probs[valid].cpu().numpy())
                val_labels.extend(primary_labels[valid].cpu().numpy())

                loss = focal_loss(primary_logit[valid], primary_labels[valid])
                val_loss_total += loss.item() * valid.sum().item()
                val_n += valid.sum().item()

    val_loss = val_loss_total / max(val_n, 1)
    history["val_loss"].append(val_loss)

    try:
        val_auc = roc_auc_score(val_labels, val_preds)
    except Exception:
        val_auc = 0.5

    history["val_auc"].append(val_auc)
    scheduler.step(epoch)

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
        torch.save(model.state_dict(), str(CHECKPOINT_DIR / "gnn_model.pt"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n      Early stopping at epoch {epoch} (best: {best_epoch})")
            break

print(f"\n      Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")


# ================================================================
# 7. EVALUATE ON TEST SET
# ================================================================
print("\n[4/6] Evaluating on test set...")

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
        primary_logit, _, embeddings = model(batch)
        probs = torch.sigmoid(primary_logit)

        labels = batch.y.view(-1, len(TOX21_ENDPOINTS))
        primary_labels = labels[:, PRIMARY_ENDPOINT_IDX]
        valid = primary_labels >= 0

        if valid.sum() > 0:
            test_preds.extend(probs[valid].cpu().numpy())
            test_labels.extend(primary_labels[valid].cpu().numpy())

        test_embeddings.extend(embeddings.cpu().numpy())

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)
test_embeddings = np.array(test_embeddings)

test_auc = roc_auc_score(test_labels, test_preds)
test_brier = brier_score_loss(test_labels, test_preds)

# ── AUTO-OPTIMIZE THRESHOLD ON VALIDATION SET ──
# Find threshold that maximizes F1 on validation predictions
print("\n      Auto-optimizing threshold on validation set...")
val_preds_arr = np.array(val_preds)
val_labels_arr = np.array(val_labels)
best_f1 = 0.0
THRESHOLD = 0.5  # default fallback
for t in np.arange(0.10, 0.70, 0.01):
    pred_bin = (val_preds_arr >= t).astype(int)
    tp = ((pred_bin == 1) & (val_labels_arr == 1)).sum()
    fp = ((pred_bin == 1) & (val_labels_arr == 0)).sum()
    fn = ((pred_bin == 0) & (val_labels_arr == 1)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    if f1 > best_f1:
        best_f1 = f1
        THRESHOLD = float(t)
print(f"      Optimal threshold: {THRESHOLD:.2f} (val F1={best_f1:.3f})")

test_pred_binary = (test_preds >= THRESHOLD).astype(int)
report = classification_report(
    test_labels.astype(int), test_pred_binary, output_dict=True
)

print(f"\n      Test ROC-AUC : {test_auc:.4f}")
print(f"      Test Brier   : {test_brier:.4f}")
print(f"      Threshold    : {THRESHOLD:.2f} (auto-optimized)")

toxic_key = "1" if "1" in report else "1.0"
safe_key = "0" if "0" in report else "0.0"
if toxic_key in report:
    print(f"      Precision (toxic): {report[toxic_key]['precision']:.3f}")
    print(f"      Recall    (toxic): {report[toxic_key]['recall']:.3f}")
    print(f"      F1        (toxic): {report[toxic_key]['f1-score']:.3f}")
if safe_key in report:
    print(f"      Precision (safe) : {report[safe_key]['precision']:.3f}")
    print(f"      Recall    (safe) : {report[safe_key]['recall']:.3f}")


# ================================================================
# 8. EXPORT EMBEDDINGS & PCA PROJECTOR
# ================================================================
print("\n[5/6] Exporting embeddings and PCA projector...")

train_embeddings = []
model.eval()
train_loader_full = DataLoader(
    train_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

with torch.no_grad():
    for batch in train_loader_full:
        batch = batch.to(device)
        _, _, emb = model(batch)
        train_embeddings.extend(emb.cpu().numpy())

train_embeddings = np.array(train_embeddings)
np.save(str(CHECKPOINT_DIR / "gnn_embeddings_train.npy"), train_embeddings)
print(f"      Train embeddings: {train_embeddings.shape}")

# PCA projector: 128-d → 20-d
pca = PCA(n_components=GNN_PROJECTION_DIM, random_state=RANDOM_STATE)
pca.fit(train_embeddings)
explained = sum(pca.explained_variance_ratio_)
print(
    f"      PCA {GNN_EMBEDDING_DIM}-d → {GNN_PROJECTION_DIM}-d  ({explained:.1%} variance explained)"
)

with open(str(CHECKPOINT_DIR / "gnn_projector.pkl"), "wb") as f:
    pickle.dump(pca, f)


# ================================================================
# 9. REFERENCE MOLECULE VALIDATION
# ================================================================
print("\n  ── Reference Molecule Validation ──")
print(f"  (Threshold = {THRESHOLD})")
for name, (smiles, true_label) in REFERENCE_MOLECULES.items():
    g = graph_svc.smiles_to_graph(smiles)
    if g is None:
        print(f"  ✗  {name:<28} → INVALID SMILES")
        continue
    g.y = torch.tensor([-1.0] * len(TOX21_ENDPOINTS), dtype=torch.float)
    batch = next(iter(DataLoader([g], batch_size=1)))
    batch = batch.to(device)
    with torch.no_grad():
        logit, _, _ = model(batch)
        prob = torch.sigmoid(logit).item()
    predicted_toxic = prob >= THRESHOLD
    correct = predicted_toxic == bool(true_label)
    status = "✓" if correct else "✗"
    label_str = "Toxic" if true_label else "Safe "
    pred_str = "TOXIC" if predicted_toxic else "safe "
    print(f"  {status}  {name:<28} → {prob:.1%} (pred={pred_str}, true={label_str})")


# ================================================================
# 10. SAVE REPORT
# ================================================================
print("\n[6/6] Saving training report...")
ckpt = str(CHECKPOINT_DIR)

report_data = {
    "model": "GIN (Graph Isomorphism Network) — V2 Balanced + Multi-Task",
    "architecture": {
        "num_layers": GNN_NUM_LAYERS,
        "hidden_dim": GNN_HIDDEN_DIM,
        "embedding_dim": GNN_EMBEDDING_DIM,
        "dropout": DROPOUT,
        "total_params": total_params,
        "num_tasks": len(TOX21_ENDPOINTS),
    },
    "training": {
        "epochs_run": best_epoch,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_val_auc": round(best_val_auc, 4),
        "focal_loss_gamma": FOCAL_GAMMA,
        "focal_loss_alpha": FOCAL_ALPHA,
        "oversample_ratio": OVERSAMPLE_RATIO,
        "classification_threshold": THRESHOLD,
    },
    "test_metrics": {
        "roc_auc": round(test_auc, 4),
        "brier": round(test_brier, 4),
        "threshold": THRESHOLD,
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
        "primary_endpoint": TOX21_ENDPOINT,
        "all_endpoints": TOX21_ENDPOINTS,
    },
    "improvements_over_v1": [
        "Focal Loss (γ=2, α=0.5 neutral) — sampler handles balance",
        "WeightedRandomSampler oversamples toxic 3x (tuned from 5x)",
        "Multi-task learning on all 12 Tox21 endpoints",
        "Auto-optimized threshold via val F1 maximization",
        "CosineAnnealingWarmRestarts + AdamW for stable training",
    ],
    "device": str(device),
}

with open(f"{ckpt}/gnn_training_report.json", "w") as f:
    json.dump(
        report_data,
        f,
        indent=2,
        default=lambda x: float(x) if hasattr(x, "item") else str(x),
    )

print(f"\n  Saved: {ckpt}/gnn_model.pt")
print(f"  Saved: {ckpt}/gnn_projector.pkl")
print(f"  Saved: {ckpt}/gnn_embeddings_train.npy")
print(f"  Saved: {ckpt}/gnn_training_report.json")

print("\n" + "=" * 65)
print(f" ✅ GNN Training V2 Complete!")
print(f"    Test AUC: {test_auc:.4f}  |  Brier: {test_brier:.4f}")
print(
    f"    Embedding: {GNN_EMBEDDING_DIM}-d → PCA → {GNN_PROJECTION_DIM}-d for quantum"
)
print(f"\n    To enable in app_v2:")
print(f"      1. Set ENABLE_GNN = True in config.py (line ~103)")
print(f"      2. Restart streamlit")
print("=" * 65)
