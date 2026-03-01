"""
Embedding Service — GNN Encoder + Cache
========================================
Manages molecular embeddings produced by the GNN encoder.

Initially operates in PASSTHROUGH mode using orthogonal descriptors
as "embeddings". When GNN is trained and enabled, seamlessly switches
to learned embeddings.
"""

import numpy as np
import pickle
from pathlib import Path

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, ENABLE_GNN, GNN_EMBEDDING_DIM, GNN_PROJECTION_DIM
from services.graph_service import GraphService


class EmbeddingService:
    """
    Molecular embedding service with cache.

    Two modes:
      - PASSTHROUGH (default): Uses orthogonal descriptors as embeddings
      - GNN (when ENABLE_GNN=True): Uses trained GNN encoder

    Provides PCA projection from embedding space → quantum-compatible
    low-dimensional space.
    """

    def __init__(self, feature_service=None, model_path=None, device="cpu"):
        """
        Args:
            feature_service: FeatureService instance (for passthrough mode)
            model_path: Path to trained GNN model weights (.pt state dict)
            device: 'cpu' or 'cuda'
        """
        self.feature_svc = feature_service
        self.device = device
        self.model = None
        self.graph_svc = GraphService()
        self.projector = None  # PCA or learned linear projection
        self._cache = {}  # canonical_smiles → embedding

        if model_path and Path(model_path).exists() and ENABLE_GNN:
            self._load_gnn(model_path)

        # Load PCA projector if available
        projector_path = Path(CHECKPOINT_DIR) / "gnn_projector.pkl"
        if projector_path.exists():
            with open(projector_path, "rb") as f:
                self.projector = pickle.load(f)

    def _load_gnn(self, model_path):
        """Load trained GINEncoder from state dict checkpoint."""
        try:
            import torch
            from training.train_gnn import GINEncoder

            # Get atom feature dim from graph service
            atom_dim = self.graph_svc.atom_feature_dim

            self.model = GINEncoder(in_dim=atom_dim)
            self.model.load_state_dict(
                torch.load(str(model_path), map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"  [GNN] GINEncoder loaded from {model_path}")
        except Exception as e:
            print(f"  [GNN] Failed to load model: {e}. Using passthrough mode.")
            self.model = None

    def get_embedding(self, smiles, selected_features=None):
        """
        Get molecular embedding vector.

        In PASSTHROUGH mode: returns orthogonal descriptors (20-d).
        In GNN mode: returns learned embedding (128-d).

        Args:
            smiles: SMILES string
            selected_features: Feature names for passthrough mode

        Returns:
            np.ndarray: Embedding vector
        """
        # Check cache
        if self.feature_svc:
            canon = self.feature_svc.canonical_smiles(smiles)
            if canon and canon in self._cache:
                return self._cache[canon]

        if self.model is not None and ENABLE_GNN:
            embedding = self._gnn_embed(smiles)
        else:
            # Passthrough: use orthogonal descriptors
            if self.feature_svc is None:
                raise ValueError("FeatureService required for passthrough mode")
            embedding = self.feature_svc.extract_orthogonal_descriptors(
                smiles, selected_features
            )

        # Cache
        if self.feature_svc:
            canon = self.feature_svc.canonical_smiles(smiles)
            if canon:
                self._cache[canon] = embedding

        return embedding

    def _gnn_embed(self, smiles):
        """Run GIN forward pass to produce 128-d embedding."""
        import torch
        from torch_geometric.data import DataLoader

        graph = self.graph_svc.smiles_to_graph(smiles)
        if graph is None:
            return np.zeros(GNN_EMBEDDING_DIM, dtype=np.float32)

        # Add dummy label for data loader compatibility
        graph.y = torch.tensor([0.0], dtype=torch.float)
        batch = next(iter(DataLoader([graph], batch_size=1)))
        batch = batch.to(self.device)

        with torch.no_grad():
            _, embedding = self.model(batch)

        return embedding.cpu().numpy().flatten()

    def project_for_quantum(self, embedding, n_dims=GNN_PROJECTION_DIM):
        """
        Project embedding to lower dimension for quantum kernel input.

        If projector is fitted: applies PCA/linear projection.
        Otherwise: truncates to first n_dims dimensions.

        Args:
            embedding: High-dimensional embedding vector
            n_dims: Target dimensions (default: matches N_QUBITS)

        Returns:
            np.ndarray: Projected embedding
        """
        if self.projector is not None:
            return self.projector.transform(embedding.reshape(1, -1))[0][:n_dims]

        # Truncate fallback
        if len(embedding) <= n_dims:
            return embedding
        return embedding[:n_dims]

    def fit_projector(self, embeddings, n_dims=GNN_PROJECTION_DIM):
        """
        Fit PCA projector on a set of embeddings.

        Args:
            embeddings: (N, embedding_dim) array
            n_dims: Target projection dimension
        """
        from sklearn.decomposition import PCA

        self.projector = PCA(n_components=n_dims)
        self.projector.fit(embeddings)
        explained = sum(self.projector.explained_variance_ratio_[:n_dims])
        print(f"  PCA projector: {n_dims} dims explain {explained:.1%} variance")

    def clear_cache(self):
        self._cache.clear()

    @property
    def cache_size(self):
        return len(self._cache)

    @property
    def mode(self):
        return "gnn" if (self.model is not None and ENABLE_GNN) else "passthrough"
