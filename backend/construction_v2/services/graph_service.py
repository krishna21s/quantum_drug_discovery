"""
Graph Service — SMILES → PyG Graph Object
==========================================
Converts SMILES to torch_geometric Data objects for GNN input.
This is the API contract for Phase 2 (GNN training).

Currently operational as a stub — does not require torch/PyG installed
to import (lazy imports). When GNN is enabled, this becomes the graph
featurizer feeding into EmbeddingService.
"""

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Atom feature definitions (used by GIN/MPNN encoders)
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),  # H through Og
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "is_aromatic": [False, True],
}

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "is_conjugated": [False, True],
    "is_in_ring": [False, True],
}


def _one_hot(value, choices):
    """One-hot encode a value against a list of choices."""
    encoding = [0] * (len(choices) + 1)  # +1 for unknown
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1  # unknown
    return encoding


class GraphService:
    """
    Converts SMILES strings to graph representations suitable for GNNs.

    When torch_geometric is available, returns Data objects.
    Otherwise returns raw dict of numpy arrays (for testing / CPU pipelines).
    """

    def __init__(self):
        self._pyg_available = False
        try:
            import torch
            from torch_geometric.data import Data

            self._pyg_available = True
        except ImportError:
            pass

    def smiles_to_graph(self, smiles: str):
        """
        Convert SMILES to a graph representation.

        Returns:
            torch_geometric.data.Data if PyG available, else dict with:
              - node_features: np.ndarray (num_atoms, feat_dim)
              - edge_index: np.ndarray (2, num_edges)
              - edge_features: np.ndarray (num_edges, edge_feat_dim)
              - smiles: str
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Atom features
        atom_feats = []
        for atom in mol.GetAtoms():
            feat = []
            feat += _one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
            feat += _one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
            feat += _one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
            feat += _one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
            feat += _one_hot(atom.GetIsAromatic(), ATOM_FEATURES["is_aromatic"])
            feat.append(atom.GetMass() / 100.0)  # Normalized mass
            atom_feats.append(feat)

        node_features = np.array(atom_feats, dtype=np.float32)

        # Edge features & connectivity
        edge_indices = []
        edge_feats = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # Undirected: add both directions
            edge_indices.extend([[i, j], [j, i]])

            feat = []
            feat += _one_hot(bond.GetBondType(), BOND_FEATURES["bond_type"])
            feat += _one_hot(bond.GetIsConjugated(), BOND_FEATURES["is_conjugated"])
            feat += _one_hot(bond.IsInRing(), BOND_FEATURES["is_in_ring"])
            edge_feats.extend([feat, feat])  # Same for both directions

        if len(edge_indices) == 0:
            # Single-atom molecule
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, 11), dtype=np.float32)
        else:
            edge_index = np.array(edge_indices, dtype=np.int64).T
            edge_features = np.array(edge_feats, dtype=np.float32)

        if self._pyg_available:
            import torch
            from torch_geometric.data import Data

            return Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_features, dtype=torch.float),
                smiles=smiles,
            )
        else:
            return {
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_features": edge_features,
                "smiles": smiles,
            }

    @property
    def atom_feature_dim(self) -> int:
        """Dimension of per-atom feature vector."""
        dim = (
            len(ATOM_FEATURES["atomic_num"])
            + 1
            + len(ATOM_FEATURES["degree"])
            + 1
            + len(ATOM_FEATURES["formal_charge"])
            + 1
            + len(ATOM_FEATURES["hybridization"])
            + 1
            + len(ATOM_FEATURES["is_aromatic"])
            + 1
            + 1  # mass
        )
        return dim

    @property
    def edge_feature_dim(self) -> int:
        """Dimension of per-edge feature vector."""
        dim = (
            len(BOND_FEATURES["bond_type"])
            + 1
            + len(BOND_FEATURES["is_conjugated"])
            + 1
            + len(BOND_FEATURES["is_in_ring"])
            + 1
        )
        return dim
