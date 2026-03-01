import os
import warnings
import time
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

# Suppress annoying warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

print("⚛️ Initializing 14-Qubit Physical Descriptor Quantum Pipeline...")


# ==========================================
# 1. INDUSTRIAL DESCRIPTOR EXTRACTION
# ==========================================
def extract_14_physical_descriptors(smiles):
    """
    Replaces Morgan Fingerprints. Calculates 14 continuous physical
    properties that dictate drug toxicity and ADMET behavior.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Fallback for corrupted SMILES
        return np.zeros(14)

    features = [
        Descriptors.MolWt(mol),  # 1. Molecular Weight
        Descriptors.MolLogP(mol),  # 2. Lipophilicity
        Descriptors.TPSA(mol),  # 3. Polar Surface Area
        Descriptors.NumHDonors(mol),  # 4. H-Bond Donors
        Descriptors.NumHAcceptors(mol),  # 5. H-Bond Acceptors
        Descriptors.NumRotatableBonds(mol),  # 6. Flexibility
        Descriptors.FractionCSP3(mol),  # 7. 3D sp3 Carbon Fraction
        Descriptors.HeavyAtomCount(mol),  # 8. Heavy Atoms
        Descriptors.NHOHCount(mol),  # 9. N/O with attached H
        Descriptors.NOCount(mol),  # 10. Total N/O atoms
        Descriptors.RingCount(mol),  # 11. Total Rings
        Descriptors.NumAromaticRings(mol),  # 12. Aromatic Rings
        Descriptors.NumSaturatedRings(mol),  # 13. Saturated Rings
        Descriptors.NumAliphaticRings(mol),  # 14. Aliphatic Rings
    ]
    return np.array(features)

 
def get_subset_data(num_train=200, num_test=50):
    print(f"Downloading Tox21 and extracting {num_train} training samples...")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    df = pd.read_csv(url).dropna(subset=["NR-AR"])

    toxic_df = df[df["NR-AR"] == 1].head(num_train // 2)
    safe_df = df[df["NR-AR"] == 0].head(num_train // 2 + num_test)
    combined = pd.concat([toxic_df, safe_df]).sample(frac=1, random_state=42)

    train_df = combined.iloc[:num_train]
    test_df = combined.iloc[num_train : num_train + num_test]

    print("Calculating 14 RDKit Physical Descriptors per molecule...")
    X_train = np.array([extract_14_physical_descriptors(s) for s in train_df["smiles"]])
    X_test = np.array([extract_14_physical_descriptors(s) for s in test_df["smiles"]])

    y_train = train_df["NR-AR"].values
    y_test = test_df["NR-AR"].values

    return X_train, y_train, X_test, y_test


# ==========================================
# 2. THE QUANTUM KERNEL SVM ARCHITECTURE
# ==========================================
class QuantumPhysicalSVM:
    def __init__(self, n_qubits=14):
        self.n_qubits = n_qubits

        # We use MinMaxScaler to bind physical properties to quantum angles [-pi, pi]
        # This translates chemistry directly into phase rotations.
        self.scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        def feature_map(x):
            # Encode physical properties as rotation angles
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")

            # Entangle the physical properties together
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            feature_map(x1)
            qml.adjoint(feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))

        self.qnode = kernel_circuit

    def _compute_kernel_matrix(self, X1, X2):
        N, M = len(X1), len(X2)
        matrix = np.zeros((N, M))

        print(
            f"Executing {N*M} quantum circuit simulations on {self.n_qubits} qubits..."
        )
        start_time = time.time()

        for i in range(N):
            if i % 25 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f"  ... {i}/{N} rows processed ({elapsed:.1f} seconds)")
            for j in range(M):
                matrix[i, j] = self.qnode(X1[i], X2[j])[0]

        return matrix

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test):
        print("\n--- Scaling Physical Data to Quantum Phase Angles [-π, π] ---")
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_train_scaled = X_train_scaled
        X_test_scaled = self.scaler.transform(X_test)

        print("\n--- Calculating Quantum Kernel (Training) ---")
        K_train = self._compute_kernel_matrix(X_train_scaled, X_train_scaled)

        print("\n--- Training Classical SVM ---")
        self.svm = SVC(kernel="precomputed", probability=True, class_weight="balanced")
        self.svm.fit(K_train, y_train)

        print("\n--- Calculating Quantum Kernel (Testing) ---")
        K_test = self._compute_kernel_matrix(X_test_scaled, self.X_train_scaled)

        valid_preds = self.svm.predict_proba(K_test)[:, 1]
        auc_score = roc_auc_score(y_test, valid_preds)
        print(f"\n🚀 14-Qubit Physical QSVM ROC-AUC on Test Set: {auc_score:.4f}")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_subset_data(num_train=100, num_test=50)

    qsvm = QuantumPhysicalSVM(n_qubits=14)
    qsvm.fit_and_evaluate(X_train, y_train, X_test, y_test)