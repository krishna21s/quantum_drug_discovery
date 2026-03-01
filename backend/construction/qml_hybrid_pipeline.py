import os
import warnings
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem
import time

# Suppress warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

print("⚛️ Initializing 12-Qubit Hybrid Quantum Machine Learning Pipeline...")


# ==========================================
# 1. DATA PREPARATION (Sub-sampled for laptop limits)
# ==========================================
def get_subset_data(num_train=200, num_test=50):
    print(f"Downloading Tox21 and extracting {num_train} training samples...")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    df = pd.read_csv(url)

    # Target the NR-AR assay
    df = df.dropna(subset=["NR-AR"])

    # We ensure a mix of toxic (1) and safe (0) molecules
    toxic_df = df[df["NR-AR"] == 1].head(num_train // 2)
    safe_df = df[df["NR-AR"] == 0].head(num_train // 2 + num_test)

    combined = pd.concat([toxic_df, safe_df]).sample(frac=1, random_state=42)

    train_df = combined.iloc[:num_train]
    test_df = combined.iloc[num_train : num_train + num_test]

    # Featurize
    print("Generating Morgan Fingerprints...")
    X_train_fp = np.array(
        [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024)
            for s in train_df["smiles"]
        ]
    )
    X_test_fp = np.array(
        [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024)
            for s in test_df["smiles"]
        ]
    )

    y_train = train_df["NR-AR"].values
    y_test = test_df["NR-AR"].values

    return X_train_fp, y_train, X_test_fp, y_test


# ==========================================
# 2. THE QUANTUM KERNEL SVM ARCHITECTURE
# ==========================================
class QuantumKernelSVM:
    def __init__(self, n_qubits=12):
        self.n_qubits = n_qubits
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_qubits)  # 1024 bits -> 12 qubits

        # Initialize the PennyLane Simulator
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # 1. Define the Feature Map (How data becomes quantum)
        def feature_map(x):
            qml.AngleEmbedding(x, wires=range(self.n_qubits))
            # Create a ring of entanglement across all 12 qubits
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        # 2. Define the Kernel Circuit (Measuring similarity)
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            feature_map(x1)  # Encode molecule A
            qml.adjoint(feature_map)(x2)  # Apply inverse of molecule B
            return qml.probs(wires=range(self.n_qubits))  # Measure overlap

        self.qnode = kernel_circuit

    def _compute_kernel_matrix(self, X1, X2):
        N, M = len(X1), len(X2)
        matrix = np.zeros((N, M))

        total_circuits = N * M
        print(
            f"Executing {total_circuits} quantum circuit simulations on {self.n_qubits} qubits..."
        )

        start_time = time.time()
        for i in range(N):
            if i % 25 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f"  ... {i}/{N} rows processed ({elapsed:.1f} seconds)")
            for j in range(M):
                # We take the probability of the |00...0> state, which represents the fidelity
                matrix[i, j] = self.qnode(X1[i], X2[j])[0]

        return matrix

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test):
        print("\n--- Dimensionality Reduction ---")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        self.X_train_pca = X_train_pca

        X_test_scaled = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_scaled)

        print("\n--- Calculating Quantum Kernel (Training) ---")
        K_train = self._compute_kernel_matrix(X_train_pca, X_train_pca)

        print("\n--- Training Classical SVM ---")
        self.svm = SVC(kernel="precomputed", probability=True, class_weight="balanced")
        self.svm.fit(K_train, y_train)

        print("\n--- Calculating Quantum Kernel (Testing) ---")
        K_test = self._compute_kernel_matrix(X_test_pca, self.X_train_pca)

        valid_preds = self.svm.predict_proba(K_test)[:, 1]
        auc_score = roc_auc_score(y_test, valid_preds)
        print(f"\n🚀 12-Qubit QSVM ROC-AUC on Test Set: {auc_score:.4f}")


if __name__ == "__main__":
    # 1. Get our subset of data
    X_train, y_train, X_test, y_test = get_subset_data(num_train=100, num_test=50)

    # 2. Run the Hybrid Pipeline
    qsvm = QuantumKernelSVM(n_qubits=15)
    qsvm.fit_and_evaluate(X_train, y_train, X_test, y_test)
