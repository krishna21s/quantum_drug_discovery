import os
import time
import json
import numpy as np
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
import warnings

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ================================================================
# CONFIGURATION
# ================================================================
N_QUBITS = 20
N_SHOTS = 1024
NYSTROM_LANDMARKS = 50
MAX_TRAIN = 500
MAX_TEST = 100
CHECKPOINT_DIR = "./checkpoints"

# -------- NEW FEATURE ----------
MODE = "exact"  # "exact" or "noisy"
# -------------------------------

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Cap workers: threads for exact mode (shared memory), processes for noisy
_RAW_CORES = max(1, multiprocessing.cpu_count() - 1)
CPU_CORES = min(_RAW_CORES, 6)  # Limit to avoid memory exhaustion


# ================================================================
# 1. DESCRIPTOR EXTRACTION (UNCHANGED CORE)
# ================================================================
def extract_rich_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc_dict = Descriptors.CalcMolDescriptors(mol)
    safe_descs = {
        k: float(v) for k, v in desc_dict.items() if not np.isnan(v) and not np.isinf(v)
    }
    return safe_descs


def load_and_filter_data():
    print(f"\n[DATA] Loading Tox21 dataset ({MAX_TRAIN} train / {MAX_TEST} test)...")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    df = pd.read_csv(url).dropna(subset=["NR-AR"])

    toxic = df[df["NR-AR"] == 1]
    safe = df[df["NR-AR"] == 0]

    n_toxic_train = min(MAX_TRAIN // 2, len(toxic))
    n_safe_train = min(MAX_TRAIN - n_toxic_train, len(safe) - MAX_TEST)

    train_df = pd.concat([toxic.head(n_toxic_train), safe.head(n_safe_train)]).sample(
        frac=1, random_state=42
    )

    test_df = pd.concat(
        [
            toxic.iloc[n_toxic_train : n_toxic_train + MAX_TEST // 2],
            safe.iloc[n_safe_train : n_safe_train + MAX_TEST // 2],
        ]
    ).sample(frac=1, random_state=42)

    print("  Extracting descriptor pool...")
    train_features = [extract_rich_descriptors(s) for s in train_df["smiles"]]
    test_features = [extract_rich_descriptors(s) for s in test_df["smiles"]]

    valid_train = [i for i, f in enumerate(train_features) if f is not None]
    valid_test = [i for i, f in enumerate(test_features) if f is not None]

    train_df = train_df.iloc[valid_train]
    test_df = test_df.iloc[valid_test]

    X_train_df = pd.DataFrame([train_features[i] for i in valid_train])
    X_test_df = pd.DataFrame([test_features[i] for i in valid_test])

    print("  Applying strict orthogonality filter...")
    corr_matrix = X_train_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.85)]
    X_train_filtered = X_train_df.drop(columns=to_drop)

    variances = X_train_filtered.var().sort_values(ascending=False)
    selected_features = variances.head(N_QUBITS).index.tolist()

    print(f"  Selected {len(selected_features)} quantum features.")

    with open(f"{CHECKPOINT_DIR}/selected_features.json", "w") as f:
        json.dump(selected_features, f)

    X_train = X_train_df[selected_features].values
    X_test = X_test_df[selected_features].values
    y_train = train_df["NR-AR"].values
    y_test = test_df["NR-AR"].values

    return X_train, y_train, X_test, y_test


# ================================================================
# 2. HEA ANSATZ (UNCHANGED)
# ================================================================
def build_hea_circuit(x1, x2):
    qc = QuantumCircuit(N_QUBITS)

    for i in range(N_QUBITS):
        qc.ry(float(x1[i]), i)

    for i in range(0, N_QUBITS - 1, 2):
        qc.cx(i, i + 1)
    for i in range(1, N_QUBITS - 1, 2):
        qc.cx(i, i + 1)

    for i in range(1, N_QUBITS - 1, 2)[::-1]:
        qc.cx(i, i + 1)
    for i in range(0, N_QUBITS - 1, 2)[::-1]:
        qc.cx(i, i + 1)

    for i in range(N_QUBITS):
        qc.ry(-float(x2[i]), i)

    return qc


# ================================================================
# 3. NOISE MODEL (NEW)
# ================================================================
def build_noise_model():
    noise_model = NoiseModel()

    error_1 = depolarizing_error(0.001, 1)
    error_2 = depolarizing_error(0.01, 2)

    noise_model.add_all_qubit_quantum_error(error_1, ["ry"])
    noise_model.add_all_qubit_quantum_error(error_2, ["cx"])

    readout_error = ReadoutError([[0.99, 0.01], [0.01, 0.99]])
    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


# ================================================================
# 4. FIDELITY WORKER (DUAL MODE)
# ================================================================
def fidelity_worker(args):
    x1, x2 = args
    qc = build_hea_circuit(x1, x2)

    if MODE == "exact":
        from qiskit.quantum_info import Statevector
        sv = Statevector.from_instruction(qc)
        return np.abs(sv.data[0]) ** 2

    else:  # noisy
        qc.measure_all()
        noise_model = build_noise_model()
        sim = AerSimulator(noise_model=noise_model)
        result = sim.run(qc, shots=N_SHOTS).result()
        counts = result.get_counts()
        zeros = "0" * N_QUBITS
        return counts.get(zeros, 0) / N_SHOTS


# ================================================================
# 5. NYSTROM (UNCHANGED LOGIC, PARALLEL SAFE)
# ================================================================
def compute_nystrom_stateful(X_train, X_test):
    N, T = len(X_train), len(X_test)
    m = min(NYSTROM_LANDMARKS, N)
    landmark_idx = np.linspace(0, N - 1, m, dtype=int)
    X_lm = X_train[landmark_idx]

    def process_matrix(name, rows, cols, X_A, X_B, symmetric=False):
        path = f"{CHECKPOINT_DIR}/{name}_{MODE}.npy"

        if os.path.exists(path):
            K = np.load(path)
        else:
            K = np.full((rows, cols), np.nan)

        for i in range(rows):
            if not np.isnan(K[i, 0]):
                continue

            tasks = []
            for j in range(cols):
                if symmetric and j < i:
                    K[i, j] = K[j, i]
                elif symmetric and i == j:
                    K[i, j] = 1.0
                else:
                    tasks.append((X_A[i], X_B[j]))

            if tasks:
                # Threads for exact (numpy releases GIL, no DLL reload)
                # Processes for noisy (Aer simulation benefits from isolation)
                PoolClass = ThreadPoolExecutor if MODE == "exact" else ProcessPoolExecutor
                with PoolClass(max_workers=CPU_CORES) as executor:
                    results = list(executor.map(fidelity_worker, tasks))

                idx = 0
                for j in range(cols):
                    if symmetric and j < i:
                        continue
                    elif symmetric and i == j:
                        continue
                    else:
                        K[i, j] = results[idx]
                        idx += 1

            if i % 5 == 0:
                np.save(path, K)

        np.save(path, K)
        return K

    print("\n[KERNEL] Computing matrices...")
    K_mm = process_matrix("K_mm", m, m, X_lm, X_lm, symmetric=True)
    K_nm = process_matrix("K_nm", N, m, X_train, X_lm)
    K_tm = process_matrix("K_tm", T, m, X_test, X_lm)

    K_mm_inv = np.linalg.pinv(K_mm + 1e-6 * np.eye(m))
    K_train = K_nm @ K_mm_inv @ K_nm.T
    K_test = K_tm @ K_mm_inv @ K_nm.T

    np.fill_diagonal(K_train, 1.0)
    return (K_train + K_train.T) / 2.0, K_test


# ================================================================
# 6. MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" 🚀 PRODUCTION CORE ENGINE: 20-QUBIT QSVM (Dual Mode)")
    print("    Modes: Exact Statevector | Noisy Hardware Simulation")
    print("=" * 65)
    print(f" 💻 Detected {multiprocessing.cpu_count()} cores | Using {CPU_CORES} workers")
    print(f" ⚙ Running in MODE: {MODE.upper()}")

    X_train_raw, y_train, X_test_raw, y_test = load_and_filter_data()

    print("\n[SCALE] Scaling descriptors...")
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    K_train, K_test = compute_nystrom_stateful(X_train, X_test)

    print("\n[SVM] Training...")
    svm = SVC(kernel="precomputed", probability=True, class_weight="balanced", C=1.0)
    svm.fit(K_train, y_train)

    y_pred = svm.predict_proba(K_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    if auc < 0.5:
        auc = 1.0 - auc

    print("\n" + "=" * 50)
    print(" 🏆 VALIDATION RESULTS")
    print(f" MODE            : {MODE.upper()}")
    print(f" Train Samples   : {len(y_train)}")
    print(f" Test Samples    : {len(y_test)}")
    print(f" Qubits          : {N_QUBITS}")
    print(f" ROC-AUC         : {auc:.4f}")
    print("=" * 50)
