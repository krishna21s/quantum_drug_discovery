"""
qml_hardware_pipeline.py
========================
20-Qubit Quantum Kernel SVM with qBraid -> IonQ Cloud Execution

Upgraded from qml_physics_pipeline.py with:
  - 20 physical molecular descriptors (expanded from 15)
  - Cloud-based quantum kernel via qBraid -> IonQ simulator
  - Nystrom kernel approximation (drastically reduces circuit count for safety)
  - Local Qiskit-Aer fallback when cloud is unavailable
  - Safety-first design: heavy quantum work on cloud, laptop stays cool

Prerequisites:
  pip install qiskit qiskit-aer qbraid rdkit-pypi scikit-learn pandas numpy python-dotenv

Usage:
  1. Set environment variables (or .env file):
       QBRAID_API_KEY=<your key>
       QBRAID_API_BASE=https://api.qbraid.com

  2. Run (cloud mode - IonQ simulator via qBraid):
       python qml_hardware_pipeline.py

  3. Run (local-only fallback):
       python qml_hardware_pipeline.py --local
  
  4. Run (Dry-run to see circuit estimates without execution):
       python qml_hardware_pipeline.py --dry-run

  5. Adjust samples safely:
       python qml_hardware_pipeline.py --train 80 --test 25 --landmarks 30
"""

import os
import sys
import warnings
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, classification_report
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

# Qiskit for circuit construction & local simulation
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Attempt qBraid SDK import for cloud execution
try:
    from qbraid.runtime import QbraidProvider

    QBRAID_SDK_AVAILABLE = True
except ImportError:
    try:
        from qbraid import QbraidProvider  # older SDK layout

        QBRAID_SDK_AVAILABLE = True
    except ImportError:
        QBRAID_SDK_AVAILABLE = False
        print("WARNING: qBraid SDK not found. Install with: pip install qbraid[runtime]")
        print("         Falling back to local Qiskit-Aer simulation.\n")

# Load .env if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


# ================================================================
# CONFIGURATION  (conservative defaults - safe for any laptop)
# ================================================================
N_QUBITS = 20  # 20 physical descriptors -> 20 qubits
N_SHOTS = 1024  # shots per circuit on IonQ
NYSTROM_LANDMARKS = 25  # landmark points for Nystrom approximation
MAX_TRAIN = 60  # training samples (start small, increase later)
MAX_TEST = 20  # test samples
BATCH_PROGRESS = 50  # print progress every N circuits

# qBraid / IonQ
QBRAID_API_KEY = os.getenv("QBRAID_API_KEY", "").strip()
QBRAID_API_BASE = os.getenv("QBRAID_API_BASE", "https://api.qbraid.com").strip()
IONQ_DEVICE_ID = os.getenv("IONQ_DEVICE_ID", "")  # auto-detect if empty

# Safety caps
MAX_CIRCUITS_CAP = 15_000  # abort if total circuits exceed this

print("=" * 65)
print("  20-Qubit Hardware Quantum Kernel SVM Pipeline")
print("  qBraid -> IonQ Simulator Integration")
print("=" * 65)


# ================================================================
# 1. TWENTY PHYSICAL / CHEMICAL DESCRIPTORS
# ================================================================
def extract_20_physical_descriptors(smiles):
    """
    Compute 20 continuous physicochemical descriptors from a SMILES string.
    Each descriptor maps to one qubit via angle embedding (RY rotation).

    Descriptors 1-14 : same as qml_physics_pipeline.py
    Descriptor  15   : Formal Charge
    Descriptors 16-20: complexity, surface area, connectivity & shape indices
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(N_QUBITS)

    features = [
        Descriptors.MolWt(mol),             #  1. Molecular Weight
        Descriptors.MolLogP(mol),            #  2. Lipophilicity (LogP)
        Descriptors.TPSA(mol),               #  3. Topological Polar Surface Area
        Descriptors.NumHDonors(mol),         #  4. H-Bond Donors
        Descriptors.NumHAcceptors(mol),      #  5. H-Bond Acceptors
        Descriptors.NumRotatableBonds(mol),  #  6. Rotatable Bonds (flexibility)
        Descriptors.FractionCSP3(mol),       #  7. sp3 Carbon Fraction
        Descriptors.HeavyAtomCount(mol),     #  8. Heavy Atom Count
        Descriptors.NHOHCount(mol),          #  9. N/O with attached H
        Descriptors.NOCount(mol),            # 10. Total N + O atoms
        Descriptors.RingCount(mol),          # 11. Total Rings
        Descriptors.NumAromaticRings(mol),   # 12. Aromatic Rings
        Descriptors.NumSaturatedRings(mol),  # 13. Saturated Rings
        Descriptors.NumAliphaticRings(mol),  # 14. Aliphatic Rings
        Chem.GetFormalCharge(mol),           # 15. Formal Charge
        Descriptors.BertzCT(mol),            # 16. Bertz Complexity Index
        Descriptors.LabuteASA(mol),          # 17. Labute Approx. Surface Area
        Descriptors.Chi0v(mol),              # 18. Kier-Hall Chi0v Connectivity
        Descriptors.HallKierAlpha(mol),      # 19. Hall-Kier Alpha Shape
        Descriptors.Kappa1(mol),             # 20. Kappa1 Shape Index
    ]
    return np.array(features, dtype=np.float64)


# ================================================================
# 2. DATA LOADING  (Tox21 NR-AR endpoint)
# ================================================================
def load_tox21_data(num_train=MAX_TRAIN, num_test=MAX_TEST):
    """Download Tox21, balance classes, extract 20 descriptors."""
    print(f"\n[DATA] Loading Tox21 dataset ({num_train} train / {num_test} test)...")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    try:
        df = pd.read_csv(url).dropna(subset=["NR-AR"])
    except Exception as e:
        print(f"  ERROR downloading Tox21: {e}")
        sys.exit(1)

    toxic = df[df["NR-AR"] == 1]
    safe = df[df["NR-AR"] == 0]

    # Balanced split
    n_toxic_train = min(num_train // 2, len(toxic))
    n_safe_train = min(num_train - n_toxic_train, len(safe) - num_test)

    train_toxic = toxic.head(n_toxic_train)
    train_safe = safe.head(n_safe_train)
    test_safe = safe.iloc[n_safe_train : n_safe_train + num_test // 2]
    test_toxic = toxic.iloc[n_toxic_train : n_toxic_train + num_test // 2]

    train_df = pd.concat([train_toxic, train_safe]).sample(frac=1, random_state=42)
    test_df = pd.concat([test_toxic, test_safe]).sample(frac=1, random_state=42)

    print(f"  Computing 20 physicochemical descriptors for {len(train_df)} train molecules...")
    X_train = np.array([extract_20_physical_descriptors(s) for s in train_df["smiles"]])
    X_test = np.array([extract_20_physical_descriptors(s) for s in test_df["smiles"]])
    y_train = train_df["NR-AR"].values
    y_test = test_df["NR-AR"].values

    print(f"  Train: {len(y_train)} samples ({int(sum(y_train))} toxic, {len(y_train) - int(sum(y_train))} safe)")
    print(f"  Test:  {len(y_test)} samples  ({int(sum(y_test))} toxic, {len(y_test) - int(sum(y_test))} safe)")
    return X_train, y_train, X_test, y_test


# ================================================================
# 3. QUANTUM KERNEL CIRCUIT  (Qiskit)
# ================================================================
def build_kernel_circuit(x1, x2, n_qubits=N_QUBITS):
    """
    Build the fidelity kernel circuit:
        K(x1, x2) = |<0| U_dag(x2) U(x1) |0>|^2

    Feature map U(x):
      1) RY(x_i) angle embedding on each qubit i
      2) Ring of CNOTs for entanglement

    Kernel value = probability of measuring all-zeros.
    """
    qc = QuantumCircuit(n_qubits, n_qubits)

    # --- Forward feature map U(x1) ---
    for i in range(n_qubits):
        qc.ry(float(x1[i]), i)
    for i in range(n_qubits):
        qc.cx(i, (i + 1) % n_qubits)

    # --- Adjoint feature map U_dag(x2) ---
    #     Reverse CNOT ring (CNOT is self-adjoint, but order reverses)
    for i in range(n_qubits - 1, -1, -1):
        qc.cx(i, (i + 1) % n_qubits)
    #     Reverse RY rotations
    for i in range(n_qubits - 1, -1, -1):
        qc.ry(-float(x2[i]), i)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# ================================================================
# 4. EXECUTION BACKENDS
# ================================================================
class LocalAerBackend:
    """
    Qiskit Aer local simulation.
    Safe for laptops: 20 qubits = ~16 MB per statevector.
    Each circuit takes ~0.1-0.5 s locally.
    """

    def __init__(self, n_qubits, n_shots):
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.sim = AerSimulator(method="statevector")
        self.name = f"Local Aer (statevector, {n_qubits}q)"
        self._all_zeros = "0" * n_qubits

    def kernel_value(self, x1, x2):
        """Run one kernel circuit, return P(|00...0>)."""
        qc = build_kernel_circuit(x1, x2, self.n_qubits)
        result = self.sim.run(qc, shots=self.n_shots).result()
        counts = result.get_counts()
        return counts.get(self._all_zeros, 0) / self.n_shots

    def estimate_seconds_per_circuit(self):
        return 0.3  # rough estimate for 20-qubit local sim


class QBraidIonQBackend:
    """
    Cloud execution via qBraid -> IonQ simulator.
    All heavy computation happens on cloud. Laptop only sends/receives data.
    """

    def __init__(self, n_qubits, n_shots, api_key, api_base=None, device_id=None):
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.device = None
        self.name = "qBraid -> IonQ Simulator"
        self._all_zeros = "0" * n_qubits

        if not QBRAID_SDK_AVAILABLE:
            raise RuntimeError("qBraid SDK not installed. Run: pip install qbraid[runtime]")
        if not api_key:
            raise RuntimeError("QBRAID_API_KEY not set in environment.")

        print(f"\n[CLOUD] Connecting to qBraid (base: {api_base or 'default'})...")

        # Initialize provider
        try:
            self.provider = QbraidProvider(api_key=api_key)
        except TypeError:
            # Some SDK versions use different kwarg names
            self.provider = QbraidProvider()
            os.environ["QBRAID_API_KEY"] = api_key

        # Find IonQ simulator device
        if device_id:
            self.device = self.provider.get_device(device_id)
            print(f"  Using specified device: {device_id}")
        else:
            self.device = self._auto_discover_ionq()

        if self.device is None:
            raise RuntimeError(
                "Could not find IonQ simulator on qBraid. "
                "Try setting IONQ_DEVICE_ID env var or use --device flag."
            )

        self.name = f"qBraid -> {self.device}"
        print(f"  Connected: {self.device}")

    def _auto_discover_ionq(self):
        """Search for IonQ simulator among available qBraid devices."""
        print("  Searching for IonQ simulator device...")

        # Try known device IDs first (fastest path)
        candidates = [
            "qbraid_qir_simulator",
            "ionq_simulator",
            "aws_ionq_harmony",
            "ionq_harmony",
            "ionq.simulator",
            "aws_dm_sim",
        ]
        for did in candidates:
            try:
                dev = self.provider.get_device(did)
                print(f"  Found device: {did}")
                return dev
            except Exception:
                continue

        # Fallback: list all devices and search by name
        try:
            devices = self.provider.get_devices()
            print(f"  Available devices: {len(devices)}")
            for d in devices:
                desc = str(d).lower()
                if "ionq" in desc and "simul" in desc:
                    print(f"  Auto-selected IonQ simulator: {d}")
                    return d
            # If no IonQ, take any simulator
            for d in devices:
                desc = str(d).lower()
                if "simul" in desc:
                    print(f"  IonQ not found, using simulator: {d}")
                    return d
        except Exception as e:
            print(f"  WARNING: Device listing failed: {e}")

        return None

    def kernel_value(self, x1, x2):
        """Submit one kernel circuit to IonQ, return P(|00...0>)."""
        qc = build_kernel_circuit(x1, x2, self.n_qubits)
        job = self.device.run(qc, shots=self.n_shots)
        result = job.result()

        # Handle different qBraid result formats
        try:
            counts = result.measurement_counts()
        except AttributeError:
            try:
                counts = result.get_counts()
            except AttributeError:
                counts = result.measurements if hasattr(result, "measurements") else {}

        return counts.get(self._all_zeros, 0) / self.n_shots

    def estimate_seconds_per_circuit(self):
        return 3.0  # cloud round-trip estimate


def create_backend(mode="cloud"):
    """Factory: return the appropriate execution backend."""
    if mode == "cloud" and QBRAID_SDK_AVAILABLE and QBRAID_API_KEY:
        try:
            return QBraidIonQBackend(
                n_qubits=N_QUBITS,
                n_shots=N_SHOTS,
                api_key=QBRAID_API_KEY,
                api_base=QBRAID_API_BASE,
                device_id=IONQ_DEVICE_ID or None,
            )
        except Exception as e:
            print(f"\n  WARNING: Cloud backend failed: {e}")
            print("  Falling back to local Aer simulation.\n")

    print("[BACKEND] Using local Qiskit Aer simulator.")
    return LocalAerBackend(N_QUBITS, N_SHOTS)


# ================================================================
# 5. NYSTROM KERNEL APPROXIMATION
# ================================================================
def compute_kernel_nystrom(X_train, X_test, backend, n_landmarks=NYSTROM_LANDMARKS):
    """
    Nystrom approximation for quantum kernel matrices.

    Instead of computing the full N x N kernel matrix (N^2 circuits),
    we only compute:
      K_mm : m x m  (landmarks vs landmarks)    ->  m(m-1)/2 circuits
      K_nm : N x m  (all train vs landmarks)    ->  N * m circuits
      K_tm : T x m  (all test vs landmarks)     ->  T * m circuits

    Then reconstruct:
      K_train ~ K_nm @ pinv(K_mm) @ K_nm.T
      K_test  ~ K_tm @ pinv(K_mm) @ K_nm.T

    This reduces circuit count from O(N^2) to O(N*m + m^2).
    """
    N = len(X_train)
    T = len(X_test)
    m = min(n_landmarks, N)

    # Select landmark indices (evenly spaced for reproducibility)
    landmark_idx = np.linspace(0, N - 1, m, dtype=int)
    X_landmarks = X_train[landmark_idx]

    # Circuit count estimate
    n_mm = m * (m - 1) // 2  # symmetric, diagonal = 1
    n_nm = N * m
    n_tm = T * m
    total_circuits = n_mm + n_nm + n_tm
    est_seconds = backend.estimate_seconds_per_circuit() * total_circuits

    print(f"\n{'=' * 60}")
    print(f"  Nystrom Kernel Approximation Plan")
    print(f"  Landmarks: {m}  |  Train: {N}  |  Test: {T}")
    print(f"  K_mm: {n_mm} circuits  |  K_nm: {n_nm}  |  K_tm: {n_tm}")
    print(f"  Total circuits: {total_circuits:,}")
    print(f"  Estimated time: {est_seconds / 60:.1f} min")
    print(f"  Backend: {backend.name}")
    print(f"{'=' * 60}")

    # ---- Safety gate ----
    if total_circuits > MAX_CIRCUITS_CAP:
        print(f"\n  SAFETY LIMIT: {total_circuits:,} exceeds cap of {MAX_CIRCUITS_CAP:,}.")
        print("  Reduce --landmarks or --train. Aborting.")
        sys.exit(1)

    # ========== K_mm: landmark-landmark kernel ==========
    print(f"\n[1/3] Computing K_mm ({m} x {m}) ...")
    K_mm = np.eye(m)  # K(x,x) = 1 always
    count = 0
    t0 = time.time()

    for i in range(m):
        for j in range(i + 1, m):
            val = backend.kernel_value(X_landmarks[i], X_landmarks[j])
            K_mm[i, j] = val
            K_mm[j, i] = val  # symmetric
            count += 1
            if count % BATCH_PROGRESS == 0:
                elapsed = time.time() - t0
                print(f"    {count}/{n_mm} pairs  ({elapsed:.1f}s)")

    elapsed_mm = time.time() - t0
    print(f"  K_mm complete: {count} circuits in {elapsed_mm:.1f}s")

    # ========== K_nm: train-landmark kernel ==========
    print(f"\n[2/3] Computing K_nm ({N} x {m}) ...")
    K_nm = np.zeros((N, m))
    count = 0
    t0 = time.time()

    for i in range(N):
        for j in range(m):
            # If train point IS a landmark -> K(x,x) = 1
            if i == landmark_idx[j]:
                K_nm[i, j] = 1.0
            else:
                K_nm[i, j] = backend.kernel_value(X_train[i], X_landmarks[j])
            count += 1
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    {i + 1}/{N} train points  ({elapsed:.1f}s)")

    elapsed_nm = time.time() - t0
    print(f"  K_nm complete: {count} circuits in {elapsed_nm:.1f}s")

    # ========== K_tm: test-landmark kernel ==========
    print(f"\n[3/3] Computing K_tm ({T} x {m}) ...")
    K_tm = np.zeros((T, m))
    count = 0
    t0 = time.time()

    for i in range(T):
        for j in range(m):
            K_tm[i, j] = backend.kernel_value(X_test[i], X_landmarks[j])
            count += 1
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"    {i + 1}/{T} test points  ({elapsed:.1f}s)")

    elapsed_tm = time.time() - t0
    print(f"  K_tm complete: {count} circuits in {elapsed_tm:.1f}s")

    # ========== Nystrom reconstruction ==========
    print("\n[RECONSTRUCT] Building approximate kernel matrices...")

    # Regularized pseudo-inverse for numerical stability
    K_mm_reg = K_mm + 1e-6 * np.eye(m)
    K_mm_inv = np.linalg.pinv(K_mm_reg)

    K_train_approx = K_nm @ K_mm_inv @ K_nm.T
    K_test_approx = K_tm @ K_mm_inv @ K_nm.T

    # Fix diagonal to 1.0, ensure symmetry
    np.fill_diagonal(K_train_approx, 1.0)
    K_train_approx = (K_train_approx + K_train_approx.T) / 2.0

    total_time = elapsed_mm + elapsed_nm + elapsed_tm
    print(f"\n  Kernel computation complete: {total_time:.1f}s total")
    return K_train_approx, K_test_approx


# ================================================================
# 6. QUANTUM HARDWARE SVM (Main Pipeline)
# ================================================================
class QuantumHardwareSVM:
    """
    20-Qubit Quantum Kernel SVM with cloud hardware support.
    Uses Nystrom approximation for tractable kernel computation.
    """

    def __init__(self, backend):
        self.backend = backend
        self.scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        self.svm = None

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test):
        # Scale descriptors to quantum rotation angles
        print("\n[SCALE] Scaling 20 descriptors to quantum angles [-pi, pi]...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Replace NaN/Inf produced by degenerate descriptors
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=np.pi, neginf=-np.pi)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=np.pi, neginf=-np.pi)

        # Compute quantum kernels via Nystrom
        K_train, K_test = compute_kernel_nystrom(
            X_train_scaled, X_test_scaled, self.backend
        )

        # Train SVM on precomputed quantum kernel
        print("\n[SVM] Training SVM on quantum kernel...")
        self.svm = SVC(
            kernel="precomputed",
            probability=True,
            class_weight="balanced",
            C=1.0,
        )
        self.svm.fit(K_train, y_train)

        # Evaluate
        print("\n[EVAL] Evaluating on test set...")
        y_pred_proba = self.svm.predict_proba(K_test)[:, 1]
        y_pred = self.svm.predict(K_test)

        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = float("nan")
            print("  WARNING: ROC-AUC undefined (only one class in test set).")

        # ---- Results ----
        print(f"\n{'=' * 65}")
        print(f"  20-Qubit Hardware QSVM Results")
        print(f"  Backend:    {self.backend.name}")
        print(f"  Qubits:     {N_QUBITS}")
        print(f"  Landmarks:  {NYSTROM_LANDMARKS}")
        print(f"  Train:      {len(y_train)}  |  Test: {len(y_test)}")
        print(f"  ROC-AUC:    {auc:.4f}")
        print(f"{'=' * 65}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test, y_pred,
                target_names=["Safe", "Toxic"],
                zero_division=0,
            )
        )

        return {
            "roc_auc": auc,
            "y_pred": y_pred.tolist(),
            "y_pred_proba": y_pred_proba.tolist(),
            "n_qubits": N_QUBITS,
            "n_landmarks": NYSTROM_LANDMARKS,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "backend": self.backend.name,
        }


# ================================================================
# 7. MAIN
# ================================================================
def main():
    global MAX_TRAIN, MAX_TEST, NYSTROM_LANDMARKS, N_SHOTS, IONQ_DEVICE_ID

    parser = argparse.ArgumentParser(
        description="20-Qubit Hardware Quantum Kernel SVM (qBraid -> IonQ)"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Force local Aer simulation (no cloud)",
    )
    parser.add_argument("--train", type=int, default=MAX_TRAIN, help="Training samples (default: 60)")
    parser.add_argument("--test", type=int, default=MAX_TEST, help="Test samples (default: 20)")
    parser.add_argument("--landmarks", type=int, default=NYSTROM_LANDMARKS, help="Nystrom landmarks (default: 25)")
    parser.add_argument("--shots", type=int, default=N_SHOTS, help="Shots per circuit (default: 1024)")
    parser.add_argument("--device", type=str, default=None, help="qBraid device ID override")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show circuit estimate and exit without running",
    )
    args = parser.parse_args()

    # Apply CLI overrides
    MAX_TRAIN = args.train
    MAX_TEST = args.test
    NYSTROM_LANDMARKS = args.landmarks
    N_SHOTS = args.shots
    if args.device:
        IONQ_DEVICE_ID = args.device

    mode = "local" if args.local else "cloud"

    # Safety summary
    m = min(NYSTROM_LANDMARKS, MAX_TRAIN)
    total_circuits_est = m * (m - 1) // 2 + MAX_TRAIN * m + MAX_TEST * m

    print(f"\n  Configuration:")
    print(f"   Mode:         {mode}")
    print(f"   Qubits:       {N_QUBITS}")
    print(f"   Train:        {MAX_TRAIN}")
    print(f"   Test:         {MAX_TEST}")
    print(f"   Landmarks:    {NYSTROM_LANDMARKS}")
    print(f"   Shots:        {N_SHOTS}")
    print(f"   Est. circuits: ~{total_circuits_est:,}")

    if mode == "local":
        mem_mb = (2**N_QUBITS * 16) / (1024 * 1024)  # complex128 per amplitude
        print(f"   Local RAM/circuit: ~{mem_mb:.1f} MB (safe for 20 qubits)")
    else:
        print(f"   qBraid API:   {QBRAID_API_BASE}")
        print(f"   API key:      {'*' * 6}...{QBRAID_API_KEY[-4:]}" if len(QBRAID_API_KEY) >= 4 else "   API key:      (not set)")

    if total_circuits_est > MAX_CIRCUITS_CAP:
        print(f"\n  SAFETY: {total_circuits_est:,} circuits exceeds cap. Reduce --landmarks or --train.")
        sys.exit(1)

    if args.dry_run:
        print("\n  [DRY RUN] Would execute the above. Exiting.")
        sys.exit(0)

    print()

    # ---- Load data ----
    X_train, y_train, X_test, y_test = load_tox21_data(MAX_TRAIN, MAX_TEST)

    # ---- Create backend ----
    backend = create_backend(mode)

    # ---- Run pipeline ----
    t_start = time.time()
    qsvm = QuantumHardwareSVM(backend)
    results = qsvm.fit_and_evaluate(X_train, y_train, X_test, y_test)
    t_total = time.time() - t_start

    print(f"\n  Pipeline complete in {t_total:.1f}s. ROC-AUC = {results['roc_auc']:.4f}")

    # ---- Save results ----
    try:
        import json

        results_path = os.path.join(os.path.dirname(__file__), "hardware_pipeline_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {results_path}")
    except Exception as e:
        print(f"  Could not save results: {e}")

    return results


if __name__ == "__main__":
    main()
