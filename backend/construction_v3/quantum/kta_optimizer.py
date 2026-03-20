"""
Quantum Kernel Alignment Optimizer (V4)
=======================================
Maximises Kernel Target Alignment (KTA) between the quantum kernel
and pIC50 labels by learning circuit parameters theta/phi.

KTA = <K, y·yᵀ>_F / (‖K‖_F · ‖y·yᵀ‖_F)

KTA ∈ [-1, 1]. Maximizing KTA directly aligns the kernel geometry
with the regression target — this is the root fix for CV R²=0.07.

Reference:
    Hubregtsen et al. (2022). "Training quantum embedding kernels on
    near-term quantum computers." Physical Review A, 106(4).

Strategy:
    1. Sample N_KTA points from training set (50–100 is enough for signal)
    2. Build mini kernel matrix K_sub(params) via statevector simulation
    3. Minimise -KTA using scipy L-BFGS-B (gradient-free)
    4. Save optimised params → use in all downstream kernel computation

Typical runtime on 8-qubit statevector:
    N_KTA=50:  ~5–10 min per 50 iterations
    N_KTA=80:  ~15–25 min per 50 iterations
"""

import os
import time
import json
import numpy as np
from scipy.optimize import minimize
from qiskit_aer import AerSimulator

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS, N_REUPLOADING_LAYERS, RANDOM_STATE, CHECKPOINT_DIR
from quantum.circuits import (
    build_reuploading_circuit, default_params, unpack_params, N_CIRCUIT_PARAMS
)


# ─────────────────────────────────────────────────────────────────────
# FAST STATEVECTOR FIDELITY (no shots — pure overlap)
# ─────────────────────────────────────────────────────────────────────

_SV_SIM = None

def _get_sim():
    global _SV_SIM
    if _SV_SIM is None:
        _SV_SIM = AerSimulator(method="statevector")
    return _SV_SIM


def fidelity_sv(x1, x2, params, n_qubits=N_QUBITS, n_shots=2048):
    """
    Shot-based fidelity using statevector backend.
    Faster than ShotBackend for KTA because we reuse the global sim.
    """
    qc = build_reuploading_circuit(x1, x2, n_qubits=n_qubits,
                                    measure=True, params=params)
    sim    = _get_sim()
    result = sim.run(qc, shots=n_shots).result()
    counts = result.get_counts()
    zeros  = "0" * n_qubits
    return counts.get(zeros, 0) / n_shots


def fidelity_batch_sv(x_query, landmark_list, params, n_qubits=N_QUBITS, n_shots=2048):
    """Batch fidelity for x_query vs all landmarks (single sim.run call)."""
    circuits = [
        build_reuploading_circuit(x_query, lm, n_qubits=n_qubits,
                                   measure=True, params=params)
        for lm in landmark_list
    ]
    sim     = _get_sim()
    results = sim.run(circuits, shots=n_shots).result()
    zeros   = "0" * n_qubits
    fids    = np.zeros(len(landmark_list), dtype=np.float32)
    for j in range(len(landmark_list)):
        counts    = results.get_counts(j)
        fids[j]   = counts.get(zeros, 0) / n_shots
    return fids


# ─────────────────────────────────────────────────────────────────────
# MINI KERNEL MATRIX
# ─────────────────────────────────────────────────────────────────────

def build_mini_kernel(X_sub, params, n_qubits=N_QUBITS, n_shots=2048, verbose=False):
    """
    Build n×n kernel matrix for X_sub using current params.

    Uses batch calls: for each row i, compute fidelity against all j>i
    in one sim.run(), then mirror.

    Args:
        X_sub:   (n, n_features) sub-sampled training features (scaled)
        params:  Flat circuit param vector
        n_shots: Shots per fidelity estimate

    Returns:
        K: (n, n) symmetric kernel matrix with diagonal=1
    """
    n = len(X_sub)
    K = np.eye(n, dtype=np.float64)

    for i in range(n):
        upper = X_sub[i+1:]
        if len(upper) == 0:
            continue
        fids     = fidelity_batch_sv(X_sub[i], upper, params,
                                      n_qubits=n_qubits, n_shots=n_shots)
        K[i, i+1:] = fids
        K[i+1:, i] = fids
        if verbose and i % 10 == 0:
            print(f"    mini-kernel row {i+1}/{n}", end="\r")

    return K


# ─────────────────────────────────────────────────────────────────────
# KERNEL TARGET ALIGNMENT
# ─────────────────────────────────────────────────────────────────────

def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Compute KTA between kernel K and label vector y.

    KTA = <K, y·yᵀ>_F / (‖K‖_F · ‖y·yᵀ‖_F)
    ∈ [-1, 1]. Higher = kernel better aligned with regression labels.

    Uses centred KTA (CKTA) which is more stable for regression:
        K_c = H·K·H  where H = I - (1/n)·11ᵀ
    """
    n = len(y)
    H = np.eye(n) - np.ones((n, n)) / n

    # Centre the kernel
    K_c = H @ K @ H

    # Centre the label kernel
    y_c = H @ np.outer(y, y) @ H

    num   = np.trace(K_c.T @ y_c)
    denom = np.linalg.norm(K_c, 'fro') * np.linalg.norm(y_c, 'fro') + 1e-10
    return float(num / denom)


# ─────────────────────────────────────────────────────────────────────
# KTA OPTIMIZER CLASS
# ─────────────────────────────────────────────────────────────────────

class KTAOptimizer:
    """
    Optimises quantum circuit parameters to maximise KTA.

    Usage:
        opt = KTAOptimizer(n_kta=60, n_shots=1024, max_iter=80)
        best_params, history = opt.fit(X_train_scaled, y_train)
        np.save("kta_params.npy", best_params)
    """

    def __init__(
        self,
        n_kta: int = 60,
        n_shots: int = 1024,
        max_iter: int = 80,
        n_restarts: int = 3,
        n_qubits: int = N_QUBITS,
        checkpoint_dir=None,
        random_state: int = RANDOM_STATE,
        verbose: bool = True,
    ):
        """
        Args:
            n_kta:         Number of training samples to use for KTA computation
                           (60-80 gives good signal; more = slower per iteration)
            n_shots:       Shots per fidelity in mini-kernel (1024 is fast + stable)
            max_iter:      L-BFGS-B iterations per restart
            n_restarts:    Number of random restarts (best params kept)
            n_qubits:      Circuit qubits (must match N_QUBITS in config)
            checkpoint_dir: Where to save params during optimisation
            random_state:  RNG seed
            verbose:       Print progress
        """
        self.n_kta          = n_kta
        self.n_shots        = n_shots
        self.max_iter       = max_iter
        self.n_restarts     = n_restarts
        self.n_qubits       = n_qubits
        self.checkpoint_dir = str(checkpoint_dir or CHECKPOINT_DIR)
        self.random_state   = random_state
        self.verbose        = verbose

        self.best_params_   = None
        self.best_kta_      = -np.inf
        self.history_       = []

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _subsample(self, X, y):
        """Stratified subsample of n_kta points covering the pIC50 range."""
        rng = np.random.RandomState(self.random_state)
        n   = min(self.n_kta, len(X))

        # Stratify by pIC50 quartile to cover the full label range
        bins    = np.percentile(y, [0, 25, 50, 75, 100])
        idx_out = []
        per_bin = max(n // 4, 1)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask    = (y >= lo) & (y <= hi)
            idxs    = np.where(mask)[0]
            chosen  = rng.choice(idxs, size=min(per_bin, len(idxs)), replace=False)
            idx_out.extend(chosen.tolist())

        idx_out = list(set(idx_out))[:n]
        if len(idx_out) < n:
            remaining = [i for i in range(len(X)) if i not in set(idx_out)]
            extra     = rng.choice(remaining, size=n - len(idx_out), replace=False)
            idx_out.extend(extra.tolist())

        idx_out = np.array(idx_out[:n])
        return X[idx_out], y[idx_out]

    def _objective(self, params, X_sub, y_sub):
        """Objective: negative centred KTA (minimise → maximise KTA)."""
        try:
            K   = build_mini_kernel(X_sub, params, n_qubits=self.n_qubits,
                                     n_shots=self.n_shots)
            kta = kernel_target_alignment(K, y_sub)
        except Exception as e:
            if self.verbose:
                print(f"\n  [WARN] objective failed: {e}")
            kta = -1.0

        self.history_.append(float(kta))
        if self.verbose and len(self.history_) % 5 == 0:
            best = max(self.history_)
            print(f"    iter {len(self.history_):4d}  KTA={kta:.4f}  best={best:.4f}",
                  end="\r")
        return -kta   # minimise

    def fit(self, X_train_scaled: np.ndarray, y_train: np.ndarray):
        """
        Run KTA optimisation.

        Args:
            X_train_scaled: (N, n_features) scaled features in [-π/2, π/2]
            y_train:        (N,) pIC50 labels

        Returns:
            best_params: (N_CIRCUIT_PARAMS,) optimised flat param vector
            history:     list of KTA values per iteration
        """
        if self.verbose:
            print(f"\n[KTA] Subsampling {self.n_kta} points from {len(X_train_scaled)}...")

        X_sub, y_sub = self._subsample(X_train_scaled, y_train)

        if self.verbose:
            print(f"  Sub-sample shape: {X_sub.shape}")
            print(f"  y_sub range: [{y_sub.min():.2f}, {y_sub.max():.2f}]")
            print(f"  Running {self.n_restarts} restarts × {self.max_iter} iter...")

        rng = np.random.RandomState(self.random_state + 100)

        # ── Restart 0: from default params (theta=1, phi=0) ──
        init_candidates = [default_params()]

        # ── Restarts 1..: small random perturbations ──
        for k in range(1, self.n_restarts):
            p = default_params()
            # Perturb theta around 1.0 (scale matters most)
            n_theta = N_QUBITS * N_REUPLOADING_LAYERS
            p[:n_theta] += rng.normal(0, 0.3, size=n_theta)
            p[n_theta:] += rng.normal(0, 0.1, size=n_theta)
            init_candidates.append(p)

        # L-BFGS-B bounds: theta in [0.1, 3.0], phi in [-π, π]
        n_theta = N_QUBITS * N_REUPLOADING_LAYERS
        bounds  = (
            [(0.1, 3.0)] * n_theta        # theta: scale must stay positive
          + [(-np.pi, np.pi)] * n_theta   # phi: full rotation range
        )

        for restart, p0 in enumerate(init_candidates):
            if self.verbose:
                print(f"\n  Restart {restart+1}/{self.n_restarts} "
                      f"(init KTA = {-self._objective(p0, X_sub, y_sub):.4f})")

            t0     = time.time()
            result = minimize(
                self._objective,
                x0      = p0,
                args    = (X_sub, y_sub),
                method  = "L-BFGS-B",
                bounds  = bounds,
                options = {"maxiter": self.max_iter, "ftol": 1e-6, "gtol": 1e-5},
            )

            final_kta = -result.fun
            elapsed   = (time.time() - t0) / 60

            if self.verbose:
                print(f"\n  Restart {restart+1} done: KTA={final_kta:.4f}  "
                      f"({result.nit} iters, {elapsed:.1f} min)")

            if final_kta > self.best_kta_:
                self.best_kta_    = final_kta
                self.best_params_ = result.x.copy()

                # Save intermediate best
                ckpt_path = os.path.join(self.checkpoint_dir,
                                         "kta_params_best.npy")
                np.save(ckpt_path, self.best_params_)
                if self.verbose:
                    print(f"  [SAVED] New best params → {ckpt_path}")

        # Final save
        final_path = os.path.join(self.checkpoint_dir, "kta_params_final.npy")
        np.save(final_path, self.best_params_)

        # Save KTA history
        hist_path = os.path.join(self.checkpoint_dir, "kta_history.json")
        with open(hist_path, "w") as f:
            json.dump({"history": self.history_, "best_kta": self.best_kta_}, f)

        if self.verbose:
            print(f"\n[KTA] Optimisation complete.")
            print(f"  Best KTA achieved: {self.best_kta_:.4f}")
            print(f"  (KTA=0 means random kernel; KTA>0.1 means useful alignment)")
            print(f"  Params saved to: {final_path}")

        return self.best_params_, self.history_

    def load_params(self):
        """Load previously saved best params."""
        path = os.path.join(self.checkpoint_dir, "kta_params_final.npy")
        if not os.path.exists(path):
            path = os.path.join(self.checkpoint_dir, "kta_params_best.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No KTA params found in {self.checkpoint_dir}.\n"
                "Run KTAOptimizer.fit() first."
            )
        params = np.load(path)
        self.best_params_ = params
        return params

    def report_param_shift(self, params=None):
        """Show how much params deviate from fixed baseline (theta=1, phi=0)."""
        p     = params if params is not None else self.best_params_
        if p is None:
            print("  No params to report.")
            return
        n_theta = N_QUBITS * N_REUPLOADING_LAYERS
        theta   = p[:n_theta].reshape(N_REUPLOADING_LAYERS, N_QUBITS)
        phi     = p[n_theta:].reshape(N_REUPLOADING_LAYERS, N_QUBITS)
        print(f"\n  Theta (scale) — mean: {theta.mean():.3f}  std: {theta.std():.3f}  "
              f"range: [{theta.min():.3f}, {theta.max():.3f}]")
        print(f"  Phi   (bias)  — mean: {phi.mean():.3f}  std: {phi.std():.3f}  "
              f"range: [{phi.min():.3f}, {phi.max():.3f}]")
        print(f"\n  Per-layer theta means: "
              + "  ".join([f"L{i}={theta[i].mean():.3f}" for i in range(N_REUPLOADING_LAYERS)]))
