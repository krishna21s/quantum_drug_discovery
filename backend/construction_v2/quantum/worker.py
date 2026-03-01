"""
Parallel Quantum Kernel Worker
==============================
Distributes kernel matrix computation across multiple CPU cores.
Each worker instantiates its own AerSimulator to avoid serialization issues.

Uses multiprocessing.Pool (no Celery/Dask dependency).
"""

import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS, N_SHOTS, N_KERNEL_WORKERS, CHECKPOINT_EVERY_N_ROWS


def _worker_init():
    """Initialize a fresh AerSimulator per worker process."""
    global _worker_backend
    from quantum.backends import StatevectorBackend

    _worker_backend = StatevectorBackend(n_qubits=N_QUBITS, n_shots=N_SHOTS)


def _compute_row(args):
    """Compute one row of the kernel matrix: fidelity between x_i and all landmarks."""
    global _worker_backend
    i, x_i, landmarks = args
    row = np.array([_worker_backend.fidelity(x_i, lm) for lm in landmarks])
    return i, row


def _compute_entry(args):
    """Compute a single kernel entry: fidelity between x_a and x_b."""
    global _worker_backend
    i, j, x_a, x_b = args
    return i, j, _worker_backend.fidelity(x_a, x_b)


class KernelWorkerPool:
    """
    Parallel kernel computation using multiprocessing.Pool.

    Each worker instantiates its own AerSimulator to avoid
    serialization problems with Qiskit objects.
    """

    def __init__(self, n_workers=None):
        self.n_workers = n_workers or N_KERNEL_WORKERS

    def compute_rows_parallel(self, X_data, landmarks, checkpoint_path=None):
        """
        Compute K_nm matrix: each row is the fidelity between X_data[i]
        and all landmarks.

        Args:
            X_data: (N, d) array of feature vectors
            landmarks: (m, d) array of landmark vectors
            checkpoint_path: If provided, save/resume from this .npy file

        Returns:
            K: (N, m) kernel matrix
        """
        N = len(X_data)
        m = len(landmarks)

        # Resume from checkpoint if exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            K = np.load(checkpoint_path)
            completed = set(np.where(~np.isnan(K[:, 0]))[0])
            print(f"  [RESUME] Loaded {len(completed)}/{N} rows from checkpoint")
        else:
            K = np.full((N, m), np.nan)
            completed = set()

        # Build work items for incomplete rows
        work_items = [(i, X_data[i], landmarks) for i in range(N) if i not in completed]

        if not work_items:
            print("  All rows already computed, skipping.")
            return K

        t0 = time.time()
        batch_count = 0

        with Pool(processes=self.n_workers, initializer=_worker_init) as pool:
            for i, row in pool.imap_unordered(_compute_row, work_items):
                K[i] = row
                batch_count += 1

                # Checkpoint periodically
                if checkpoint_path and batch_count % CHECKPOINT_EVERY_N_ROWS == 0:
                    np.save(checkpoint_path, K)

        # Final save
        if checkpoint_path:
            np.save(checkpoint_path, K)

        elapsed = time.time() - t0
        print(
            f"  Computed {len(work_items)} rows in {elapsed:.1f}s "
            f"({len(work_items) * m} circuits, {self.n_workers} workers)"
        )

        return K

    def compute_symmetric_matrix(self, X_data, checkpoint_path=None):
        """
        Compute symmetric kernel matrix K_mm where K[i,j] = fidelity(X[i], X[j]).

        Exploits symmetry: only computes upper triangle + diagonal.

        Args:
            X_data: (m, d) array of feature vectors
            checkpoint_path: If provided, save/resume from this .npy file

        Returns:
            K: (m, m) symmetric kernel matrix
        """
        m = len(X_data)

        if checkpoint_path and os.path.exists(checkpoint_path):
            K = np.load(checkpoint_path)
            print(f"  [RESUME] Loaded K_mm from checkpoint")
        else:
            K = np.full((m, m), np.nan)

        # Build work items for upper triangle
        work_items = []
        for i in range(m):
            for j in range(i, m):
                if np.isnan(K[i, j]):
                    if i == j:
                        K[i, j] = 1.0  # Self-similarity
                    else:
                        work_items.append((i, j, X_data[i], X_data[j]))

        if not work_items:
            print("  K_mm already computed, skipping.")
            return K

        t0 = time.time()
        batch_count = 0

        with Pool(processes=self.n_workers, initializer=_worker_init) as pool:
            for i, j, fid in pool.imap_unordered(_compute_entry, work_items):
                K[i, j] = fid
                K[j, i] = fid  # Symmetry
                batch_count += 1

                if (
                    checkpoint_path
                    and batch_count % (CHECKPOINT_EVERY_N_ROWS * 10) == 0
                ):
                    np.save(checkpoint_path, K)

        if checkpoint_path:
            np.save(checkpoint_path, K)

        elapsed = time.time() - t0
        print(
            f"  K_mm: {len(work_items)} unique entries in {elapsed:.1f}s "
            f"({self.n_workers} workers)"
        )

        return K
