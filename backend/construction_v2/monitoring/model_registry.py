"""
Model Registry — Version Tracking for Model Artifacts
=======================================================
Tracks model artifacts with version, timestamp, training metrics,
and config hash for reproducibility.
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR


class ModelRegistry:
    """
    Lightweight model registry for tracking checkpoint versions.

    Stores metadata about each registered model artifact:
      - model_name, version, timestamp
      - training metrics (AUC, Brier, etc.)
      - config hash (for reproducibility verification)
      - file path and file hash
    """

    def __init__(self, registry_dir=None):
        self.registry_dir = Path(registry_dir or CHECKPOINT_DIR)
        self.registry_path = self.registry_dir / "model_registry.json"
        self._registry = self._load()

    def _load(self):
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"models": {}}

    def _save(self):
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def register(self, model_name, file_path, metrics=None, config=None, notes=""):
        """
        Register a model artifact.

        Args:
            model_name: e.g., 'xgboost_v2', 'qsvm', 'gnn'
            file_path: Path to the model file
            metrics: Dict of training metrics
            config: Dict of training config (for hash)
            notes: Free-text notes
        """
        file_path = str(file_path)
        file_hash = self._file_hash(file_path) if os.path.exists(file_path) else None
        config_hash = hashlib.md5(
            json.dumps(config or {}, sort_keys=True).encode()
        ).hexdigest()

        # Auto-increment version
        existing = self._registry["models"].get(model_name, [])
        version = len(existing) + 1

        entry = {
            "version": version,
            "file_path": file_path,
            "file_hash": file_hash,
            "config_hash": config_hash,
            "metrics": metrics or {},
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
        }

        if model_name not in self._registry["models"]:
            self._registry["models"][model_name] = []
        self._registry["models"][model_name].append(entry)

        self._save()
        return entry

    def get_latest(self, model_name):
        """Get the latest version of a model."""
        versions = self._registry["models"].get(model_name, [])
        return versions[-1] if versions else None

    def list_models(self):
        """List all registered models with their latest version."""
        return {
            name: versions[-1]["version"]
            for name, versions in self._registry["models"].items()
            if versions
        }

    def _file_hash(self, path, chunk_size=65536):
        """Compute MD5 hash of a file."""
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
