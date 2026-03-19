"""
Pydantic Schemas — Request/Response Models
============================================
Type-safe API contracts for the toxicity prediction endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request Models ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single molecule toxicity prediction request."""
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="SMILES string of the molecule to evaluate",
        json_schema_extra={"examples": ["CC(=O)OC1=CC=CC=C1C(=O)O"]},
    )
    enable_ci: bool = Field(
        default=False,
        description="Enable shot-based quantum evaluation with bootstrap CI",
    )
    n_bootstrap: int = Field(
        default=5,
        ge=3,
        le=20,
        description="Number of bootstrap repeats for CI estimation",
    )


class BatchPredictRequest(BaseModel):
    """Batch molecule toxicity prediction request."""
    smiles_list: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of SMILES strings (max 50)",
    )


# ── Response Models ─────────────────────────────────────────────────

class TimingInfo(BaseModel):
    """Latency breakdown for a prediction."""
    xgb_ms: float = Field(description="XGBoost inference time in milliseconds")
    quantum_s: float = Field(description="Quantum kernel inference time in seconds")
    total_s: float = Field(description="Total end-to-end time in seconds")
    shot_s: Optional[float] = Field(
        default=None,
        description="Shot-based evaluation time (if CI enabled)",
    )


class ConfidenceInterval(BaseModel):
    """Bootstrap confidence interval from shot-based evaluation."""
    probability: float = Field(description="Mean shot-based toxicity probability")
    std: float = Field(description="Standard deviation across bootstrap runs")
    ci_lower: float = Field(description="2.5th percentile (lower bound)")
    ci_upper: float = Field(description="97.5th percentile (upper bound)")
    n_bootstrap: int = Field(description="Number of bootstrap repeats")


class PredictResponse(BaseModel):
    """Single molecule toxicity prediction result."""
    smiles: str
    canonical_smiles: Optional[str] = None
    classical_probability: float = Field(
        description="XGBoost toxicity probability (0-1)"
    )
    quantum_probability: float = Field(
        description="Quantum kernel QSVM toxicity probability (0-1)"
    )
    ensemble_probability: float = Field(
        description="Hybrid ensemble toxicity probability (0-1)"
    )
    baseline_score: float = Field(
        description="Rule-based heuristic toxicity score (0-1)"
    )
    verdict: str = Field(
        description="Human-readable verdict: 'HIGH TOXICITY RISK' or 'LOW TOXICITY RISK'"
    )
    confidence: float = Field(
        description="Confidence in the verdict (0-1)"
    )
    timings: TimingInfo
    quantum_cached: bool = Field(
        default=False,
        description="Whether quantum kernel was served from cache",
    )
    ci: Optional[ConfidenceInterval] = Field(
        default=None,
        description="Confidence interval (only if enable_ci=True)",
    )
    mode: str = Field(
        description="Prediction mode: 'fast' or 'full'",
    )


class BatchPredictResponse(BaseModel):
    """Batch prediction results."""
    predictions: list[PredictResponse]
    summary: dict = Field(
        description="Aggregate statistics across the batch",
    )
    total_time_s: float


class HealthResponse(BaseModel):
    """Pipeline health check result."""
    status: str = Field(description="'healthy' or 'unhealthy'")
    version: str = Field(default="2.0.0")
    models_loaded: list[str] = Field(
        description="List of loaded model names",
    )
    pipeline_ready: bool
    checkpoint_dir: str
