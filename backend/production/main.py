"""
Production FastAPI Server — Q-PharmX Drug Discovery Platform
==============================================================
Entry point for the production-grade hybrid quantum-classical
drug discovery API. Integrates:
  - Toxicity screening (V2) — XGBoost + 20-qubit QSVM
  - Drug candidate generation (V4) — RL-tuned CharRNN
  - Binding affinity scoring (V3 QSVR) — Dual XGB + quantum SVR

Start with:
    cd backend
    python -m uvicorn production.main:app --reload --port 8000
"""

import os
import sys
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Ensure all construction packages are importable ─────────
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V2_DIR = os.path.join(_BACKEND_DIR, "construction_v2")
_V3_DIR = os.path.join(_BACKEND_DIR, "construction_v3")
_V4_DIR = os.path.join(_BACKEND_DIR, "construction_v4")
sys.path.insert(0, _V2_DIR)
sys.path.insert(0, _V3_DIR)
sys.path.insert(0, _V4_DIR)

from .routes import router as toxicity_router
from .candidates_routes import router as candidates_router
from .binding_routes import router as binding_router
from .db_routes import router as db_router
from .admet_routes import router as admet_router
from .refinement_routes import router as refinement_router
from .pipeline_loader import load_pipeline

# Database Imports
sys.path.insert(0, _BACKEND_DIR)  # Ensure backend root is in path structure
from database.database import engine, Base
import database.models


# ── Lifespan: load all pipelines at startup ─────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML pipelines on startup, release on shutdown."""
    print("=" * 60)
    print("  Q-PharmX Production Server — Starting Up")
    print("=" * 60)
    
    # 0. Initialize Database Tables
    try:
        Base.metadata.create_all(bind=engine)
        print("  ✅ Database tables initialized successfully")
    except Exception as e:
        print(f"  ⚠️  Database alignment failed: {e}")


    # 1. Toxicity pipeline (V2)
    try:
        pipeline, feature_svc = load_pipeline()
        app.state.pipeline = pipeline
        app.state.feature_svc = feature_svc
        print("  ✅ Toxicity pipeline (V2) loaded")
    except Exception as e:
        print(f"  ⚠️  Toxicity pipeline failed: {e}")
        app.state.pipeline = None
        app.state.feature_svc = None

    # 2. Drug candidates (V4) — load pre-computed JSON
    try:
        candidates_path = os.path.join(
            _V4_DIR, "checkpoints", "final_candidates.json"
        )
        if os.path.exists(candidates_path):
            with open(candidates_path, "r") as f:
                app.state.candidates_data = json.load(f)
            n = len(app.state.candidates_data.get("candidates", []))
            print(f"  ✅ Drug candidates (V4) loaded — {n} candidates")
        else:
            app.state.candidates_data = None
            print("  ⚠️  Candidates not found (run RL fine-tuning first)")
    except Exception as e:
        print(f"  ⚠️  Candidates loading failed: {e}")
        app.state.candidates_data = None

    # 3. Binding affinity oracle (V3 QSVR + XGB)
    try:
        from oracle.quantum_oracle import QuantumOracle
        app.state.binding_oracle = QuantumOracle()
        mode = "quantum" if app.state.binding_oracle.is_quantum_mode else "QSVR-RBF"
        print(f"  ✅ Binding oracle (V3) loaded — mode: {mode}")
    except Exception as e:
        print(f"  ⚠️  Binding oracle failed: {e}")
        app.state.binding_oracle = None

    # 4. Live generator (RL-trained ConditionedRNN)
    try:
        from .generator_service import GeneratorService
        app.state.generator = GeneratorService(
            binding_oracle=app.state.binding_oracle
        )
        print(f"  ✅ Generator (V4 RL) loaded — ready for on-demand generation")
    except Exception as e:
        print(f"  ⚠️  Generator loading failed: {e}")
        app.state.generator = None

    print("=" * 60)
    loaded = sum([
        app.state.pipeline is not None,
        app.state.candidates_data is not None,
        app.state.binding_oracle is not None,
        app.state.generator is not None,
    ])
    print(f"  {loaded}/4 systems ready — server starting")
    print("=" * 60)

    yield  # Server is running

    # Cleanup
    print("Shutting down pipelines...")
    app.state.pipeline = None
    app.state.feature_svc = None
    app.state.candidates_data = None
    app.state.binding_oracle = None


# ── FastAPI App ─────────────────────────────────────────────

app = FastAPI(
    title="Q-PharmX Drug Discovery API",
    description=(
        "Production hybrid quantum-classical drug discovery platform. "
        "Toxicity screening (V2), RL-based candidate generation (V4), "
        "and dual-oracle binding affinity prediction (XGB + QSVR)."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register All Routes ─────────────────────────────────────

app.include_router(toxicity_router)
app.include_router(candidates_router)
app.include_router(binding_router)
app.include_router(db_router)
app.include_router(admet_router)
app.include_router(refinement_router)


# ── Root ────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Q-PharmX Drug Discovery API",
        "version": "3.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "toxicity": "/api/predict",
            "candidates": "/api/candidates",
            "binding": "/api/binding/score",
        },
    }
