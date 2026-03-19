"""
Production FastAPI Server — Toxicity Prediction
=================================================
Entry point for the production-grade hybrid quantum-classical
toxicity screening API.

Start with:
    cd backend/production
    python -m uvicorn main:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure construction_v2 is importable
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V2_DIR = os.path.join(_BACKEND_DIR, "construction_v2")
sys.path.insert(0, _V2_DIR)

from .routes import router
from .pipeline_loader import load_pipeline


# ── Lifespan: load pipeline once at startup ─────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML pipeline on startup, release on shutdown."""
    print("=" * 60)
    print("  Q-PharmX Production Server — Starting Up")
    print("=" * 60)

    try:
        pipeline, feature_svc = load_pipeline()
        app.state.pipeline = pipeline
        app.state.feature_svc = feature_svc
        print("=" * 60)
        print("  ✅ Pipeline loaded — server ready for requests")
        print("=" * 60)
    except Exception as e:
        print(f"  ❌ Pipeline loading failed: {e}")
        print("  Server will start but /predict endpoints will return 503")
        app.state.pipeline = None
        app.state.feature_svc = None

    yield  # Server is running

    # Cleanup
    print("Shutting down pipeline...")
    app.state.pipeline = None
    app.state.feature_svc = None


# ── FastAPI App ─────────────────────────────────────────────────────

app = FastAPI(
    title="Q-PharmX Toxicity Prediction API",
    description=(
        "Production-grade hybrid quantum-classical toxicity screening. "
        "Combines XGBoost (4278-d fingerprint) with a 20-qubit quantum "
        "kernel SVM (Nyström approximation) in a conservative max-alert ensemble."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",     # Vite dev server
        "http://localhost:5173",     # Vite default port
        "http://localhost:3000",     # Alternate dev port
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routes ─────────────────────────────────────────────────

app.include_router(router)


# ── Root redirect ───────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Q-PharmX Toxicity Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
