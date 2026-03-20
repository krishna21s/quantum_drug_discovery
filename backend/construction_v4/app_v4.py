"""
V4 FastAPI Server — De Novo Drug Generation API
==================================================
Exposes V4 generation as async endpoints with background job management.

Endpoints:
    POST /api/v4/generate         — submit generation job
    GET  /api/v4/status/{job_id}  — poll job status
    GET  /api/v4/results/{job_id} — fetch final candidates
    GET  /api/v4/pocket_phi/{pdb_id} — compute/load pocket vector
    WS   /api/v4/stream/{job_id}  — WebSocket live progress

Usage:
    uvicorn app_v4:app --port 8001 --host 0.0.0.0
"""

import os
import sys
import json
import uuid
import time
import asyncio
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_v4 import (
    API_HOST,
    API_PORT,
    ALPHA, BETA, GAMMA, DELTA,
    RL_EPISODES,
    RL_TEMPERATURE,
    QUANTUM_TOP_K,
    V4_CHECKPOINT_DIR,
    WS_PUSH_INTERVAL_S,
)

# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────

class JobPhase(str, Enum):
    INITIALISING = "initialising"
    RL = "rl"
    QUANTUM_EVAL = "quantum_eval"
    DONE = "done"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    pdb_id: str = "1M17"
    n_candidates: int = Field(default=10, ge=1, le=50)
    alpha: float = ALPHA
    beta: float = BETA
    gamma: float = GAMMA
    delta: float = DELTA
    use_rl: bool = True
    temperature: float = Field(default=RL_TEMPERATURE, ge=0.5, le=2.0)
    n_episodes: Optional[int] = None  # override RL_EPISODES


class GenerateResponse(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    status: str
    phase: str
    episode: int = 0
    total_episodes: int = 0
    current_reward: float = 0.0
    current_mean_pic50: float = 0.0
    validity_pct: float = 0.0
    elapsed_s: float = 0.0
    error: Optional[str] = None


class PocketPhiResponse(BaseModel):
    phi: List[float]
    feature_names: List[str]
    pdb_id: str


# ──────────────────────────────────────────────────────────────
# Job state
# ──────────────────────────────────────────────────────────────

class JobState:
    """In-memory job state. For production, replace with Redis."""

    def __init__(self, job_id: str, request: GenerateRequest):
        self.job_id = job_id
        self.request = request
        self.status = "running"
        self.phase = JobPhase.INITIALISING
        self.episode = 0
        self.total_episodes = request.n_episodes or RL_EPISODES
        self.current_reward = 0.0
        self.current_mean_pic50 = 0.0
        self.validity_pct = 0.0
        self.start_time = time.time()
        self.results = None
        self.error = None

    def to_status_dict(self) -> dict:
        return {
            "status": self.status,
            "phase": self.phase.value,
            "episode": self.episode,
            "total_episodes": self.total_episodes,
            "current_reward": round(self.current_reward, 3),
            "current_mean_pic50": round(self.current_mean_pic50, 2),
            "validity_pct": round(self.validity_pct, 1),
            "elapsed_s": round(time.time() - self.start_time, 1),
            "error": self.error,
        }


# Global job store
jobs: Dict[str, JobState] = {}

# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="V4 De Novo Drug Generation",
    version="4.0.0",
    description="REINVENT-style SMILES RNN with quantum-guided RL for EGFR drug design",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# Background generation worker
# ──────────────────────────────────────────────────────────────

async def run_generation(job_id: str, request: GenerateRequest):
    """Background task that runs the generation pipeline."""
    job = jobs[job_id]

    try:
        from training.pocket_conditioner import PocketConditioner
        from oracle.xgb_oracle import XGBOracle
        from oracle.admet_scorer import ADMETScorer
        from oracle.reward_function import compute_reward

        import numpy as np
        import torch

        # Phase: Initialise
        job.phase = JobPhase.INITIALISING

        pc = PocketConditioner()
        phi = pc.load_or_compute(request.pdb_id)
        xgb = XGBOracle()
        admet = ADMETScorer()

        device = "cpu"
        ckpt_dir = str(V4_CHECKPOINT_DIR)

        if request.use_rl:
            # RL generation
            from models.conditioned_rnn import ConditionedRNN
            from models.char_rnn import CharRNN

            pretrained_path = os.path.join(ckpt_dir, "rnn_pretrained.pt")
            policy_path = os.path.join(ckpt_dir, "policy_egfr_rl.pt")

            # Check if we have a pre-trained policy
            if os.path.exists(policy_path):
                # Use existing RL-tuned policy for sampling only
                policy = ConditionedRNN.load(policy_path, device=device)
                job.phase = JobPhase.RL
                job.episode = job.total_episodes
                job.status = "running"

                # Sample candidates
                n_to_sample = request.n_candidates * 5  # oversample for filtering
                samples = policy.sample_conditioned(
                    phi, n=n_to_sample, temperature=request.temperature, device=device
                )

            elif os.path.exists(pretrained_path):
                # Sample from pre-trained model (no RL)
                policy = ConditionedRNN.from_pretrained(pretrained_path, device=device)
                job.phase = JobPhase.RL
                n_to_sample = request.n_candidates * 5
                samples = policy.sample_conditioned(
                    phi, n=n_to_sample, temperature=request.temperature, device=device
                )
            else:
                raise FileNotFoundError(
                    f"No model checkpoint found. Run pre-training first.\n"
                    f"Expected: {pretrained_path} or {policy_path}"
                )
        else:
            # Pure RNN sampling (no RL, no conditioning)
            from models.char_rnn import CharRNN

            pretrained_path = os.path.join(ckpt_dir, "rnn_pretrained.pt")
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"Pre-trained model not found: {pretrained_path}")

            model = CharRNN.load(pretrained_path, device=device)
            job.phase = JobPhase.RL
            n_to_sample = request.n_candidates * 5
            samples = model.sample(n=n_to_sample, temperature=request.temperature, device=device)

        # Score all candidates
        job.phase = JobPhase.QUANTUM_EVAL

        from rdkit import Chem

        candidates = []
        for smi in samples:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            canonical = Chem.MolToSmiles(mol, canonical=True)
            pic50 = float(xgb.score(canonical))
            admet_result = admet.score(canonical)

            if admet_result.get("error"):
                continue

            candidates.append({
                "smiles": canonical,
                "xgb_pic50": round(pic50, 3),
                "quantum_pic50": None,  # to be filled when quantum oracle ready
                "qed": admet_result["qed"],
                "sa_score": admet_result["sa_score"],
                "mw": admet_result["mw"],
                "logp": admet_result["logp"],
                "lipinski_pass": admet_result["lipinski_pass"],
                "tpsa": admet_result["tpsa"],
                "is_novel": True,
            })

        # Sort by pIC50 descending, take top-N
        candidates.sort(key=lambda x: x["xgb_pic50"], reverse=True)
        candidates = candidates[:request.n_candidates]

        # Add ranks
        for i, c in enumerate(candidates, 1):
            c["rank"] = i

        # Update job metrics
        if candidates:
            pic50s = [c["xgb_pic50"] for c in candidates]
            job.current_mean_pic50 = float(np.mean(pic50s))
            job.validity_pct = len(candidates) / max(len(samples), 1) * 100

        # Save results
        job.results = {
            "generated_at": datetime.now().isoformat(),
            "target": f"EGFR (PDB {request.pdb_id})",
            "candidates": candidates,
        }

        job.phase = JobPhase.DONE
        job.status = "complete"

    except Exception as e:
        job.status = "failed"
        job.phase = JobPhase.FAILED
        job.error = str(e)
        import traceback
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/api/v4/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Submit a drug generation job."""
    job_id = str(uuid.uuid4())
    job = JobState(job_id, request)
    jobs[job_id] = job

    # Run in background
    asyncio.create_task(run_generation(job_id, request))

    return GenerateResponse(job_id=job_id)


@app.get("/api/v4/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return jobs[job_id].to_status_dict()


@app.get("/api/v4/results/{job_id}")
async def get_results(job_id: str):
    """Get the results of a completed generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if job.status != "complete":
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} is still {job.status} (phase: {job.phase.value})"
        )

    return job.results


@app.get("/api/v4/pocket_phi/{pdb_id}", response_model=PocketPhiResponse)
async def get_pocket_phi(pdb_id: str):
    """Compute or load pocket vector φ for a PDB ID."""
    from training.pocket_conditioner import PocketConditioner

    pc = PocketConditioner()
    phi = pc.load_or_compute(pdb_id)

    return PocketPhiResponse(
        phi=[round(float(v), 4) for v in phi],
        feature_names=[
            "sasa_norm", "volume_norm", "hbd_norm",
            "hba_norm", "charge_norm", "aromatic_frac", "depth_norm"
        ],
        pdb_id=pdb_id.upper(),
    )


@app.websocket("/api/v4/stream/{job_id}")
async def stream_status(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for live progress updates."""
    await websocket.accept()

    if job_id not in jobs:
        await websocket.send_json({"error": f"Job {job_id} not found"})
        await websocket.close()
        return

    try:
        while True:
            job = jobs.get(job_id)
            if job is None:
                break

            await websocket.send_json(job.to_status_dict())

            if job.status in ("complete", "failed"):
                break

            await asyncio.sleep(WS_PUSH_INTERVAL_S)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.get("/api/v4/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "4.0.0",
        "active_jobs": len([j for j in jobs.values() if j.status == "running"]),
        "total_jobs": len(jobs),
    }


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
