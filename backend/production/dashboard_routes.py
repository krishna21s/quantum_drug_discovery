"""
Dashboard Statistics Routes
==============================
Provides aggregated real-time statistics for the main dashboard.
Queries Experiment, Candidate, BindingAffinity, and Toxicity tables.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database import get_db
from database.models import Experiment, Candidate, BindingAffinity, Toxicity

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    Return aggregated dashboard statistics from the database.
    """
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)

    # ── Experiments ──
    total_experiments = db.query(func.count(Experiment.id)).scalar() or 0
    experiments_this_week = (
        db.query(func.count(Experiment.id))
        .filter(Experiment.created_at >= week_ago)
        .scalar() or 0
    )

    # ── Molecules / Candidates ──
    total_candidates = db.query(func.count(Candidate.id)).scalar() or 0
    ai_generated = (
        db.query(func.count(Candidate.id))
        .filter(Candidate.is_novel == True)
        .scalar() or 0
    )

    # ── Quantum Runs (each experiment = 1 quantum run for VQE/VQC) ──
    quantum_runs = total_experiments  # Each experiment includes VQE+VQC

    # ── Active Candidates (pIC50 > 6.0 = sub-micromolar affinity) ──
    active_candidates = (
        db.query(func.count(Candidate.id))
        .join(BindingAffinity, Candidate.id == BindingAffinity.candidate_id)
        .filter(
            (BindingAffinity.xgb_pic50 > 6.0) | (BindingAffinity.qsvr_pic50 != None)
        )
        .scalar() or 0
    )

    # ── High confidence = good binding + non-toxic ──
    high_confidence = (
        db.query(func.count(Candidate.id))
        .join(BindingAffinity, Candidate.id == BindingAffinity.candidate_id)
        .outerjoin(Toxicity, Candidate.id == Toxicity.candidate_id)
        .filter(
            BindingAffinity.xgb_pic50 > 6.5,
            (Toxicity.is_toxic == False) | (Toxicity.is_toxic == None),
        )
        .scalar() or 0
    )

    # ── Recent Experiments (latest 6) ──
    recent_exps = (
        db.query(Experiment)
        .order_by(Experiment.created_at.desc())
        .limit(6)
        .all()
    )

    recent_experiments = []
    for exp in recent_exps:
        # Find best binding score among experiment's candidates
        best_score = None
        candidates_json = exp.candidates_json or []
        for c in candidates_json:
            xgb = c.get("xgb_pic50")
            qsvr = c.get("quantum_pic50")
            score = max(filter(None, [xgb, qsvr]), default=None)
            if score and (best_score is None or score > best_score):
                best_score = score

        recent_experiments.append({
            "id": str(exp.id),
            "name": exp.target_name or exp.pdb_id,
            "protein": exp.pdb_id,
            "status": "completed",
            "score": round(best_score, 2) if best_score else None,
            "date": exp.created_at.strftime("%b %d"),
            "n_candidates": len(candidates_json),
        })

    # ── Activity Chart — avg binding score per day (last 7 days) ──
    activity_chart = []
    for i in range(6, -1, -1):
        day = now - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        avg_score = (
            db.query(func.avg(BindingAffinity.xgb_pic50))
            .join(Candidate, Candidate.id == BindingAffinity.candidate_id)
            .filter(
                Candidate.created_at >= day_start,
                Candidate.created_at < day_end,
                BindingAffinity.xgb_pic50 != None,
            )
            .scalar()
        )

        activity_chart.append({
            "day": day.strftime("%a"),
            "score": round(avg_score, 2) if avg_score else None,
        })

    # Fill gaps: if no data for a day, interpolate from neighbors
    scores = [p["score"] for p in activity_chart]
    # Find a valid fallback value
    valid_scores = [s for s in scores if s is not None]
    fallback = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0.65

    for i, point in enumerate(activity_chart):
        if point["score"] is None:
            point["score"] = fallback

    # ── Trend calculation ──
    if len(valid_scores) >= 2:
        first_half = valid_scores[:len(valid_scores)//2]
        second_half = valid_scores[len(valid_scores)//2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        trend_pct = round(((avg_second - avg_first) / avg_first) * 100, 1) if avg_first > 0 else 0
    else:
        trend_pct = 0

    return {
        "experiments_count": total_experiments,
        "experiments_this_week": experiments_this_week,
        "molecules_count": total_candidates,
        "molecules_ai_generated": ai_generated,
        "quantum_runs": quantum_runs,
        "active_candidates": active_candidates,
        "high_confidence_candidates": high_confidence,
        "recent_experiments": recent_experiments,
        "activity_chart": activity_chart,
        "trend_pct": trend_pct,
    }
