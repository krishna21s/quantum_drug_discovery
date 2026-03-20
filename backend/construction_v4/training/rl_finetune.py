"""
RL Fine-tuning — REINFORCE Policy Gradient for EGFR Drug Generation
=====================================================================
Fine-tunes the pre-trained SMILES RNN using REINFORCE with:
  - XGBoost fast oracle for pIC50 scoring
  - ADMET multi-property reward
  - Tanimoto diversity penalty
  - KL regularisation against frozen prior
  - Pocket conditioning on EGFR 7D φ vector

After RL converges, selects top-K candidates and optionally scores
them with the quantum oracle.

Usage:
    # Full RL fine-tuning:
    python training/rl_finetune.py \\
        --pretrained-checkpoint checkpoints/rnn_pretrained.pt

    # Quick smoke test (50 episodes):
    python training/rl_finetune.py \\
        --pretrained-checkpoint checkpoints/rnn_pretrained.pt \\
        --n-episodes 50 --skip-quantum

    # Evaluate distribution shift (no training):
    python training/rl_finetune.py \\
        --eval-distribution \\
        --checkpoint checkpoints/policy_egfr_rl.pt
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_v4 import (
    RL_EPISODES,
    RL_BATCH_SIZE,
    RL_LR,
    RL_BASELINE_DECAY,
    RL_KL_WEIGHT,
    RL_TEMPERATURE,
    RL_GRAD_CLIP,
    RL_LOG_EVERY,
    RL_PLATEAU_PATIENCE,
    QUANTUM_TOP_K,
    SA_SCORE_CUTOFF,
    QUANTUM_DIVERSITY_MAX,
    V4_CHECKPOINT_DIR,
    EGFR_PHI_PATH,
    RANDOM_STATE,
)
from models.conditioned_rnn import ConditionedRNN
from models.char_rnn import CharRNN
from oracle.xgb_oracle import XGBOracle
from oracle.admet_scorer import ADMETScorer
from oracle.reward_function import compute_reward
from training.pocket_conditioner import PocketConditioner

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


def plateau_detected(reward_history: list, patience: int = RL_PLATEAU_PATIENCE) -> bool:
    """Detect if reward has plateaued over the last `patience` episodes."""
    if len(reward_history) < patience + 10:
        return False
    recent = reward_history[-patience:]
    older = reward_history[-(patience + 10):-patience]
    if not older:
        return False
    improvement = np.mean(recent) - np.mean(older)
    return improvement < 0.005  # less than 0.5% improvement


def select_top_k(
    all_results: list,
    k: int = QUANTUM_TOP_K,
    max_tanimoto: float = QUANTUM_DIVERSITY_MAX,
) -> List[str]:
    """
    Select top-K diverse candidates from all generated molecules.

    Filters:
        1. Valid SMILES
        2. Lipinski pass
        3. SA score ≤ cutoff
        4. Ranked by XGB pIC50 descending
        5. Tanimoto diversity filter (max pairwise < threshold)
    """
    # Sort by pIC50 descending
    valid = [r for r in all_results if r.get("valid") and r.get("lipinski_pass")]
    valid = [r for r in valid if r.get("sa_score", 10) <= SA_SCORE_CUTOFF]
    valid.sort(key=lambda x: x.get("pic50", 0), reverse=True)

    if not valid:
        print("  [WARNING] No valid candidates passed filters")
        return []

    # Diversity filter using Tanimoto
    selected = []
    selected_fps = []

    for r in valid:
        if len(selected) >= k:
            break

        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

        # Check diversity against already selected
        too_similar = False
        for sel_fp in selected_fps:
            sim = DataStructs.TanimotoSimilarity(fp, sel_fp)
            if sim > max_tanimoto:
                too_similar = True
                break

        if not too_similar:
            selected.append(r)
            selected_fps.append(fp)

    return [r["smiles"] for r in selected]


def evaluate_distribution(model, phi, device, n_samples=512, label=""):
    """Sample molecules and report distribution stats."""
    print(f"\n  {label} Distribution Analysis ({n_samples} samples):")

    if isinstance(model, ConditionedRNN):
        samples = model.sample_conditioned(phi, n=n_samples, temperature=1.0, device=device)
    else:
        samples = model.sample(n=n_samples, temperature=1.0, device=device)

    # Validity
    valid_mols = []
    for smi in samples:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mols.append(Chem.MolToSmiles(mol, canonical=True))

    validity = len(valid_mols) / max(len(samples), 1)
    uniqueness = len(set(valid_mols)) / max(len(valid_mols), 1)

    print(f"    Validity:   {validity:.1%} ({len(valid_mols)}/{len(samples)})")
    print(f"    Uniqueness: {uniqueness:.1%}")

    # pIC50 distribution (using XGB)
    try:
        oracle = XGBOracle()
        pic50s = oracle.score_batch(valid_mols[:200])
        print(f"    Mean pIC50: {np.mean(pic50s):.2f} ± {np.std(pic50s):.2f}")
        print(f"    pIC50 > 6:  {(pic50s > 6).sum()}/{len(pic50s)}")
        print(f"    pIC50 > 7:  {(pic50s > 7).sum()}/{len(pic50s)}")
    except Exception as e:
        print(f"    pIC50: could not compute ({e})")
        pic50s = np.array([0])

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "mean_pic50": float(np.mean(pic50s)),
        "samples": valid_mols[:10],
    }


def run_rl_finetuning(args):
    """Main RL fine-tuning loop."""
    device = args.device
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  RL Fine-tuning — EGFR Drug Generation")
    print(f"  Device:   {device}")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Batch:    {RL_BATCH_SIZE}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # ── Load pocket vector ──
    pc = PocketConditioner(checkpoint_dir=ckpt_dir)
    phi = pc.load_or_compute("1M17")
    print(f"  Pocket φ: {phi.round(3)}")

    # ── Load policy (conditioned RNN from pre-trained) ──
    print(f"\n  Loading pre-trained model: {args.pretrained_checkpoint}")
    policy = ConditionedRNN.from_pretrained(
        args.pretrained_checkpoint, device=device
    )
    policy.to(device)
    policy.train()

    # ── Load frozen prior (unconditioned CharRNN) ──
    prior = CharRNN.load(args.pretrained_checkpoint, device=device)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad = False
    print(f"  Prior loaded and frozen")

    # ── Oracles ──
    xgb = XGBOracle()
    admet = ADMETScorer()
    print(f"  Oracles ready (XGB + ADMET)")

    # ── Optimizer ──
    optimizer = torch.optim.Adam(policy.parameters(), lr=RL_LR)

    # ── Training state ──
    baseline = 0.0
    reward_history = []
    all_generated = []  # stores all generated molecule results

    print(f"\n{'─'*60}")
    print(f"  Starting REINFORCE training...")
    print(f"{'─'*60}\n")

    t_start = time.time()

    for episode in range(args.n_episodes):
        t_ep = time.time()

        # 1. Sample batch from policy
        smiles_batch, log_probs = policy.sample_conditioned_with_logprobs(
            phi, n=RL_BATCH_SIZE, temperature=RL_TEMPERATURE, device=device
        )

        # 2. Score batch with fast XGB oracle
        pic50_batch = xgb.score_batch(smiles_batch)

        # 3. Score ADMET
        admet_batch = [admet.score(s) for s in smiles_batch]

        # 4. Compute rewards
        rewards = []
        for s, p, a in zip(smiles_batch, pic50_batch, admet_batch):
            r = compute_reward(s, float(p), a, smiles_batch)
            rewards.append(r)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

        # 5. KL regularisation vs prior
        with torch.no_grad():
            prior_log_probs = prior.log_prob_batch(smiles_batch, device=device)
        kl_penalty = (log_probs - prior_log_probs).mean()

        # 6. REINFORCE with baseline
        baseline = RL_BASELINE_DECAY * baseline + (1 - RL_BASELINE_DECAY) * rewards_t.mean().item()
        advantages = rewards_t - baseline
        policy_loss = -(advantages.detach() * log_probs).mean()
        loss = policy_loss + RL_KL_WEIGHT * kl_penalty

        # 7. Update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=RL_GRAD_CLIP)
        optimizer.step()

        # 8. Track results
        mean_reward = rewards_t.mean().item()
        reward_history.append(mean_reward)

        # Store generated molecules
        for s, p, a in zip(smiles_batch, pic50_batch, admet_batch):
            mol = Chem.MolFromSmiles(s) if Chem else None
            all_generated.append({
                "smiles": s,
                "pic50": float(p),
                "valid": mol is not None,
                "lipinski_pass": a.get("lipinski_pass", False) if a.get("error") is None else False,
                "sa_score": a.get("sa_score", 10) if a.get("error") is None else 10,
                "qed": a.get("qed", 0) if a.get("error") is None else 0,
            })

        # 9. Logging
        if episode % RL_LOG_EVERY == 0:
            valid_mols = [s for s in smiles_batch if Chem.MolFromSmiles(s)]
            valid_pic50s = [float(p) for s, p in zip(smiles_batch, pic50_batch) if Chem.MolFromSmiles(s) and p > 2.0]

            mean_pic50 = np.mean(valid_pic50s) if valid_pic50s else 0
            validity = len(valid_mols) / len(smiles_batch)

            elapsed = time.time() - t_start
            print(
                f"  ep={episode:4d}  "
                f"reward={mean_reward:.3f}  "
                f"pIC50={mean_pic50:.2f}  "
                f"valid={validity:.0%}  "
                f"kl={kl_penalty.item():.3f}  "
                f"loss={loss.item():.3f}  "
                f"elapsed={elapsed:.0f}s"
            )

        # 10. Early stopping
        if plateau_detected(reward_history):
            print(f"\n  Early stop at episode {episode} (reward plateau)")
            break

    # ── Save policy ──
    total_time = time.time() - t_start
    policy_path = os.path.join(ckpt_dir, "policy_egfr_rl.pt")
    policy.save(policy_path)

    print(f"\n{'='*60}")
    print(f"  RL Fine-tuning Complete")
    print(f"  Episodes: {len(reward_history)}")
    print(f"  Final reward: {reward_history[-1]:.3f}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Policy: {policy_path}")
    print(f"{'='*60}")

    # ── Phase 4: Select top-K and (optionally) quantum eval ──
    print(f"\n  Selecting top-{QUANTUM_TOP_K} diverse candidates...")
    top_smiles = select_top_k(all_generated, k=QUANTUM_TOP_K)
    print(f"  Selected {len(top_smiles)} candidates")

    # Build final results
    final_candidates = []
    for rank, smi in enumerate(top_smiles, 1):
        pic50 = float(xgb.score(smi))
        admet_result = admet.score(smi)

        candidate = {
            "rank": rank,
            "smiles": smi,
            "xgb_pic50": pic50,
            "quantum_pic50": None,  # filled by quantum oracle when available
            "qed": admet_result.get("qed"),
            "sa_score": admet_result.get("sa_score"),
            "mw": admet_result.get("mw"),
            "logp": admet_result.get("logp"),
            "lipinski_pass": admet_result.get("lipinski_pass"),
            "tpsa": admet_result.get("tpsa"),
            "is_novel": True,
        }

        # Quantum scoring if available
        if not args.skip_quantum:
            try:
                from oracle.quantum_oracle import QuantumOracle
                q_oracle = QuantumOracle()
                q_result = q_oracle.score(smi)
                candidate["quantum_pic50"] = q_result.get("pic50")
                candidate["quantum_mode"] = q_result.get("mode")
            except Exception as e:
                print(f"    Quantum scoring skipped: {e}")

        final_candidates.append(candidate)

    # Save final results
    final_output = {
        "generated_at": datetime.now().isoformat(),
        "target": "EGFR (PDB 1M17)",
        "n_rl_episodes": len(reward_history),
        "total_generated": len(all_generated),
        "total_time_min": round(total_time / 60, 1),
        "final_reward": round(reward_history[-1], 3) if reward_history else 0,
        "candidates": final_candidates,
    }

    results_path = os.path.join(ckpt_dir, "final_candidates.json")
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Print top-5
    print(f"\n  Top-5 candidates:")
    for c in final_candidates[:5]:
        q_str = f"  Q={c['quantum_pic50']:.2f}" if c['quantum_pic50'] else ""
        print(
            f"    #{c['rank']:2d}: {c['smiles'][:50]:50s}  "
            f"XGB={c['xgb_pic50']:.2f}{q_str}  "
            f"QED={c['qed']:.2f}  SA={c['sa_score']:.1f}"
        )


def run_eval_distribution(args):
    """Evaluate distribution shift between pre-trained and RL-tuned model."""
    device = args.device
    pc = PocketConditioner()
    phi = pc.load_or_compute("1M17")

    n_samples = args.n_samples

    # Pre-RL model
    print(f"  Loading pre-trained model: {args.pretrained_checkpoint}")
    pre_model = CharRNN.load(args.pretrained_checkpoint, device=device)
    pre_stats = evaluate_distribution(pre_model, phi, device, n_samples, "Pre-RL")

    # Post-RL model
    if args.checkpoint:
        print(f"\n  Loading RL-tuned model: {args.checkpoint}")
        post_model = ConditionedRNN.load(args.checkpoint, device=device)
        post_stats = evaluate_distribution(post_model, phi, device, n_samples, "Post-RL")

        print(f"\n{'='*50}")
        print(f"  Distribution Shift Summary:")
        print(f"    pIC50: {pre_stats['mean_pic50']:.2f} → {post_stats['mean_pic50']:.2f}")
        print(f"    Validity: {pre_stats['validity']:.1%} → {post_stats['validity']:.1%}")
        print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="REINFORCE RL Fine-tuning")
    parser.add_argument("--pretrained-checkpoint", type=str,
                        default=str(V4_CHECKPOINT_DIR / "rnn_pretrained.pt"),
                        help="Path to pre-trained CharRNN checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to RL-tuned model (for --eval-distribution)")
    parser.add_argument("--n-episodes", type=int, default=RL_EPISODES)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default=str(V4_CHECKPOINT_DIR))
    parser.add_argument("--skip-quantum", action="store_true",
                        help="Skip quantum oracle evaluation")
    parser.add_argument("--eval-distribution", action="store_true",
                        help="Evaluate distribution shift (no training)")
    parser.add_argument("--n-samples", type=int, default=512,
                        help="Samples for distribution evaluation")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    if args.eval_distribution:
        run_eval_distribution(args)
    else:
        run_rl_finetuning(args)


if __name__ == "__main__":
    main()
