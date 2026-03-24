"""
LLM Service — Groq/Llama Auto-Configure for Experiments
==========================================================
Uses the Groq API with Llama 3.3 70B to intelligently set
experiment parameters based on protein target analysis.
"""

import os
import json
import logging
from typing import Optional

import numpy as np

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    _BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(_BACKEND_DIR, ".env"))
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Known protein targets for enriched context
PROTEIN_DATABASE = {
    "6LU7": {"name": "SARS-CoV-2 Main Protease (Mpro)", "disease": "COVID-19", "type": "cysteine protease"},
    "1M17": {"name": "EGFR Kinase Domain", "disease": "Non-small cell lung cancer", "type": "tyrosine kinase"},
    "1HHP": {"name": "HIV-1 Protease", "disease": "HIV/AIDS", "type": "aspartic protease"},
    "1ZG4": {"name": "Beta-Lactamase (TEM-1)", "disease": "Antibiotic resistance", "type": "serine hydrolase"},
    "3ERT": {"name": "Estrogen Receptor Alpha", "disease": "Breast cancer", "type": "nuclear receptor"},
}

PHI_FEATURE_NAMES = ["SASA", "Volume", "HBD_count", "HBA_count", "Net_charge", "Aromatic_fraction", "Pocket_depth"]


def auto_configure_experiment(pdb_id: str, phi: Optional[np.ndarray] = None) -> dict:
    """
    Use Llama 3.3 70B via Groq to intelligently configure experiment parameters.

    Args:
        pdb_id: PDB identifier for the target protein
        phi:    7D pocket conditioning vector (optional, for enriched context)

    Returns:
        dict with recommended parameters + reasoning
    """
    # Read API key lazily — ensures dotenv has already loaded by this point
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not groq_api_key:
        logger.warning("GROQ_API_KEY not set, returning defaults")
        return _default_config(pdb_id)

    # Build protein context
    protein_info = PROTEIN_DATABASE.get(pdb_id.upper(), {
        "name": f"Protein {pdb_id.upper()}",
        "disease": "Unknown",
        "type": "Unknown"
    })

    phi_context = ""
    if phi is not None:
        phi_lines = [f"  {name}: {val:.3f}" for name, val in zip(PHI_FEATURE_NAMES, phi)]
        phi_context = f"\nPocket feature vector (normalized 0-1):\n" + "\n".join(phi_lines)

    prompt = f"""You are an expert computational chemist and drug discovery scientist.

Given the following protein target, recommend optimal experiment parameters for a hybrid quantum-classical drug candidate generation pipeline.

Target Protein:
  PDB ID: {pdb_id.upper()}
  Name: {protein_info['name']}
  Disease: {protein_info['disease']}
  Protein Type: {protein_info['type']}
{phi_context}

Pipeline Details:
- ConditionedRNN generates SMILES molecules conditioned on a 7D pocket vector
- Temperature controls sampling diversity (0.5=conservative, 2.0=creative)
- Stress factors perturb the pocket vector to simulate real-world variations:
  * "mutation" - simulates point mutations in binding pocket residues
  * "folding" - simulates protein misfolding/expanded pocket
  * "thermal" - simulates thermal instability (random structural noise)
  * "binding" - simulates binding site deformation
- Docking engines: "autodock_vina" (physics-based), "gnina" (CNN-based), "none"
- VQE optimizers: "COBYLA" (derivative-free, robust), "SPSA" (stochastic, fast), "L-BFGS-B" (gradient-based, precise)
- ADMET prediction: safety/druglikeness screening

You MUST respond with ONLY a valid JSON object (no markdown, no explanation outside the JSON):
{{
  "temperature": <float 0.5-2.0>,
  "n_candidates": <int 10-100>,
  "vqe_optimizer": "<COBYLA|SPSA|L-BFGS-B>",
  "vqe_max_iterations": <int 10-1000>,
  "docking_engine": "<autodock_vina|gnina|none>",
  "stress_factors": [<subset of "mutation","folding","thermal","binding">],
  "run_admet": <true|false>,
  "reasoning": "<2-3 sentence explanation of why these parameters were chosen>"
}}"""

    try:
        from groq import Groq

        client = Groq(api_key=groq_api_key)

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a computational chemistry expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_completion_tokens=500,
            response_format={"type": "json_object"},
        )

        response_text = chat_completion.choices[0].message.content
        config = json.loads(response_text)

        # Validate and clamp values
        config["temperature"] = max(0.5, min(2.0, float(config.get("temperature", 1.0))))
        config["n_candidates"] = max(10, min(100, int(config.get("n_candidates", 20))))
        config["vqe_optimizer"] = config.get("vqe_optimizer", "COBYLA")
        if config["vqe_optimizer"] not in ("COBYLA", "SPSA", "L-BFGS-B"):
            config["vqe_optimizer"] = "COBYLA"
        config["vqe_max_iterations"] = max(10, min(1000, int(config.get("vqe_max_iterations", 100))))
        config["docking_engine"] = config.get("docking_engine", "autodock_vina")
        if config["docking_engine"] not in ("autodock_vina", "gnina", "none"):
            config["docking_engine"] = "autodock_vina"

        valid_stress = {"mutation", "folding", "thermal", "binding"}
        config["stress_factors"] = [s for s in config.get("stress_factors", []) if s in valid_stress]
        config["run_admet"] = bool(config.get("run_admet", True))
        config["reasoning"] = config.get("reasoning", "Parameters set by AI analysis.")
        config["pdb_id"] = pdb_id.upper()
        config["protein_name"] = protein_info["name"]
        config["disease"] = protein_info["disease"]

        logger.info(f"LLM auto-config for {pdb_id}: {json.dumps(config, indent=2)}")
        return config

    except ImportError:
        logger.error("groq package not installed. Install with: pip install groq")
        return _default_config(pdb_id)
    except Exception as e:
        logger.error(f"LLM auto-configure failed: {e}")
        return _default_config(pdb_id)


def _default_config(pdb_id: str) -> dict:
    """Fallback configuration when LLM is unavailable."""
    protein_info = PROTEIN_DATABASE.get(pdb_id.upper(), {
        "name": f"Protein {pdb_id.upper()}",
        "disease": "Unknown",
    })
    return {
        "temperature": 1.0,
        "n_candidates": 20,
        "vqe_optimizer": "COBYLA",
        "vqe_max_iterations": 100,
        "docking_engine": "autodock_vina",
        "stress_factors": [],
        "run_admet": True,
        "reasoning": "Default configuration (LLM unavailable). Balanced parameters suitable for most targets.",
        "pdb_id": pdb_id.upper(),
        "protein_name": protein_info.get("name", pdb_id.upper()),
        "disease": protein_info.get("disease", "Unknown"),
    }
