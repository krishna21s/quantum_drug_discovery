"""
Conditioned RNN — Pocket-conditioned SMILES Generator
=======================================================
Extends CharRNN with EGFR pocket conditioning. The 7D pocket
vector φ is projected to HIDDEN_DIM and added to the initial
LSTM hidden state before generation begins.

Conditioning mechanism:
    φ (7,) → Linear(7, 512) → tanh → added to h₀ for all layers

Two-stage training:
    1. Pre-train CharRNN (unconditioned) on ZINC250k
    2. Fine-tune ConditionedRNN (load pre-trained weights + add φ projection)
"""

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_v4 import PHI_DIM, HIDDEN_DIM, N_LAYERS, MAX_SMILES_LEN
from models.char_rnn import CharRNN
from data.smiles_dataset import SMILESVocab


class ConditionedRNN(CharRNN):
    """
    CharRNN extended with pocket conditioning.

    Inherits all pre-trained weights from CharRNN. Adds a small
    φ → hidden projection layer that biases the initial hidden state
    toward generating molecules complementary to the binding pocket.
    """

    def __init__(
        self,
        phi_dim: int = PHI_DIM,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.phi_dim = phi_dim

        # Pocket conditioning projection: φ → LSTM initial hidden state bias
        self.phi_projection = nn.Linear(phi_dim, self.hidden_dim)

    def _condition_hidden(
        self,
        phi: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create conditioned LSTM hidden state from pocket vector φ.

        Args:
            phi:        (phi_dim,) or (1, phi_dim) pocket vector
            batch_size: number of sequences in batch
            device:     target device

        Returns:
            (h0, c0) where h0 has φ bias added to all layers
        """
        # Ensure phi is on correct device and shape
        if isinstance(phi, np.ndarray):
            phi = torch.tensor(phi, dtype=torch.float32)
        phi = phi.to(device)
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)  # (1, phi_dim)

        # Project: (1, phi_dim) → (1, hidden_dim) → tanh for stability
        phi_hidden = torch.tanh(self.phi_projection(phi))  # (1, hidden_dim)

        # Expand to all layers and all batch items
        # h0: (n_layers, batch_size, hidden_dim)
        h0 = phi_hidden.unsqueeze(0).expand(self.n_layers, batch_size, -1).contiguous()
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)

        return (h0, c0)

    def forward(
        self,
        x: torch.Tensor,
        phi: torch.Tensor = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional pocket conditioning.

        Args:
            x:      (batch, seq_len) token IDs
            phi:    (phi_dim,) pocket vector (optional)
            hidden: optional (h0, c0)

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: (h_n, c_n)
        """
        if hidden is None:
            if phi is not None:
                hidden = self._condition_hidden(phi, x.size(0), x.device)
            else:
                hidden = self.init_hidden(x.size(0), x.device)

        embedded = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(embedded, hidden)
        logits = self.output_proj(self.dropout(lstm_out))

        return logits, hidden

    # ------------------------------------------------------------------
    # CONDITIONED SAMPLING
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_conditioned(
        self,
        phi: torch.Tensor,
        n: int,
        temperature: float = 1.0,
        device: str = "cpu",
        max_len: int = MAX_SMILES_LEN,
    ) -> List[str]:
        """
        Sample n SMILES conditioned on pocket vector φ.

        Args:
            phi:         (phi_dim,) pocket vector
            n:           number of molecules to generate
            temperature: sampling temperature
            device:      compute device

        Returns:
            List[str]: generated SMILES strings
        """
        self.eval()
        device_obj = torch.device(device)
        self.to(device_obj)

        temperature = max(0.1, min(temperature, 3.0))

        # Conditioned initial hidden state
        hidden = self._condition_hidden(phi, n, device_obj)

        current = torch.full((n, 1), self.vocab.sos_idx, dtype=torch.long, device=device_obj)
        all_tokens = [current]
        finished = torch.zeros(n, dtype=torch.bool, device=device_obj)

        for _ in range(max_len):
            embedded = self.dropout(self.embedding(current))
            lstm_out, hidden = self.lstm(embedded, hidden)
            logits = self.output_proj(lstm_out[:, -1, :]) / temperature

            probs = F.softmax(logits, dim=-1)
            current = torch.multinomial(probs, num_samples=1)
            all_tokens.append(current)

            just_finished = (current.squeeze(-1) == self.vocab.eos_idx)
            finished = finished | just_finished
            if finished.all():
                break

        tokens = torch.cat(all_tokens, dim=1)
        smiles_list = []
        for i in range(n):
            smi = self.vocab.decode(tokens[i].cpu().tolist(), stop_at_eos=True)
            smiles_list.append(smi)

        self.train()
        return smiles_list

    def sample_conditioned_with_logprobs(
        self,
        phi: torch.Tensor,
        n: int,
        temperature: float = 1.0,
        device: str = "cpu",
        max_len: int = MAX_SMILES_LEN,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Sample n SMILES conditioned on φ and return log probabilities.

        Args:
            phi:         (phi_dim,) pocket vector
            n:           number of molecules
            temperature: sampling temperature

        Returns:
            smiles_list: List[str]
            log_probs:   (n,) tensor of log P(smiles | φ)
        """
        self.eval()
        device_obj = torch.device(device)
        self.to(device_obj)

        temperature = max(0.1, min(temperature, 3.0))

        hidden = self._condition_hidden(phi, n, device_obj)

        current = torch.full((n, 1), self.vocab.sos_idx, dtype=torch.long, device=device_obj)
        all_tokens = [current]
        log_prob_sum = torch.zeros(n, device=device_obj)
        finished = torch.zeros(n, dtype=torch.bool, device=device_obj)

        for _ in range(max_len):
            embedded = self.dropout(self.embedding(current))
            lstm_out, hidden = self.lstm(embedded, hidden)
            logits = self.output_proj(lstm_out[:, -1, :]) / temperature

            log_probs_step = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            current = torch.multinomial(probs, num_samples=1)

            chosen_log_probs = log_probs_step.gather(1, current).squeeze(-1)
            log_prob_sum = log_prob_sum + chosen_log_probs * (~finished).float()

            all_tokens.append(current)
            just_finished = (current.squeeze(-1) == self.vocab.eos_idx)
            finished = finished | just_finished
            if finished.all():
                break

        tokens = torch.cat(all_tokens, dim=1)
        smiles_list = []
        for i in range(n):
            smi = self.vocab.decode(tokens[i].cpu().tolist(), stop_at_eos=True)
            smiles_list.append(smi)

        self.train()
        return smiles_list, log_prob_sum

    # ------------------------------------------------------------------
    # LOAD PRE-TRAINED WEIGHTS
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, charnn_path: str, phi_dim: int = PHI_DIM, device: str = "cpu") -> "ConditionedRNN":
        """
        Create ConditionedRNN with pre-trained CharRNN weights.

        Loads all embedding/LSTM/output weights from CharRNN checkpoint.
        Initialises φ projection layer fresh (Xavier init).

        Args:
            charnn_path: path to CharRNN checkpoint (rnn_pretrained.pt)
            phi_dim:     pocket vector dimension

        Returns:
            ConditionedRNN with pre-trained weights
        """
        # Load CharRNN checkpoint
        checkpoint = torch.load(charnn_path, map_location=device, weights_only=False)

        model = cls(
            phi_dim=phi_dim,
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            n_layers=checkpoint["n_layers"],
        )

        # Load pre-trained weights (ignoring phi_projection which doesn't exist in CharRNN)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = model.state_dict()

        # Only load matching keys
        matched = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)

        # Xavier init for the fresh phi_projection layer
        nn.init.xavier_uniform_(model.phi_projection.weight)
        nn.init.zeros_(model.phi_projection.bias)

        model.to(device)
        print(f"  [ConditionedRNN] Loaded {len(matched)}/{len(pretrained_dict)} pre-trained weights")
        print(f"  [ConditionedRNN] φ projection: ({phi_dim} → {checkpoint['hidden_dim']}) initialised fresh")

        return model

    def save(self, path: str):
        """Save conditioned model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        checkpoint = {
            "state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "phi_dim": self.phi_dim,
        }
        torch.save(checkpoint, path)
        print(f"  [ConditionedRNN] Saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ConditionedRNN":
        """Load conditioned model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            phi_dim=checkpoint.get("phi_dim", PHI_DIM),
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            n_layers=checkpoint["n_layers"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        print(f"  [ConditionedRNN] Loaded from {path}")
        return model


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ConditionedRNN test:")

    model = ConditionedRNN()
    n_params = sum(p.numel() for p in model.parameters())
    phi_params = sum(p.numel() for p in model.phi_projection.parameters())
    print(f"  Total parameters:  {n_params:,}")
    print(f"  φ projection:      {phi_params:,} ({model.phi_dim} → {model.hidden_dim})")

    # Test conditioned sampling
    phi = np.random.rand(PHI_DIM).astype(np.float32)
    samples = model.sample_conditioned(phi, n=3, temperature=1.2)
    print(f"\n  Conditioned samples (random φ, untrained):")
    for i, s in enumerate(samples):
        print(f"    [{i+1}] '{s}'")

    # Test with logprobs
    samples, lps = model.sample_conditioned_with_logprobs(phi, n=3, temperature=1.0)
    print(f"\n  With log probs:")
    for s, lp in zip(samples, lps):
        print(f"    '{s}' (log P = {lp:.2f})")

    print(f"\n  ✓ ConditionedRNN tests passed")
