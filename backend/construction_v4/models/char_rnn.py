"""
CharRNN — Character-level LSTM for SMILES Generation
======================================================
REINVENT-style autoregressive model that generates SMILES strings
one character at a time. Trained on ZINC250k for molecular validity,
then fine-tuned with RL for target-specific drug design.

Architecture:
    Embedding(VOCAB_SIZE, 128) → LSTM(128, 512, 3 layers) → Linear(512, VOCAB_SIZE)

Key methods:
    sample()               — generate SMILES via autoregressive decoding
    sample_with_logprobs() — same + returns log P(smiles) for REINFORCE
    log_prob_batch()       — compute log P under this model (for KL with prior)
"""

import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_v4 import (
    VOCAB_SIZE,
    EMBED_DIM,
    HIDDEN_DIM,
    N_LAYERS,
    DROPOUT,
    MAX_SMILES_LEN,
)
from data.smiles_dataset import SMILESVocab


class CharRNN(nn.Module):
    """
    Character-level LSTM for autoregressive SMILES generation.

    Input:  (batch, seq_len) integer token IDs
    Output: (batch, seq_len, vocab_size) logits over vocabulary
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
        vocab: Optional[SMILESVocab] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.vocab = vocab or SMILESVocab()

        # Layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.vocab.pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x:      (batch, seq_len) integer token IDs
            hidden: optional (h0, c0) each (n_layers, batch, hidden_dim)

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: (h_n, c_n)
        """
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(x))

        # LSTM
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Project to vocab
        logits = self.output_proj(self.dropout(lstm_out))

        return logits, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialise LSTM hidden state with zeros."""
        device = device or next(self.parameters()).device
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    # ------------------------------------------------------------------
    # SAMPLING
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n: int,
        temperature: float = 1.0,
        device: str = "cpu",
        max_len: int = MAX_SMILES_LEN,
    ) -> List[str]:
        """
        Autoregressively sample n SMILES strings.

        Starts with SOS token, samples until EOS or max_len.
        Returns list of decoded SMILES strings (may include invalid ones).
        """
        self.eval()
        device = torch.device(device)
        self.to(device)

        temperature = max(0.1, min(temperature, 3.0))  # safety clamp

        # Start with SOS token
        current = torch.full((n, 1), self.vocab.sos_idx, dtype=torch.long, device=device)
        hidden = self.init_hidden(n, device)

        # Collect generated tokens
        all_tokens = [current]
        finished = torch.zeros(n, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits, hidden = self.forward(current, hidden)
            # logits: (n, 1, vocab_size) -> (n, vocab_size)
            logits = logits[:, -1, :] / temperature

            # Sample from categorical distribution
            probs = F.softmax(logits, dim=-1)
            current = torch.multinomial(probs, num_samples=1)  # (n, 1)
            all_tokens.append(current)

            # Check for EOS
            just_finished = (current.squeeze(-1) == self.vocab.eos_idx)
            finished = finished | just_finished

            if finished.all():
                break

        # Decode
        tokens = torch.cat(all_tokens, dim=1)  # (n, seq_len)
        smiles_list = []
        for i in range(n):
            token_ids = tokens[i].cpu().tolist()
            smi = self.vocab.decode(token_ids, stop_at_eos=True)
            smiles_list.append(smi)

        self.train()
        return smiles_list

    def sample_with_logprobs(
        self,
        n: int,
        temperature: float = 1.0,
        device: str = "cpu",
        max_len: int = MAX_SMILES_LEN,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Sample n SMILES and return log P(smiles) for REINFORCE.

        Returns:
            smiles_list: List[str] of generated SMILES
            log_probs:   (n,) tensor of log probabilities
        """
        self.eval()
        device_obj = torch.device(device)
        self.to(device_obj)

        temperature = max(0.1, min(temperature, 3.0))

        current = torch.full((n, 1), self.vocab.sos_idx, dtype=torch.long, device=device_obj)
        hidden = self.init_hidden(n, device_obj)

        all_tokens = [current]
        log_prob_sum = torch.zeros(n, device=device_obj)
        finished = torch.zeros(n, dtype=torch.bool, device=device_obj)

        for _ in range(max_len):
            logits, hidden = self.forward(current, hidden)
            logits = logits[:, -1, :] / temperature

            log_probs_step = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            current = torch.multinomial(probs, num_samples=1)  # (n, 1)

            # Gather log probs of chosen tokens
            chosen_log_probs = log_probs_step.gather(1, current).squeeze(-1)  # (n,)

            # Only accumulate for non-finished sequences
            log_prob_sum = log_prob_sum + chosen_log_probs * (~finished).float()

            all_tokens.append(current)
            just_finished = (current.squeeze(-1) == self.vocab.eos_idx)
            finished = finished | just_finished

            if finished.all():
                break

        # Decode
        tokens = torch.cat(all_tokens, dim=1)
        smiles_list = []
        for i in range(n):
            token_ids = tokens[i].cpu().tolist()
            smi = self.vocab.decode(token_ids, stop_at_eos=True)
            smiles_list.append(smi)

        self.train()
        return smiles_list, log_prob_sum

    def log_prob_batch(
        self,
        smiles_list: List[str],
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Compute log P(smiles) under this model for a batch of SMILES.

        Used by the frozen prior for KL regularisation during RL.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            log_probs: (n,) tensor of log probabilities
        """
        self.eval()
        device_obj = torch.device(device)
        self.to(device_obj)

        n = len(smiles_list)
        log_probs = torch.zeros(n, device=device_obj)

        for i, smi in enumerate(smiles_list):
            tokens = self.vocab.encode(smi)
            if not tokens:
                log_probs[i] = -100.0  # very unlikely
                continue

            # Build input: [SOS, t1, t2, ..., tN]
            input_ids = [self.vocab.sos_idx] + tokens
            # Build target: [t1, t2, ..., tN, EOS]
            target_ids = tokens + [self.vocab.eos_idx]

            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device_obj)
            target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device_obj)

            with torch.no_grad():
                logits, _ = self.forward(input_tensor)
                # logits: (1, seq_len, vocab_size)
                log_probs_all = F.log_softmax(logits[0], dim=-1)  # (seq_len, vocab_size)

                # Gather log probs of actual target tokens
                seq_log_probs = log_probs_all.gather(
                    1, target_tensor.unsqueeze(-1)
                ).squeeze(-1)
                log_probs[i] = seq_log_probs.sum()

        self.train()
        return log_probs

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save model checkpoint with architecture config."""
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        checkpoint = {
            "state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
        }
        torch.save(checkpoint, path)
        print(f"  [CharRNN] Saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CharRNN":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            n_layers=checkpoint["n_layers"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        print(f"  [CharRNN] Loaded from {path}")
        return model


# ──────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("CharRNN instantiation test:")
    model = CharRNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Vocab size: {model.vocab_size}")
    print(f"  Architecture: Embed({model.vocab_size},{model.embed_dim}) "
          f"→ LSTM({model.embed_dim},{model.hidden_dim},layers={model.n_layers}) "
          f"→ Linear({model.hidden_dim},{model.vocab_size})")

    print("\nSampling 5 SMILES (untrained — expect gibberish):")
    samples = model.sample(5, temperature=1.0, device="cpu")
    for i, s in enumerate(samples):
        print(f"  [{i+1}] '{s}'")

    print("\nSampling with log probabilities:")
    samples, log_probs = model.sample_with_logprobs(3, temperature=1.0, device="cpu")
    for i, (s, lp) in enumerate(zip(samples, log_probs)):
        print(f"  [{i+1}] '{s}' (log P = {lp:.2f})")

    print("\nLog prob batch test:")
    test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    lps = model.log_prob_batch(test_smiles, device="cpu")
    for s, lp in zip(test_smiles, lps):
        print(f"  '{s}' → log P = {lp:.2f}")

    print("\nSave/load test:")
    model.save("/tmp/test_charrnn.pt")
    loaded = CharRNN.load("/tmp/test_charrnn.pt")
    print(f"  Loaded: {loaded.vocab_size} vocab, {loaded.hidden_dim} hidden")

    print("\n✓ CharRNN tests complete")
