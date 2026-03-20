"""
SMILES Dataset — Character-level Tokenisation for RNN
======================================================
Fixed vocabulary covering all ZINC250k SMILES. Provides:
  - SMILESVocab: encode/decode between SMILES strings and integer tokens
  - SMILESDataset: PyTorch Dataset for next-character prediction training
  - collate_fn: efficient variable-length batching with pack_padded_sequence

Usage:
    from data.smiles_dataset import SMILESVocab, SMILESDataset
    vocab = SMILESVocab()
    dataset = SMILESDataset("data/zinc250k_clean.csv", vocab)
"""

import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_v4 import SMILES_CHARS, MULTI_CHAR_TOKENS, SPECIAL_TOKENS, MAX_SMILES_LEN


class SMILESVocab:
    """
    Fixed character-level SMILES vocabulary.

    Handles multi-character tokens (Cl, Br) as single units.
    Never inferred from data — ensures model portability.
    """

    def __init__(self):
        # Build vocabulary: special tokens first, then multi-char, then single chars
        self.tokens = SPECIAL_TOKENS + MULTI_CHAR_TOKENS + SMILES_CHARS
        self.char_to_idx = {ch: i for i, ch in enumerate(self.tokens)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.tokens)}

        self.pad_idx = self.char_to_idx["<PAD>"]
        self.sos_idx = self.char_to_idx["<SOS>"]
        self.eos_idx = self.char_to_idx["<EOS>"]

        # Regex pattern for tokenisation: multi-char tokens first, then single chars
        multi_escaped = [re.escape(t) for t in MULTI_CHAR_TOKENS]
        self._token_pattern = re.compile(
            "|".join(multi_escaped) + "|."
        )

    @property
    def size(self) -> int:
        return len(self.tokens)

    def tokenise(self, smiles: str) -> List[str]:
        """Split SMILES string into tokens (handles Cl, Br as single tokens)."""
        return self._token_pattern.findall(smiles)

    def encode(self, smiles: str) -> List[int]:
        """
        Encode SMILES string to integer token sequence.
        Returns list of token indices (without SOS/EOS — those are added by Dataset).
        Unknown characters map to PAD (should not happen with ZINC250k).
        """
        tokens = self.tokenise(smiles)
        return [self.char_to_idx.get(t, self.pad_idx) for t in tokens]

    def decode(self, token_ids: List[int], stop_at_eos: bool = True) -> str:
        """
        Decode integer token sequence back to SMILES string.
        Stops at first EOS token. Skips SOS and PAD tokens.
        """
        chars = []
        for idx in token_ids:
            if idx == self.eos_idx and stop_at_eos:
                break
            if idx in (self.sos_idx, self.pad_idx):
                continue
            token = self.idx_to_char.get(idx, "")
            chars.append(token)
        return "".join(chars)


class SMILESDataset(Dataset):
    """
    PyTorch Dataset for character-level SMILES next-token prediction.

    Each item returns:
        input_ids:  [SOS, t1, t2, ..., tN]     (teacher-forced input)
        target_ids: [t1, t2, ..., tN, EOS]     (shifted by 1 — next-char prediction)
        length:     actual length including SOS but excluding padding
    """

    def __init__(
        self,
        csv_path: str,
        vocab: Optional[SMILESVocab] = None,
        max_len: int = MAX_SMILES_LEN,
    ):
        self.vocab = vocab or SMILESVocab()
        self.max_len = max_len

        # Load SMILES
        df = pd.read_csv(csv_path)
        self.smiles_list = df["smiles"].values.tolist()

        # Pre-encode all SMILES for speed
        self.encoded = []
        skipped = 0
        for smi in self.smiles_list:
            tokens = self.vocab.encode(smi)
            if len(tokens) > max_len:
                skipped += 1
                continue
            self.encoded.append(tokens)

        if skipped > 0:
            print(
                f"  [SMILESDataset] Skipped {skipped} SMILES exceeding max_len={max_len}"
            )
        print(
            f"  [SMILESDataset] Loaded {len(self.encoded)} SMILES "
            f"(vocab_size={self.vocab.size})"
        )

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        tokens = self.encoded[idx]

        # Input: [SOS, t1, t2, ..., tN]
        input_ids = [self.vocab.sos_idx] + tokens
        # Target: [t1, t2, ..., tN, EOS]
        target_ids = tokens + [self.vocab.eos_idx]

        length = len(input_ids)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            length,
        )


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader. Pads sequences and returns lengths.

    Returns:
        input_ids:  (batch, max_seq_len) padded
        target_ids: (batch, max_seq_len) padded
        lengths:    (batch,) actual lengths for pack_padded_sequence
    """
    # Get pad index from first item (all use same vocab)
    pad_idx = 0  # <PAD> is index 2 but we pad with 0 for safety

    # Sort by length descending (required for pack_padded_sequence)
    batch.sort(key=lambda x: x[2], reverse=True)

    input_ids, target_ids, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)

    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=2)  # PAD=2
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=2)

    return input_ids, target_ids, lengths


def decode_smiles(token_ids: List[int], vocab: SMILESVocab) -> str:
    """Convenience function: decode token IDs to SMILES string."""
    return vocab.decode(token_ids)


# ──────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vocab = SMILESVocab()
    print(f"Vocabulary size: {vocab.size}")
    print(f"SOS={vocab.sos_idx}, EOS={vocab.eos_idx}, PAD={vocab.pad_idx}")

    # Round-trip test
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",      # aspirin
        "c1ccc2[nH]c(-c3ccccn3)nc2c1", # contains Br-free heterocycle
        "ClC1=CC=CC=C1",                # chlorobenzene (tests Cl token)
        "BrC1=CC=CC=C1",                # bromobenzene (tests Br token)
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    ]

    print(f"\nRound-trip tests:")
    all_pass = True
    for smi in test_smiles:
        tokens = vocab.encode(smi)
        decoded = vocab.decode(tokens)
        status = "✓" if decoded == smi else "✗"
        if decoded != smi:
            all_pass = False
        print(f"  {status} '{smi}' -> {len(tokens)} tokens -> '{decoded}'")

    if all_pass:
        print("\n  All round-trip tests PASSED")
    else:
        print("\n  Some round-trip tests FAILED")
