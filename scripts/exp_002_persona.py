#!/usr/bin/env python3
"""
exp_002_persona.py — skeleton
- Loads activations (v3) + token ids
- Placeholder for trait probes (sycophancy/hallucination/other) — user to fill trait labels
"""
from pathlib import Path
import torch
import json

ROOT = Path(__file__).resolve().parent.parent
ACTS_PATH = ROOT/'experiments/exp_001_sae_v3/activations.pt'


def main():
    blob = torch.load(ACTS_PATH)
    acts = blob['acts']
    tok_ids = blob['token_ids']
    print('Loaded activations', acts.shape)
    print('Token ids sample', tok_ids[:10])
    print('TODO: implement trait labels + linear probes')

if __name__ == "__main__":
    main()
