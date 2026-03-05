# exp_001_sae — GPT-2 small, layer 5, SAE

## Versions

- v1: 7 prompts, 75 tokens, steps=200, l1=5e-3 -> mse~0.234, mean|z|~12.7 (over-activated), sparsity metrics missing.
- v2: 12 prompts, 128 tokens, LayerNorm on activations, steps=400, l1=1e-3 -> mse~3.4e-4, mean|z|~0.73, ~62% near-zero.

## Setup (v2)

- Model: gpt2-small
- Layer: hidden_states[5]
- Prompts: data/prompts.txt (12 prompts)
- Activations: 128 token positions, dim 768 (LayerNorm before SAE)
- SAE: dict_size=256, steps=400, lr=1e-3, L1=1e-3

## Results (v2)

- Final MSE: ~3.4e-4, L1 term ~7.3e-4 (at step 360/400)
- Sparsity: mean|z|=0.7265, frac|z|<1e-6 ~= 0.618, frac|z|<1e-3 ~= 0.621
- Top features: `experiments/exp_001_sae_v2/top_features.json` (10 units x 5 tokens). Example observations:
  - Many units activate on stopword-ish tokens ("the", "of", "to"); a few capture patterns like "reverse", "repeat", "queen".
  - Needs human labeling (see `reports/feature_dict.md`).

## Files (v2)

- activations (build artifact): experiments/exp_001_sae_v2/activations.pt
- SAE weights (build artifact): experiments/exp_001_sae_v2/sae_weights.pt
- metrics: experiments/exp_001_sae_v2/metrics.json
- top features: experiments/exp_001_sae_v2/top_features.json

## Next steps (v2)

- Label 5-10 units: read `top_features.json`, decode token examples, add to `reports/feature_dict.md`.
- Expand prompts with more varied patterns (numbers, code, QA) to learn a broader dictionary.
- Add reconstruction histograms for the next run.
- Do patching/ablation for a hypothesis feature (e.g., a unit responding to "repeat") and measure logit shift.

## v3 (done) — LayerNorm, dict=512, 800 steps, l1=5e-4

- 17 prompts, layer 5, activations LN: `experiments/exp_001_sae_v3/activations.pt`
- Weight/artifact files: `sae_weights.pt`, `metrics.json`, `top_features.json`, `polysemanticity.json`
- Sparsity: mean|z| ~= 0.56, ~62% near-zero
- Polysemanticity: top 20 units mapped in `polysemanticity.json`

### Key features / clusters (v3)

- Antonym subspace (472/468/57/156/346): captures "opposite"/polarity (cold, dark, light, tall). Important:
  direction lives in a subspace/cluster, not a single neuron.
- Analogy pivot (132/133): "as/like/:" relations.
- Polarity temperature (421 ~ hot, 144 ~ cold) as a control pair.
- Parentheses/structure (212/360/279): paren open/close.
- Greeting/translation cue (396/478/217): "good morning" and translation prompts.
- Role pairs (137/87): president/queen slotting.

### Residual insight ("where is opposite?")

"Opposite" shows up as a direction in the residual stream, not as one unit. To observe it:

1) Base projections: cosine similarity against W_E/W_U and c_proj suggests the antonym direction points toward
   cold/dark/tall and specific c_proj rows in layer 5.
2) Subspace patching: patch the cluster mean / PCA component instead of a single unit; a layer sweep can localize
   where the effect is absorbed.
3) Residual profile: log proj_pc1/pc2, ||residual|| and logit entropy per token. Script: `scripts/opposite_probe.py`.

### Why the "compressor" feels "magic" (hypothesis)

Baseline transformer: tokens -> embeddings -> residual -> features emerge gradually across layers.
"Token compressor" view: tokens -> SAE projection -> feature space -> (rest of the model).

Observed effect:
- We jump directly into a semantic basis; subspace clusters (antonym/analogy) become explicit early.
- Feature-space patching injects directions that otherwise emerge later; a layer sweep suggests early layers show
  the largest effect while mid layers can absorb it.
- The residual becomes more interpretable: directions ~= semantic features rather than a raw token basis.

### Latent future-state: what we know and what we do not know (v3)

- Evidence: `scripts/state_rollout.py` shows we can track coordinates in the antonym subspace over multiple greedy
  steps, and logit entropy / selected tokens vary with that motion. `scripts/patch_subspace.py` changes token
  probabilities differently per layer -> there is a "future-state" structure in the residual.
- What we do not yet see: we do not yet know where the state "forms" (which heads/MLPs create it) or whether it
  transfers robustly across models and random seeds.

To reproduce "opposite" probes locally:

```bash
python3 scripts/opposite_probe.py --prompt "the opposite of hot is" --units 472 468 57 156 346
```

Output is saved to `experiments/exp_001_sae_v3/opposite_probe.json` and prints top tokens for PC1/PC2 plus a
per-token profile.

### Next steps (v3+)

- Layer sweep + head masking for subspace patching (localize the circuit).
- Add logit-shift metrics for cluster patching (similar to `scripts/patch_demo.py`, but on PCA vectors).
- Continue filling `reports/feature_dict.md` with v3 labels.

## Research alignment — FNC / Field / Future-state

- The FNC framing (Field -> Node -> Cockpit -> Collapse -> State) matches the observation that the latent
  future-state appears as directions in the residual (Field), while "collapse" happens at unembedding.
- Operator vs operand: SAE subspaces (antonym/analogy) behave like operator axes; token identity (operands)
  projects weakly onto these axes and is selected via other dimensions. This is consistent with the "field dynamics"
  narrative in Applied-Ai-Philosophy.
- Field View shows the gap: state can be far along an operator PC while candidate tokens cluster near 0. This
  suggests the Field contains multiple interacting bases (operator + entity + syntax).

### TODOs linked to the FNC hypothesis

- Alignment test: correlate (W_U · v_PC) · state_PC against logits (operator vs operand).
- Multi-subspace projection: project residual + candidates onto {antonym, analogy, entity/syntax} and search for
  overlaps/conflicts -> epistemic uncertainty.
- "Residual + alpha * W_U[token]": simulate local collapse and observe how Field coords move.

## Quick test: Field View risk signal (v3, antonym subspace pc2)

Prompts run with `scripts/field_view.py --topk 10` (units 472/468/57/156/346, layer 5):

![Field View triage (GPT-2, antonym subspace)](figures/field_view_triage.png)

| Prompt | H (logit entropy) | Risk | Field coords | |coords| (operator strength) | Candidate gap | Observation |
|---|---|---|---|---|---|---|
| `2 + 2 =` | 5.12 | 0.55 | (-0.28, -0.47) | 0.55 | small | State ~= candidate cloud -> low actual uncertainty (almost collapsed) |
| `king is to queen as man is to` | 6.47 | 1.00 | (-3.69, -0.70) | 3.76 | large | Strong operator, under-specified operands -> reasoning-like |
| `who was the president of france in 1200?` | 6.15 | 1.00 | (-5.92, -3.11) | 6.73 | very large | State far from candidate cloud -> classic hallucination geometry (misalignment) |

Files:

- `experiments/exp_001_sae_v3/field_view_math.json`
- `experiments/exp_001_sae_v3/field_view_analogy_run2.json`
- `experiments/exp_001_sae_v3/field_view_hallucination.json`

Insight: the risk signal measures "state-to-token misalignment". Operator strength (|coords|) supports triage:

- high H + strong operator -> reasoning-like
- high H + weak operator -> hallucination-like
- medium H + small gap -> locked-in fact

Formal risk definition (as implemented in `scripts/field_view.py`):

```text
entropy_norm = H / 10.0
gap_norm     = gap_state_to_candidates / (1 + candidate_spread_mean)
risk_score   = min(1.0, 0.5 * entropy_norm + 0.5 * gap_norm)
```

Where:

- `H` is logit entropy (natural log) for the next-token distribution.
- `gap_state_to_candidates` is a Euclidean distance in the chosen subspace:
  - state_coords = projection of the residual onto the SAE subspace (often PCA/cluster-mean).
  - candidate_coords = projection of top-k token vectors (W_U[topk]) onto the same subspace.
  - gap = ||state_coords - mean(candidate_coords)||_2. (We also log min-dist, but risk uses mean-dist for stability.)
- `candidate_spread_mean` is the mean ||candidate_coords||.
- `|coords|` (operator strength) is ||field_coords|| and is reported separately (not yet in the risk formula).

### Threats to validity (v3)

- **SAE basis dependence**: features are not unique; different inits can produce different bases.
  Mitigation: use cluster PCA (subspace) rather than single units and report seeds.
- **Model size**: GPT-2 small may exaggerate effects or miss phenomena.
  Mitigation: run a parallel Phi-2 experiment and log model version.
- **Prompt-set size**: 17 prompts is small.
  Mitigation: expand the prompt list and report each run with file paths + checksums.
