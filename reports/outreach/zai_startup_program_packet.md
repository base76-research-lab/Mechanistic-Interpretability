# Z.ai Startup Program — Base76 / Mechanistic Interpretability (Packet)

Senast uppdaterad: 2026-03-05

Owner: Bjorn Wikstrom (Base76 Research Lab)
Focus area: Mechanistic Interpretability + reliability signals ("Field View")
Last updated: 2026-03-05

This packet is written to share in a 1-on-1 onboarding call and to support the requested materials:
- Corporate basic materials (TBD)
- Official website (TBD)
- Business-related materials: product spec + ALAPI use cases + projected usage plan

## 1) One-paragraph overview (what we are building)

Base76 develops methods to measure and improve reliability and epistemic quality in transformer-based AI systems.
The Mechanistic Interpretability track builds instrumentation and "internal geometry" signals in small models
(starting with GPT-2) to discover interpretable subspaces (e.g., antonym/opposite) and use them for risk triage.

Current milestone: a "Field View" risk signal that correlates model state–candidate misalignment with a
hallucination-like regime in controlled prompts, and separates that from "reasoning" regimes.

## 2) Current state (concrete artifacts)

Work directory:
- `workspace/Base76_Research_Lab/Mechanistic Interpretability/`

Key report:
- `workspace/Base76_Research_Lab/Mechanistic Interpretability/reports/exp_001_sae.md`

Key script (risk signal definition):
- `workspace/Base76_Research_Lab/Mechanistic Interpretability/scripts/field_view.py`

What exists today (public-safe summary):
- SAE-trained feature dictionaries on GPT-2 layer 5 (dict sizes 256/512).
- Identified clusters/subspaces: antonym/opposite, analogy pivot, parentheses/syntax markers, role pairs, etc.
- A risk score computed from logit entropy + subspace gap between residual state and top-k candidate tokens.

## 3) What help we want from Z.ai (why this program matters)

We can do the mechanistic work locally, but Z.ai can help by providing:
- API credits + stable rate limits for automated labeling/eval runs.
- A strong judge model for feature labeling, polysemanticity annotation, and prompt-set expansion.
- A deployment-grade OpenAI-compatible API for repeatable experiment automation (batch runs, retries, logging).

We are NOT asking for hidden-state access from Z.ai models; we use Z.ai primarily as:
- a labeler/judge, and
- a comparative blackbox baseline for hallucination/reasoning behavior on the same prompt suites.

## 4) ALAPI use cases (specific)

Use case A — Feature labeling at scale (SAE feature dictionary)
- Input: top-activating token snippets per SAE unit (from `top_features.json`).
- Output: short semantic label + confidence + polysemanticity notes.
- Constraint: no private data; only synthetic prompts and token snippets.

Use case B — Prompt suite expansion (controlled probes)
- Generate controlled prompt templates for:
  - antonyms / negation / analogy / composition / math / code-like syntax
  - factual queries (including adversarial "fake facts" to test hallucination geometry)
- Output: prompt list + tags (task type, expected behavior, difficulty).

Use case C — Comparative evaluation / "judge" scoring
- Use Z.ai model responses to run a standardized evaluation rubric (truthfulness, calibration cues, refusal quality).
- Compare: local model internal-signal predictions vs Z.ai judge labels (agreement/discord).

Use case D — Writing/communication support (non-sensitive)
- Draft short research summaries, grant paragraphs, and experiment notes from already-public-safe outputs.

## 5) Projected usage plan (rough numbers)

Assumptions:
- Typical prompt length: 50-250 tokens.
- Output: 50-300 tokens per call depending on task (label vs generation).
- All content is synthetic or already-public-safe experiment text.

Phase 1 (2-3 weeks): Build dataset + label the first dictionary
- Feature labeling: 300-800 calls total
- Prompt generation/curation: 100-250 calls total
- Judge rubric scoring: 200-400 calls total
- Expected peak bursts: 5-20 requests/min during automation runs

Phase 2 (1-2 months): Iterate and validate across models/prompts
- Weekly: 500-2,000 calls (depending on sweep size)
- Peak bursts: 10-50 requests/min for short periods (batch mode)

Rate-limit note:
- Peak times are typically during scripted sweeps (evenings/weekends Stockholm time).

## 6) Users (where are they located?)

Current users are primarily:
- The researcher (Bjorn, Sweden/EU)
- A small set of collaborators/test readers (EU/US), when sharing public-safe summaries

## 7) Corporate basic materials / website (placeholders)

These are placeholders to fill in before sharing externally:
- Overseas registration certificate / Domestic business license: TBD
- Overseas app launch screenshots / planned mockups: TBD
- Official website under company domain: TBD

If needed for the call, we can share "planned mockups" as simple diagrams/screenshots of:
- the experiment dashboard (runs, artifacts, risk score plots),
- a feature dictionary UI concept (unit -> label -> examples -> patch effect).

## 8) Questions to ask Z.ai in the onboarding call (data + compliance)

API/data handling (must-have answers):
- Is API customer content excluded from training/improvement by default (opt-in only)?
- Default retention for prompts/completions and logs; any "zero data retention" option?
- Primary processing region(s) and list of sub-processors.
- DPA availability for EU-based users; SCCs for international transfers.
- Support for logprobs/top-k (useful for blackbox calibration tests).

Operational (to run experiments smoothly):
- Confirm RPM/TPM limits and burst behavior for the program tier.
- Recommended retry/backoff patterns; any batch endpoint available?
