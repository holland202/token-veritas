# Run 3 (P9): fact-level kernel. Plan, not yet registered, NOT RUN against a model

**Built so far (2026-09-27):** `token_veritas/fact_kernel.py` and `tests/test_fact_kernel.py` (4 tests,
no model needed).
- Chunks become sets of fact atoms (codes like `amber-42`, numbers). S is the cosine similarity of
  those sets.
- On 20 generated items with 3 paraphrases per fact, greedy DPP with *uniform* quality picks one chunk
  per fact before any repeat, in every item.
- The null (each chunk given a random other chunk's atoms) misses a fact in at least one item. The
  test requires that, so the instrument can fail.

**What this does not show.** Whether the selected context helps the answerer. The generator's facts
are exactly what the atom pattern matches, so this is a check that the kernel is wired right, not
evidence about real text.

## Before registering, the R/N pair must be redesigned

- **Matched filler (the Run 2 confound).** Filler must carry numbers and code-like tokens too, for
  example "Unit tests finished in 48 s on build 2217". Otherwise the atom kernel wins by treating
  "has a number" as relevance.
- **An adversarial filler class:** lines that repeat a vault number with no code ("Vault 317 was
  audited"). The kernel will see these as partly overlapping the true fact. That is the case where it
  should be allowed to fail.

## Predictions to register (thresholds from a pilot on seeds disjoint from the run)

- **P9a** At 20 % budget in R, `dpp_fact` coverage ≥ `topk_cos` coverage + 0.1.
- **P9b** `dpp_fact_uniform` (no relevance) covers < 0.2 on the needed pair: diversity alone still
  cannot know which facts are asked for (the Run 2 null, kept).
- **P9c** On adversarial filler, `dpp_fact` loses at least some of that margin. If it does not, the
  adversarial filler was not adversarial, and the pilot must say so.

Phone command (after registration): `python3 -m token_veritas.run_experiment --condition R3 ...`,
to be written together with the registration.
