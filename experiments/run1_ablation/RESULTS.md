# Token Veritas — run 1 (container, 2026-09-26)

Status: **Draft, verified reference code.** Probe: Qwen2.5-0.5B-Instruct (CPU, container) —
a stand-in, NOT the registered on-device Qwen 1.5B. n = 20 items + 2 null items, 16 chunks each, seed 11.

## Failures first
- **P2 (self mode, needle rank ≤ 2 in ≥ 80%) fails on the 0.5B stand-in:** `C1_rank_le2_rate 0.6`.
- **P3 (at 30% budget, eta retains needle in ≥ 90%) fails:** `eta@0.3 retention 0.7`.
- Cause: in self mode the probe scores influence against *its own* answer. Full-context accuracy
  is `0.5`, and needle rank-1 rate is `0.55`. When the probe is wrong, the influence points at the
  chunk supporting the wrong answer. **The selector's ceiling is the probe's own accuracy.**
- C2 null control is not zero: `C2_null_max_abs_infl_mean 1.4748` vs `needle_infl_mean 7.9182`.
  That is the noise floor, roughly 1/5 of the signal.

## What held
- C3 passes: shuffling the influence scores destroys selection
  (`eta_shuffled@0.3` retention `0.1` vs `eta@0.3` `0.7`). Selection is driven by influence, not by token length.
- eta beats random, recency, and eta_shuffled at every budget, on retention, accuracy, and gold log-prob.

| policy @ budget | needle retained | answer correct | Δ gold log-prob |
|---|---|---|---|
| full context (100%) | — | 0.5 | 0 |
| eta@0.1 | 0.55 | 0.3 | -8.4817 |
| eta@0.2 | 0.6 | 0.55 | -7.2755 |
| eta@0.3 | 0.7 | 0.5 | -5.4876 |
| eta@0.5 | 0.85 | 0.7 | -2.7962 |
| random@0.3 | 0.3 | 0.25 | -14.1442 |
| random@0.5 | 0.65 | 0.6 | -8.5096 |
| recency@0.5 | 0.3 | 0.25 | -12.7974 |

eta@0.2 and eta@0.5 score above full context (0.55 and 0.7 vs 0.5). Pruning distractors can help a
weak model. At n = 20 this is **not significant**: one item is 0.05. Treat it as a lead.

## The cost problem (unregistered, but it decides everything)
Leave-one-out scoring costs k+1 probe passes per query (17 here). The probe is local, so those
tokens are free on the API bill. But latency is real: `seconds 1078.63` for 22 items on 2 CPU threads.
Everything here runs on one query at a time. A multi-turn agent does not know the next question in
advance, so per-question influence does not transfer directly to history pruning.

## Open doors (unrun)
- P1 oracle mode (needle rank 1 in ≥ 90%). This separates "probe can't find it" from "probe answered wrong".
- P2/P3 on the registered Qwen 1.5B GGUF, on device.
- P4 probe → target transfer (Spearman ρ ≥ 0.5 against the API model's influences).
- Real agent traces instead of the synthetic vault task.

Reproduce: `python3 token_veritas.py --backend hf --mode self --items 20 --seed 11`
