# token-veritas

Measuring ways to spend fewer LLM tokens without losing the answer. Every claim here is
pre-registered, run, and scored by frozen code, and the failures are kept.

**Status: Run 2 — REFUTED (kept).** The quantum-derived selector (a determinantal point process)
did **not** beat plain relevance ranking. The cheap baseline it was compared against is the useful
result so far.

*Vincit Omnia Veritas.*

---

## Failures first

Run 2 asked whether a **determinantal point process (DPP)** selector saves tokens better than
top-k relevance. A DPP is the classical probability law of free fermions: det(L_S) is a Slater
determinant, and its Pauli-exclusion repulsion should drop near-duplicate context. It runs on an
ordinary CPU. No quantum hardware is involved and no quantum speedup is claimed.

Scored by `token_veritas/analyze.py`, frozen and pushed before the data existed (budget = 20% of tokens):

```
P1 coverage dpp_cos - topk_cos (R) = -0.037  95% CI [-0.087, 0.000]  threshold >= 0.20 -> FAIL
P2 accuracy dpp_cos - topk_cos (R) = -0.050  95% CI [-0.225, 0.125]  wins 5 losses 7 p=0.7744  threshold >= 0.15 & p<0.05 -> FAIL
P3 [control] coverage dpp_cos - topk_cos (N) = 0.050  threshold |d| <= 0.10 -> PASS
P4 [control] coverage dpp_uniform 0.800 vs random 0.625 (R)  threshold <= random+0.10 -> FAIL
P5 LOO coverage N 0.750 -> R 0.762, drop -0.012  threshold >= 0.20 -> FAIL
P6 accuracy dpp_cos 0.775 vs loo 0.450 (R)  threshold dpp >= loo-0.05 -> PASS
```

**Why P1/P2 failed** (diagnosed from the saved selections after scoring):

- The DPP's repulsion works on *wording*, not on *facts*. In the MiniLM embedding,
  paraphrases of the same fact score cosine 0.864. The two facts the question needs score 0.662
  against each other, and a needed fact against an irrelevant one scores 0.643. So the kernel
  cannot tell "a different needed fact" from "an irrelevant fact".
- The redundancy trap P1 assumed never happened. At 20% budget, top-k kept 2.12 needed-fact chunks
  per item (≈ one copy of each), not four copies of one.
- The DPP spent its diversity on distractors instead: 2.7 per item, versus 1.88 for top-k.

**Design error (mine), which weakens P3 and explains P4.** Condition R (4 copies of each fact) has
**0 filler chunks per item**; condition N (1 copy) has **18.0**. R and N therefore differ in both
redundancy and filler, so N is not the clean "no-redundancy" control that was registered, and P3's
PASS is weaker than it looks. P4 failed for the same reason: in R every chunk is a fact, so
diversity alone finds more distinct facts than random does.

**P5 failed the other way round.** Leave-one-out ablation did not lose *coverage* under redundancy
(0.750 → 0.762). It did lose *accuracy*: 0.450 in R at 20%, versus 0.825 for top-k.

## What held

- **P6:** embedding selection, which needs no LLM passes, beat leave-one-out ablation, which needs 25 per
  question: accuracy 0.775 vs 0.450 (R, 20%). Plain `topk_cos` beat it too (0.825). The ablation
  method from Run 1 is expensive *and* worse here.
- **Exploratory, not registered.** In N (18 filler chunks), keeping 30% of the tokens gave *higher*
  accuracy than the full context. Paired against full context:

```
N dpp_cos@0.3 vs full: 0.675 vs 0.425  diff +0.250 CI [+0.075,+0.425] wins 12 losses 2 p=0.0129
N topk_cos@0.3 vs full: 0.600 vs 0.425  diff +0.175 CI [-0.025,+0.375] wins 12 losses 5 p=0.1435
R dpp_cos@0.3 vs full: 0.925 vs 0.975  diff -0.050 CI [-0.150,+0.050] wins 1 losses 3 p=0.6250
R topk_cos@0.2 vs full: 0.825 vs 0.975  diff -0.150 CI [-0.275,-0.050] wins 0 losses 6 p=0.0312
```

  These are six post-hoc tests. After Bonferroni correction (α = 0.0083), none is significant.
  Treat "pruning a noisy context can help a small model" as a lead for a registered run, not a result.

## Full tables (Run 2, answerer Qwen2.5-0.5B-Instruct, n = 40 per condition)

Condition R — 4 paraphrases per fact, no filler. Full-context accuracy 0.975.

| policy | budget | tokens (mean) | coverage | accuracy |
|---|---|---|---|---|
| random | 0.1 | 30.6 | 0.287 | 0.050 |
| topk_cos | 0.1 | 30.7 | 0.900 | 0.800 |
| dpp_cos | 0.1 | 28.9 | 0.762 | 0.575 |
| loo | 0.1 | 29.6 | 0.500 | 0.150 |
| dpp_loo | 0.1 | 29.8 | 0.487 | 0.150 |
| dpp_uniform | 0.1 | 26.8 | 0.312 | 0.025 |
| random | 0.2 | 67.2 | 0.625 | 0.325 |
| topk_cos | 0.2 | 61.7 | 1.000 | 0.825 |
| dpp_cos | 0.2 | 69.8 | 0.963 | 0.775 |
| loo | 0.2 | 68.0 | 0.762 | 0.450 |
| dpp_loo | 0.2 | 68.9 | 0.812 | 0.450 |
| dpp_uniform | 0.2 | 68.8 | 0.800 | 0.475 |
| random | 0.3 | 103.5 | 0.863 | 0.525 |
| topk_cos | 0.3 | 105.7 | 1.000 | 0.875 |
| dpp_cos | 0.3 | 103.0 | 1.000 | 0.925 |
| loo | 0.3 | 103.2 | 0.875 | 0.650 |
| dpp_loo | 0.3 | 102.0 | 0.963 | 0.700 |
| dpp_uniform | 0.3 | 97.2 | 0.975 | 0.675 |

Condition N — 1 copy per fact, 18 filler chunks. Full-context accuracy 0.425.

| policy | budget | tokens (mean) | coverage | accuracy |
|---|---|---|---|---|
| random | 0.1 | 22.6 | 0.050 | 0.000 |
| topk_cos | 0.1 | 25.0 | 0.325 | 0.000 |
| dpp_cos | 0.1 | 24.5 | 0.362 | 0.000 |
| loo | 0.1 | 23.9 | 0.325 | 0.000 |
| dpp_loo | 0.1 | 23.1 | 0.325 | 0.000 |
| dpp_uniform | 0.1 | 24.0 | 0.000 | 0.000 |
| random | 0.2 | 48.0 | 0.087 | 0.000 |
| topk_cos | 0.2 | 47.0 | 0.725 | 0.475 |
| dpp_cos | 0.2 | 49.5 | 0.775 | 0.425 |
| loo | 0.2 | 48.5 | 0.750 | 0.425 |
| dpp_loo | 0.2 | 47.9 | 0.688 | 0.375 |
| dpp_uniform | 0.2 | 48.5 | 0.000 | 0.000 |
| random | 0.3 | 74.2 | 0.338 | 0.075 |
| topk_cos | 0.3 | 75.2 | 0.925 | 0.600 |
| dpp_cos | 0.3 | 74.0 | 0.938 | 0.675 |
| loo | 0.3 | 75.0 | 0.912 | 0.650 |
| dpp_loo | 0.3 | 74.3 | 0.750 | 0.425 |
| dpp_uniform | 0.3 | 75.7 | 0.000 | 0.000 |

In N, `dpp_uniform` (diversity with no relevance) covers 0.000 at every budget: it spends the whole
budget on the 18 filler lines. It is the null the instrument can return, and it returns it.

## Scope limits

- The task is synthetic (vault-code lookup), and the answerer is a 0.5B model on CPU. Nothing here has
  been measured on real agent traces or against a paid API model.
- "Tokens" means the answerer's input context tokens. Output tokens, caching, and routing were not tested.
- Timing on a 2-thread CPU container: R 4559.8 s, N 3819.8 s for 40 items each. Almost all of that is
  the leave-one-out passes and answer generation. The embedding and DPP selection themselves take milliseconds.

## Open doors (registered, unrun)

- **P7:** the effect on real agent traces (repeated tool outputs, re-sent files) with the target API model.
- **P8:** a log-det stop rule that sets the budget automatically.
- **P9 (new, from the P1 diagnosis):** a *fact-level* kernel. S would be built from shared entities and
  numbers, or from the answerer's own hidden states, not sentence embeddings. It would need a
  redesigned R/N pair with matched filler (fixes the confound above). The DPP idea is untested until
  its kernel measures informational overlap. Run 2 only shows that MiniLM wording similarity does not.

## Run 1 (kept)

`experiments/run1_ablation/` holds the leave-one-out ablation pruner. Two of its predictions failed on
a 0.5B stand-in (needle rank ≤ 2 in 0.6 of items vs ≥ 0.8 registered; retention at 30% budget 0.7 vs
≥ 0.9). The selector's ceiling was the probe's own accuracy. Details: `experiments/run1_ablation/RESULTS.md`.

## Reproduce

```bash
pip install numpy torch transformers sentence-transformers
python3 tests/test_dpp.py                        # 8/8, no model needed
python3 -m token_veritas.run_experiment --condition R --items 40 --seed 2026
python3 -m token_veritas.run_experiment --condition N --items 40 --seed 2027
python3 -m token_veritas.analyze                 # prints P1-P6 verdicts
```

On Termux, `/tmp` is not writable. The scripts write to `results/` in the repo.

## Provenance

- `PREREGISTRATION.md` and `analyze.py` were committed and pushed before any registered-seed data existed.
- The run process started at about 21:13 CDT, 8 minutes before the public push at 21:21. It writes
  results only on completion, so no outcome data existed before the push.
- The commit times all read 21:21 because they were re-authored to the GitHub no-reply address.
- Amendment A2 (answer length 16 → 40 tokens) came from a smoke test on discarded seed 999.

## Credit

Chad Edward Holland — direction, research program, the ablation idea (Run 1).
Claude (Anthropic, `claude-opus-5-5`) — implementation, pre-registration drafting, runs, analysis.
The influence-per-token (η) and token-economy framing came from earlier AI-assisted drafts shared
by Chad; the models that produced them are not recorded.

## License

MIT
