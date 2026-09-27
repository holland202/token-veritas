# Pre-registration — Run 2: fermionic-kernel (DPP) context selection

Committed before any Run 2 data was generated. The git history is the timestamp.
Nothing below may be edited after the run; amendments go in a new dated section.

## Claim
A determinantal point process (DPP) selector picks context chunks under a token budget.
It uses kernel L = diag(q) · S · diag(q), with q = relevance and S = the embedding Gram matrix.
It greedily maximizes log det(I + L_S) gain per token.
Where context is redundant, it retains more of the distinct facts a question needs, and answers
more questions correctly, than top-k relevance or leave-one-out (LOO) ablation at the same token budget.

**What "quantum" means here, stated plainly.** A DPP is the classical probability law of
non-interacting fermions: det(L_S) is a Slater-determinant probability. The Pauli-exclusion
repulsion is what penalizes near-duplicate chunks. This runs on ordinary CPUs. It claims no
quantum hardware and no quantum speedup. The mathematics is established (Macchi 1975;
Kulesza & Taskar 2012). Only the application to LLM token budgets is under test.

## Design (frozen)
- Items: a question asking for the access codes of two vaults. The context holds 2 target facts,
  4 distractor facts about other vaults, and filler.
- Condition R (redundant): every fact appears as 4 paraphrases. Condition N (no redundancy):
  every fact appears once. Filler pads both conditions to 24 chunks, shuffled.
- Embedder: sentence-transformers/all-MiniLM-L6-v2, normalized. S = E Eᵀ.
- Relevance rel_i = cosine(chunk_i, question). q_i = exp(3 · rel_i). The factor β = 3 is fixed and untuned.
- Answerer / LOO probe: Qwen/Qwen2.5-0.5B-Instruct, greedy, CPU.
- Budgets: 10%, 20%, 30% of the full-context token count (Qwen tokenizer).
- Policies: `random`, `topk_cos`, `dpp_cos`, `loo` (leave-one-out log-prob influence per token,
  from Run 1), `dpp_loo` (DPP with q from LOO influence), `dpp_uniform` (q ≡ 1, diversity only).
- Metrics:
  - coverage = fraction of the 2 needed facts with at least one copy kept (deterministic).
  - accuracy = both gold codes appear in the answer.
- n = 40 items per condition. Seeds 2026 (R) and 2027 (N).
- Stats: paired exact sign test (McNemar) on accuracy, α = 0.05. Mean paired difference with a
  95% bootstrap CI (10 000 resamples, seed 0).

## Registered predictions
- **P1** (R, 20%): coverage(dpp_cos) − coverage(topk_cos) ≥ 0.20.
- **P2** (R, 20%): accuracy(dpp_cos) − accuracy(topk_cos) ≥ 0.15, sign-test p < 0.05.
- **P3 anti-vacuity** (N, 20%): |coverage(dpp_cos) − coverage(topk_cos)| ≤ 0.10. With no redundancy,
  the repulsion should have nothing to do. If DPP still "wins" here, the win is not from redundancy.
- **P4 anti-vacuity** (R, 20%): coverage(dpp_uniform) ≤ coverage(random) + 0.10. Diversity without
  relevance should not beat chance by much.
- **P5** (R vs N, 20%): LOO coverage drops by ≥ 0.20 from N to R. When a fact has 4 copies,
  removing one copy barely changes the output, so leave-one-out influence is masked.
- **P6** (R, 20%): accuracy(dpp_cos) ≥ accuracy(loo) − 0.05. The zero-LLM-pass selector matches
  the (k+1)-pass ablation selector.
- **P7** (door, unrun): the effect holds on real agent traces (repeated tool outputs, re-sent file
  contents) and with the target API model as answerer.
- **P8** (door, unrun): a stop rule — halt when the best log det gain per token < τ — sets the
  budget automatically without losing accuracy.

## Kill criterion
If P1 or P2 fails, the repo is published as a negative result, with the failure at the top.
If P3 fails, any P1/P2 win is declared confounded.
