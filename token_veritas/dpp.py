"""Token-budgeted DPP selection (fermionic kernel), NumPy only.

L = diag(q) S diag(q). det(L_A) is the probability that a determinantal point process
(the classical law of free fermions) contains the set A. Near-duplicate rows of S
make det(L_A) collapse toward zero, so the selector avoids redundancy by construction.

Objective (monotone submodular): f(A) = log det(I + L_A).
Greedy picks the chunk with the largest marginal gain per token that fits the budget.
"""
import numpy as np


def kernel(quality, emb):
    q = np.asarray(quality, dtype=np.float64)
    E = np.asarray(emb, dtype=np.float64)
    S = E @ E.T
    return q[:, None] * S * q[None, :]


def logdet_I_plus(L, idx):
    if not idx:
        return 0.0
    sub = L[np.ix_(idx, idx)]
    sign, val = np.linalg.slogdet(np.eye(len(idx)) + sub)
    assert sign > 0, "I + L_A must be positive definite"
    return float(val)


def greedy_select(L, tokens, budget, per_token=True):
    """Return (sorted indices, tokens used, gain trace)."""
    n = L.shape[0]
    tokens = [max(1, int(t)) for t in tokens]
    chosen, used, cur, trace = [], 0, 0.0, []
    while True:
        best, best_score, best_val = None, -np.inf, None
        for i in range(n):
            if i in chosen or used + tokens[i] > budget:
                continue
            val = logdet_I_plus(L, chosen + [i])
            gain = val - cur
            score = gain / tokens[i] if per_token else gain
            if score > best_score:
                best, best_score, best_val = i, score, val
        if best is None:
            break
        chosen.append(best)
        used += tokens[best]
        trace.append((best, best_val - cur, best_score))
        cur = best_val
    return sorted(chosen), used, trace


def topk_select(scores, tokens, budget, per_token=False):
    order = sorted(range(len(scores)),
                   key=lambda i: -(scores[i] / max(1, tokens[i]) if per_token else scores[i]))
    chosen, used = [], 0
    for i in order:
        if used + tokens[i] <= budget:
            chosen.append(i)
            used += tokens[i]
    return sorted(chosen), used
