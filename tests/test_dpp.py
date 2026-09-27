"""Deterministic unit tests. Run: python3 tests/test_dpp.py (no model needed)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from token_veritas.dpp import kernel, greedy_select, topk_select, logdet_I_plus
from token_veritas.tasks import make_items, coverage

passed = 0


def check(name, cond):
    global passed
    assert cond, name
    passed += 1
    print("PASS", name)


# 1. Exact duplicates: the DPP picks one copy of each fact; top-k picks both copies of the best.
e = np.array([[1, 0], [1, 0], [0.6, 0.8]], dtype=float)
L = kernel([2.0, 2.0, 1.9], e)
sel, _, _ = greedy_select(L, [1, 1, 1], budget=2)
top, _ = topk_select([2.0, 2.0, 1.9], [1, 1, 1], budget=2)
check("dpp_skips_duplicate", sel == [0, 2] or sel == [1, 2])
check("topk_takes_duplicate (expected failure mode of top-k)", top == [0, 1])

# 2. Anti-vacuity: orthogonal chunks with no redundancy -> DPP must agree with top-k.
e2 = np.eye(4)
q2 = [3.0, 2.0, 1.0, 0.5]
sel2, _, _ = greedy_select(kernel(q2, e2), [1] * 4, budget=2)
top2, _ = topk_select(q2, [1] * 4, budget=2)
check("null_no_redundancy_dpp_equals_topk", sel2 == top2 == [0, 1])

# 3. Budget is a hard constraint.
rng = np.random.default_rng(0)
E = rng.normal(size=(12, 8)); E /= np.linalg.norm(E, axis=1, keepdims=True)
toks = rng.integers(3, 15, size=12).tolist()
sel3, used3, _ = greedy_select(kernel(rng.uniform(0.5, 3, 12), E), toks, budget=30)
check("budget_respected", used3 <= 30 and used3 == sum(toks[i] for i in sel3))

# 4. Objective is monotone: adding a chunk never lowers log det(I + L_A).
L4 = kernel(rng.uniform(0.5, 3, 12), E)
check("monotone", all(logdet_I_plus(L4, list(range(k + 1))) >= logdet_I_plus(L4, list(range(k))) - 1e-12
                      for k in range(11)))

# 5. Task generator: redundancy condition really has 4 copies; control has 1.
R = make_items(3, 1, copies=4)[0]; N = make_items(3, 1, copies=1)[0]
check("R_has_4_copies", R["fact_of"].count(0) == 4 and len(R["chunks"]) == 24)
check("N_has_1_copy", N["fact_of"].count(0) == 1 and len(N["chunks"]) == 24)
check("coverage_metric", coverage(R, [R["fact_of"].index(0)]) == 0.5)

EXPECTED = 8
print(f"{passed}/{EXPECTED} passed")
assert passed == EXPECTED, "test count drifted"
