"""P9 preparation: the fact kernel sees shared facts, and its null (shuffled atoms) does not."""
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from token_veritas.fact_kernel import atoms, fact_embeddings
from token_veritas.dpp import kernel, greedy_select
from token_veritas.tasks import make_items


def test_atoms():
    assert atoms("Vault 317 opens with code amber-42.") == ["317", "amber-42"]
    assert atoms("Unit tests finished in under a minute.") == []


def test_paraphrases_one_fact_different_facts_zero():
    E = fact_embeddings(["Vault 317 opens with code amber-42.", "Record: vault 317 -> code amber-42.",
                         "Vault 512 opens with code onyx-17.", "The linter reported no new findings."])
    S = E @ E.T
    assert abs(S[0, 1] - 1) < 1e-12 and S[0, 2] == 0 and not S[3].any()
    assert np.linalg.eigvalsh(S).min() > -1e-12


def distinct_facts_first(E, item):
    """Greedy DPP with uniform quality and 1 token per chunk; how many distinct facts are among the
    first k picks, where k is the number of facts in the item."""
    n_facts = len({f for f in item["fact_of"] if f >= 0})
    idx, _, _ = greedy_select(kernel(np.ones(len(E)), E), [1] * len(E), n_facts)
    return len({item["fact_of"][i] for i in idx if item["fact_of"][i] >= 0}), n_facts


def test_uniform_quality_picks_one_chunk_per_fact():
    for item in make_items(20, 7, copies=3):
        got, n = distinct_facts_first(fact_embeddings(item["chunks"]), item)
        assert got == n


def test_null_shuffled_atoms_loses_the_property():
    """Anti-vacuity: give each chunk the atoms of a random other chunk. The instrument must be able
    to return a failure; across 20 items at least one must miss a fact."""
    rng, misses = random.Random(3), 0
    for item in make_items(20, 7, copies=3):
        perm = list(range(len(item["chunks"])))
        rng.shuffle(perm)
        E = fact_embeddings([item["chunks"][p] for p in perm])
        got, n = distinct_facts_first(E, item)
        misses += got < n
    assert misses >= 1
