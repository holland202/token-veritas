"""P9 preparation (NOT RUN against a model): a fact-level similarity kernel.

Run 2 showed MiniLM sentence embeddings measure wording, not information: two paraphrases of one
fact can look less alike than two unrelated lines with the same template. Here each chunk becomes the
set of its fact atoms (codes such as 'amber-42' and numbers such as '317'), and S is the cosine
similarity of those sets. Paraphrases of one fact then have S = 1, different facts S = 0, and a line
with no atoms is a zero row (it adds nothing to log det(I + L_A)).

S = E E^T with non-negative rows, so it is positive semi-definite, as dpp.kernel requires.
The atom pattern is fixed here and is part of what P9 will test; it is not tuned on results.
"""
import re
import numpy as np

ATOM = re.compile(r"\b[a-z]+-\d+\b|\b\d+\b")


def atoms(text):
    return sorted(set(ATOM.findall(text.lower())))


def fact_embeddings(chunks):
    """Rows: L2-normalised indicator vectors over all atoms in the chunks (zero row if none)."""
    sets = [atoms(c) for c in chunks]
    vocab = {a: i for i, a in enumerate(sorted({a for s in sets for a in s}))}
    E = np.zeros((len(chunks), max(len(vocab), 1)))
    for r, s in enumerate(sets):
        for a in s:
            E[r, vocab[a]] = 1.0
        n = np.linalg.norm(E[r])
        if n:
            E[r] /= n
    return E
