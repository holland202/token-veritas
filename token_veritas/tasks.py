"""Two-fact retrieval items with controllable redundancy (frozen in PREREGISTRATION.md)."""
import random

WORDS = ["amber", "cobalt", "granite", "willow", "ember", "slate",
         "harbor", "cedar", "quartz", "falcon", "maple", "onyx"]
PARAPHRASES = [
    "The access code for vault {v} is {c}.",
    "Vault {v} opens with code {c}.",
    "Record: vault {v} -> code {c}.",
    "Note that {c} is the code assigned to vault {v}.",
]
FILLER = [
    "Build step completed without warnings on aarch64.",
    "The governance suite was re-run after the last commit.",
    "Thermal state remained nominal during the batch.",
    "Log rotation archived the previous session output.",
    "The operator confirmed the working directory is under HOME.",
    "A cache directory was cleared before the next run.",
    "Dependencies resolved from the standard package registry.",
    "The report file was regenerated from current results.",
    "No network access was required for this stage.",
    "The index script scanned the notes directory again.",
    "Timing overhead for the loop was within the expected range.",
    "The previous checkout was left untouched as a reference.",
    "Disk usage stayed below the session allowance.",
    "The linter reported no new findings.",
    "A backup of the configuration was written to storage.",
    "The scheduler queue was empty at the time of the check.",
    "Two warnings about deprecated flags were suppressed.",
    "The build cache was reused from the previous run.",
    "Unit tests finished in under a minute.",
    "The environment variables were reloaded from the profile.",
]


def make_items(n, seed, copies, n_chunks=24, n_distractors=4):
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        vaults = rng.sample(range(100, 999), 2 + n_distractors)
        codes = []
        while len(codes) < len(vaults):
            c = f"{rng.choice(WORDS)}-{rng.randint(10, 99)}"
            if c not in codes:
                codes.append(c)
        chunks, fact_of = [], []
        for f, (v, c) in enumerate(zip(vaults, codes)):
            for t in rng.sample(PARAPHRASES, copies):
                chunks.append(t.format(v=v, c=c))
                fact_of.append(f)
        pad = n_chunks - len(chunks)
        assert pad >= 0, "too many fact chunks for n_chunks"
        for s in rng.sample(FILLER, pad):
            chunks.append(s)
            fact_of.append(-1)
        order = list(range(len(chunks)))
        rng.shuffle(order)
        items.append(dict(
            chunks=[chunks[i] for i in order],
            fact_of=[fact_of[i] for i in order],   # 0 and 1 are the needed facts
            question=(f"What are the access codes for vault {vaults[0]} and vault {vaults[1]}? "
                      "Answer with both codes."),
            gold=codes[:2],
        ))
    return items


def coverage(item, kept):
    have = {item["fact_of"][i] for i in kept}
    return ((0 in have) + (1 in have)) / 2.0
