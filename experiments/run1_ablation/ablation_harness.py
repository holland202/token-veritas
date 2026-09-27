#!/usr/bin/env python3
"""
token_veritas.py -- ablation-influence context pruning, measured.

Status: Draft. Plumbing verified with the lexical backend ONLY.
Model-probe results: NOT RUN.

Question
  A small local probe scores each context chunk by how much removing it
  lowers the probe's log-likelihood of the answer. If we keep only chunks
  with high influence per token (eta = influence / tokens), is the answer
  preserved at a fraction of the tokens?

Modes
  oracle : influence w.r.t. the GOLD answer. Upper bound. Not deployable,
           because in production you do not know the answer.
  self   : influence w.r.t. the probe's own full-context greedy answer.
           Deployable. This is the mode that actually matters.

Anti-vacuity controls
  C1 needle rank  : the planted fact chunk should rank #1 by influence
  C2 null item    : question about an absent fact -> no chunk should matter
  C3 eta_shuffled : eta policy with permuted scores must fall to ~random.
                    If it does not, selection is not driven by influence.

Registered predictions for the Qwen 1.5B device run (NOT RUN):
  P1 oracle: needle rank 1 in >= 90% of items
  P2 self:   needle rank <= 2 in >= 80% of items
  P3 at 30% token budget, eta retains the needle in >= 90% of items;
     random and eta_shuffled retain it in ~30%
  P4 (door, unrun) probe influence transfers to the target API model:
     Spearman rho >= 0.5 between probe and target chunk influences

Stdlib only for the lexical backend; the llama backend needs
llama-cpp-python and numpy.
"""
import argparse, json, math, os, random, re, statistics, time

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
]
WORDS = ["amber", "cobalt", "granite", "willow", "ember", "slate",
         "harbor", "cedar", "quartz", "falcon", "maple", "onyx"]


def make_items(n, k, seed):
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        vaults = rng.sample(range(100, 999), 5)
        codes = [f"{rng.choice(WORDS)}-{rng.randint(10, 99)}" for _ in vaults]
        facts = [f"The access code for vault {v} is {c}." for v, c in zip(vaults, codes)]
        chunks = facts[:4] + [rng.choice(FILLER) for _ in range(k - 4)]
        rng.shuffle(chunks)
        items.append(dict(chunks=chunks, q=f"What is the access code for vault {vaults[0]}?",
                          a=codes[0], needle=chunks.index(facts[0]), null=False))
    # C2 null items: vault 5 is never placed in context
    for _ in range(max(1, n // 10)):
        vaults = rng.sample(range(100, 999), 5)
        codes = [f"{rng.choice(WORDS)}-{rng.randint(10, 99)}" for _ in vaults]
        facts = [f"The access code for vault {v} is {c}." for v, c in zip(vaults[:4], codes[:4])]
        chunks = facts + [rng.choice(FILLER) for _ in range(k - 4)]
        rng.shuffle(chunks)
        items.append(dict(chunks=chunks, q=f"What is the access code for vault {vaults[4]}?",
                          a=codes[4], needle=None, null=True))
    return items


class Lexical:
    """Plumbing check only. It sees the answer string, so it is not a probe."""
    name = "lexical"

    def tokens(self, s):
        return max(1, len(s) // 4)

    def score(self, ctx, q, a):
        c = set(re.findall(r"[a-z0-9-]+", ctx.lower()))
        w = re.findall(r"[a-z0-9-]+", a.lower())
        hit = sum(x in c for x in w) / max(1, len(w))
        return math.log((hit + 0.01) / 1.01)

    def answer(self, ctx, q):
        raise SystemExit("lexical backend cannot generate; use --mode oracle")


class LlamaProbe:
    name = "llama"

    def __init__(self, path, n_ctx):
        from llama_cpp import Llama
        import numpy as np
        self.np = np
        self.m = Llama(model_path=path, n_ctx=n_ctx, logits_all=True, verbose=False)

    def _prompt(self, ctx, q):
        return f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"

    def tokens(self, s):
        return len(self.m.tokenize(s.encode(), add_bos=False))

    def score(self, ctx, q, a):
        np = self.np
        pt = self.m.tokenize(self._prompt(ctx, q).encode())
        at = self.m.tokenize((" " + a.strip()).encode(), add_bos=False)
        self.m.reset()
        self.m.eval(pt + at)
        lp = 0.0
        for j, t in enumerate(at):
            row = np.asarray(self.m.scores[len(pt) + j - 1], dtype=np.float64)
            mx = row.max()
            lp += row[t] - (mx + math.log(np.exp(row - mx).sum()))
        return float(lp)

    def answer(self, ctx, q):
        out = self.m.create_completion(self._prompt(ctx, q), max_tokens=12,
                                       temperature=0.0, stop=["\n"])
        return out["choices"][0]["text"].strip() or "?"


class HFProbe:
    """transformers backend (container stand-in for the on-device GGUF probe)."""

    def __init__(self, repo):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
        self.name = "hf:" + repo

    def _prompt(self, ctx, q):
        return f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"

    def tokens(self, s):
        return len(self.tok(s, add_special_tokens=False).input_ids)

    def score(self, ctx, q, a):
        t = self.torch
        p = self.tok(self._prompt(ctx, q), add_special_tokens=False).input_ids
        at = self.tok(" " + a.strip(), add_special_tokens=False).input_ids
        ids = t.tensor([p + at])
        with t.no_grad():
            lg = self.m(ids).logits[0].double().log_softmax(-1)
        return float(sum(lg[len(p) + j - 1, x] for j, x in enumerate(at)))

    def answer(self, ctx, q):
        t = self.torch
        ids = self.tok(self._prompt(ctx, q), return_tensors="pt", add_special_tokens=False).input_ids
        with t.no_grad():
            out = self.m.generate(ids, max_new_tokens=8, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        s = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return s.strip().split("\n")[0].strip() or "?"


def select(policy, infl, toks, budget, rng):
    idx = list(range(len(infl)))
    if policy == "eta":
        order = sorted(idx, key=lambda i: -infl[i] / toks[i])
    elif policy == "eta_shuffled":
        perm = infl[:]
        rng.shuffle(perm)
        order = sorted(idx, key=lambda i: -perm[i] / toks[i])
    elif policy == "recency":
        order = idx[::-1]
    else:
        order = idx[:]
        rng.shuffle(order)
    keep, used = [], 0
    for i in order:
        if used + toks[i] <= budget:
            keep.append(i)
            used += toks[i]
    return sorted(keep), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["lexical", "llama", "hf"], default="lexical")
    ap.add_argument("--hf_repo", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model", help="GGUF path for llama backend")
    ap.add_argument("--mode", choices=["oracle", "self"], default="oracle")
    ap.add_argument("--items", type=int, default=50)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_ctx", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "token_veritas_report.json"))
    args = ap.parse_args()

    if args.backend == "lexical":
        be = Lexical()
    elif args.backend == "hf":
        be = HFProbe(args.hf_repo)
    else:
        be = LlamaProbe(args.model, args.n_ctx)
    model_backend = args.backend != "lexical"
    hit = lambda out, gold: gold.lower() in out.lower()
    full_correct, self_matches_gold = [], []
    acc = {}
    rng = random.Random(args.seed)
    items = make_items(args.items, args.chunks, args.seed)
    ratios = [0.1, 0.2, 0.3, 0.5]
    policies = ["eta", "eta_shuffled", "random", "recency"]
    ranks, null_max, needle_infl = [], [], []
    keep = {(p, r): [] for p in policies for r in ratios}
    dlp = {(p, r): [] for p in policies for r in ratios}
    calls, t0 = 0, time.time()

    for it in items:
        ch, q = it["chunks"], it["q"]
        toks = [be.tokens(c) for c in ch]
        full_ans = be.answer("\n".join(ch), q) if model_backend else it["a"]
        target = it["a"] if args.mode == "oracle" else full_ans
        if model_backend and not it["null"]:
            full_correct.append(hit(full_ans, it["a"]))
            if args.mode == "self":
                self_matches_gold.append(hit(target, it["a"]))
        full = be.score("\n".join(ch), q, target)
        infl = [full - be.score("\n".join(ch[:i] + ch[i + 1:]), q, target) for i in range(len(ch))]
        calls += 1 + len(ch)
        if it["null"]:
            null_max.append(max(abs(x) for x in infl))
            continue
        n = it["needle"]
        needle_infl.append(infl[n])
        ranks.append(1 + sum(infl[j] >= infl[n] for j in range(len(ch)) if j != n))
        total = sum(toks)
        for r in ratios:
            for p in policies:
                kept, _ = select(p, infl, toks, r * total, rng)
                keep[(p, r)].append(n in kept)
                if model_backend:
                    sub = "\n".join(ch[i] for i in kept)
                    s = be.score(sub, q, it["a"])
                    dlp[(p, r)].append(s - be.score("\n".join(ch), q, it["a"]))
                    acc.setdefault((p, r), []).append(hit(be.answer(sub, q), it["a"]))
                    calls += 3

    elapsed = time.time() - t0
    m = len(ranks)
    rep = {
        "backend": be.name, "mode": args.mode, "items": m, "null_items": len(null_max),
        "chunks_per_item": args.chunks, "seed": args.seed,
        "C1_rank1_rate": round(sum(r == 1 for r in ranks) / m, 3),
        "C1_rank_le2_rate": round(sum(r <= 2 for r in ranks) / m, 3),
        "C2_null_max_abs_infl_mean": round(statistics.mean(null_max), 4),
        "needle_infl_mean": round(statistics.mean(needle_infl), 4),
        "retention": {f"{p}@{r}": round(sum(keep[(p, r)]) / m, 3) for r in ratios for p in policies},
        "gold_logprob_delta": {f"{p}@{r}": round(statistics.mean(dlp[(p, r)]), 4)
                               for r in ratios for p in policies if dlp[(p, r)]},
        "probe_calls": calls, "seconds": round(elapsed, 2),
    }
    if model_backend:
        rep["full_context_accuracy"] = round(sum(full_correct) / m, 3)
        if self_matches_gold:
            rep["self_target_equals_gold"] = round(sum(self_matches_gold) / m, 3)
        rep["pruned_accuracy"] = {f"{p}@{r}": round(sum(v) / len(v), 3)
                                  for (p, r), v in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0]))}
    print(json.dumps(rep, indent=1))
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
