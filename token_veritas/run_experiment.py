"""Run 2: DPP vs top-k vs leave-one-out context selection under a token budget.

python3 -m token_veritas.run_experiment --condition R --items 40 --seed 2026
python3 -m token_veritas.run_experiment --condition N --items 40 --seed 2027
"""
import argparse, json, math, os, random, time
import numpy as np
from .dpp import kernel, greedy_select, topk_select
from .tasks import make_items, coverage

BETA = 3.0
BUDGETS = [0.1, 0.2, 0.3]
POLICIES = ["random", "topk_cos", "dpp_cos", "loo", "dpp_loo", "dpp_uniform"]


class Answerer:
    def __init__(self, repo):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.t = torch
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
        self._cache = {}

    def prompt(self, ctx, q):
        return f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"

    def ntok(self, s):
        return len(self.tok(s, add_special_tokens=False).input_ids)

    def logprob(self, ctx, q, a):
        p = self.tok(self.prompt(ctx, q), add_special_tokens=False).input_ids
        at = self.tok(" " + a.strip(), add_special_tokens=False).input_ids
        with self.t.no_grad():
            lg = self.m(self.t.tensor([p + at])).logits[0].double().log_softmax(-1)
        return float(sum(lg[len(p) + j - 1, x] for j, x in enumerate(at)))

    def answer(self, ctx, q):
        key = (ctx, q)
        if key not in self._cache:
            ids = self.tok(self.prompt(ctx, q), return_tensors="pt", add_special_tokens=False).input_ids
            with self.t.no_grad():
                out = self.m.generate(ids, max_new_tokens=40, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
            s = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            self._cache[key] = s.strip().split("\n")[0].strip()
        return self._cache[key]


def correct(ans, gold):
    a = ans.lower()
    return all(g.lower() in a for g in gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["R", "N"], required=True)
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--answerer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join("results", f"run2_{a.condition}.json")

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(a.embedder)
    llm = Answerer(a.answerer)
    items = make_items(a.items, a.seed, copies=4 if a.condition == "R" else 1)
    rng = random.Random(a.seed + 1)
    rows, t0 = [], time.time()

    for n, it in enumerate(items):
        ch, q = it["chunks"], it["question"]
        toks = [llm.ntok(c) for c in ch]
        total = sum(toks)
        E = emb.encode(ch, normalize_embeddings=True)
        qv = emb.encode([q], normalize_embeddings=True)[0]
        rel = (E @ qv).astype(float)

        full_ctx = "\n".join(ch)
        full_ans = llm.answer(full_ctx, q)
        # LOO influence, deployable (self) mode: target is the model's own full-context answer.
        base = llm.logprob(full_ctx, q, full_ans)
        infl = np.array([base - llm.logprob("\n".join(ch[:i] + ch[i + 1:]), q, full_ans)
                         for i in range(len(ch))])
        mx = infl.max()
        q_loo = np.exp(BETA * np.clip(infl, 0, None) / mx) if mx > 0 else np.ones(len(ch))

        row = dict(item=n, total_tokens=total, full_answer=full_ans,
                   full_correct=correct(full_ans, it["gold"]), sel={})
        for b in BUDGETS:
            B = b * total
            picks = {
                "random": topk_select([rng.random() for _ in ch], toks, B)[0],
                "topk_cos": topk_select(rel.tolist(), toks, B)[0],
                "dpp_cos": greedy_select(kernel(np.exp(BETA * rel), E), toks, B)[0],
                "loo": topk_select(infl.tolist(), toks, B, per_token=True)[0],
                "dpp_loo": greedy_select(kernel(q_loo, E), toks, B)[0],
                "dpp_uniform": greedy_select(kernel(np.ones(len(ch)), E), toks, B)[0],
            }
            for p, kept in picks.items():
                ans = llm.answer("\n".join(ch[i] for i in kept), q)
                row["sel"][f"{p}@{b}"] = dict(
                    kept=kept, tokens=sum(toks[i] for i in kept),
                    coverage=coverage(it, kept), correct=correct(ans, it["gold"]), answer=ans)
        rows.append(row)
        print(f"[{a.condition}] item {n + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    rep = dict(condition=a.condition, seed=a.seed, items=len(rows), answerer=a.answerer,
               embedder=a.embedder, beta=BETA, budgets=BUDGETS, seconds=round(time.time() - t0, 1),
               full_accuracy=round(sum(r["full_correct"] for r in rows) / len(rows), 3), rows=rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
