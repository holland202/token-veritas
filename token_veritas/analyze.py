"""Score Run 2 against the registered predictions. python3 -m token_veritas.analyze"""
import json, math, os, sys
import numpy as np

POL = ["random", "topk_cos", "dpp_cos", "loo", "dpp_loo", "dpp_uniform"]


def load(path):
    with open(path) as f:
        return json.load(f)


def col(rep, key, metric):
    return np.array([float(r["sel"][key][metric]) for r in rep["rows"]])


def sign_test(a, b):
    """Exact two-sided McNemar / sign test on paired binary outcomes."""
    wins = int(np.sum((a == 1) & (b == 0)))
    losses = int(np.sum((a == 0) & (b == 1)))
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return wins, losses, min(1.0, 2 * p)


def boot_ci(d, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(reps, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def table(rep):
    lines = [f"condition {rep['condition']}  n={rep['items']}  full-context accuracy {rep['full_accuracy']}",
             "| policy | budget | tokens (mean) | coverage | accuracy |", "|---|---|---|---|---|"]
    for b in rep["budgets"]:
        for p in POL:
            k = f"{p}@{b}"
            lines.append(f"| {p} | {b} | {col(rep, k, 'tokens').mean():.1f} | "
                         f"{col(rep, k, 'coverage').mean():.3f} | {col(rep, k, 'correct').mean():.3f} |")
    return "\n".join(lines)


def main(rdir="results"):
    R = load(os.path.join(rdir, "run2_R.json"))
    N = load(os.path.join(rdir, "run2_N.json"))
    k = lambda p: f"{p}@0.2"
    out = [table(R), "", table(N), "", "## Registered predictions (budget 0.2)"]
    verdict = {}

    d1 = col(R, k("dpp_cos"), "coverage") - col(R, k("topk_cos"), "coverage")
    lo, hi = boot_ci(d1)
    verdict["P1"] = d1.mean() >= 0.20
    out.append(f"P1 coverage dpp_cos - topk_cos (R) = {d1.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
               f"threshold >= 0.20 -> {'PASS' if verdict['P1'] else 'FAIL'}")

    a, b = col(R, k("dpp_cos"), "correct"), col(R, k("topk_cos"), "correct")
    w, l, p = sign_test(a, b)
    lo, hi = boot_ci(a - b)
    verdict["P2"] = (a - b).mean() >= 0.15 and p < 0.05
    out.append(f"P2 accuracy dpp_cos - topk_cos (R) = {(a - b).mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
               f"wins {w} losses {l} p={p:.4f}  threshold >= 0.15 & p<0.05 -> {'PASS' if verdict['P2'] else 'FAIL'}")

    d3 = col(N, k("dpp_cos"), "coverage") - col(N, k("topk_cos"), "coverage")
    verdict["P3"] = abs(d3.mean()) <= 0.10
    out.append(f"P3 [control] coverage dpp_cos - topk_cos (N) = {d3.mean():.3f}  threshold |d| <= 0.10 -> "
               f"{'PASS' if verdict['P3'] else 'FAIL'}")

    u, r = col(R, k("dpp_uniform"), "coverage").mean(), col(R, k("random"), "coverage").mean()
    verdict["P4"] = u <= r + 0.10
    out.append(f"P4 [control] coverage dpp_uniform {u:.3f} vs random {r:.3f} (R)  threshold <= random+0.10 -> "
               f"{'PASS' if verdict['P4'] else 'FAIL'}")

    ln, lr = col(N, k("loo"), "coverage").mean(), col(R, k("loo"), "coverage").mean()
    verdict["P5"] = ln - lr >= 0.20
    out.append(f"P5 LOO coverage N {ln:.3f} -> R {lr:.3f}, drop {ln - lr:.3f}  threshold >= 0.20 -> "
               f"{'PASS' if verdict['P5'] else 'FAIL'}")

    ad, al = col(R, k("dpp_cos"), "correct").mean(), col(R, k("loo"), "correct").mean()
    verdict["P6"] = ad >= al - 0.05
    out.append(f"P6 accuracy dpp_cos {ad:.3f} vs loo {al:.3f} (R)  threshold dpp >= loo-0.05 -> "
               f"{'PASS' if verdict['P6'] else 'FAIL'}")

    if not verdict["P3"] and (verdict["P1"] or verdict["P2"]):
        out.append("KILL RULE: P3 failed, so any P1/P2 win is declared CONFOUNDED.")
    out.append(f"seconds: R {R['seconds']}  N {N['seconds']}")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(rdir, "run2_analysis.md"), "w") as f:
        f.write(text + "\n")
    return verdict


if __name__ == "__main__":
    main(*sys.argv[1:])
