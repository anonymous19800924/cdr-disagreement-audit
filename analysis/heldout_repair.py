import csv, json, os
from collections import Counter
import numpy as np
from common import DATA, load_anchor, load_heldout, fit_theta, predict_proba, save

anchor = load_anchor()
held = load_heldout()
theta = fit_theta(anchor)
p = np.array([predict_proba(theta, r) for r in held])
y = np.array([r["z"] for r in held])
print("anchor-fit heldout acc", (p >= 0.5).mean() * len(y), "/", len(y), ((p >= 0.5) == y).mean())

# per-subgroup
for sub in ("A", "B"):
    idx = [i for i, r in enumerate(held) if r["sub"] == sub]
    print(sub, len(idx), ((p[idx] >= 0.5) == y[idx]).mean(), ((p[idx] >= 0.5) == y[idx]).sum())

# write predictions
out = os.path.join(DATA, "heldout_predictions.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["item_id", "key", "subgroup", "consensus_label", "posterior_incorrect", "predicted_label"])
    for r, pi, yi in zip(held, p, y):
        w.writerow([r["item_id"], r["key"], r["sub"], "INCORRECT" if yi else "CORRECT", f"{pi:.6f}", 1 if pi >= 0.5 else 0])
print("wrote", out)

# gap bootstrap + McNemar
rb = np.array([1 if r["y"][1] else 0 for r in held])
cal_ok = (p >= 0.5).astype(int) == y
rb_ok = rb == y
rng = np.random.RandomState(2026)
gaps = []
for _ in range(10000):
    idx = rng.randint(0, len(y), len(y))
    gaps.append(cal_ok[idx].mean() - rb_ok[idx].mean())
b = int((((p >= 0.5).astype(int) == y) & (rb != y)).sum())
c = int((((p >= 0.5).astype(int) != y) & (rb == y)).sum())
from math import comb
n = b + c
p_exact = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0
chi = (abs(b - c) - 1) ** 2 / n if n else 0
from scipy.stats import chi2
p_asy = 1 - chi2.cdf(chi, 1) if n else 1.0
print("gap CI", np.percentile(gaps, [2.5, 97.5]), "b,c", b, c, "exact", p_exact, "asy", p_asy)

# tie sensitivity
tie = [(r, pi) for r, pi in zip(held, p) if r["z"] == 0]
# identify ties: consensus CORRECT but four-family 2-2; recompute from votes
votes = {}
for row in [json.loads(l) for l in open(os.path.join(DATA, "cross_family_heldout.jsonl"), encoding="utf-8")]:
    if row.get("majority"):
        votes.setdefault(row["item_id"], {})[row["family"]] = row["majority"]
fams = ("DS-PRO", "GLM", "QWEN", "KIMI")
ties = []
for r in held:
    fv = votes.get(r["item_id"], {})
    if all(f in fv for f in fams):
        c = Counter(fv[f] for f in fams).most_common()
        if len(c) > 1 and c[0][1] == c[1][1]:
            ties.append(r["item_id"])
print("ties", len(ties), ties)
# opposite rule: ties -> INCORRECT
alt_y = y.copy()
for i, r in enumerate(held):
    if r["item_id"] in ties:
        alt_y[i] = 1
print("opposite tie rule posterior/rb/conf/base:",
      ((p >= 0.5) == alt_y).mean(), (rb == alt_y).mean(), ((1 - rb) == alt_y).mean(), alt_y.mean())

# update heldout report
rp = os.path.join(DATA, "heldout_report.json")
rep = json.load(open(rp, encoding="utf-8"))
rep["bootstrap_ci95_cal_minus_rb"] = [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
rep["bootstrap_reps"] = 10000
rep["seed"] = 2026
rep["mcnemar"] = {"b_cal_right_rb_wrong": b, "c_cal_wrong_rb_right": c,
                  "p_exact_two_sided": float(p_exact), "p_asymptotic_continuity_corrected": float(p_asy)}
rep["heldout_ties"] = {"n": len(ties), "rule": "stored consensus scores ties CORRECT",
                       "opposite_rule": {"posterior": float(((p >= 0.5) == alt_y).mean()),
                                         "rebuttal": float((rb == alt_y).mean()),
                                         "conflict": float(((1 - rb) == alt_y).mean()),
                                         "base_rate": float(alt_y.mean())}}
json.dump(rep, open(rp, "w", encoding="utf-8"), indent=1)
print("updated", rp)
