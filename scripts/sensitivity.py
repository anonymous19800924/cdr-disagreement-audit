#!/usr/bin/env python3
"""Regenerate the held-out table and the supervision-noise sensitivity curve (Secs. 6.3-6.4).

Fit: 80 anchor items, five-vote consensus, channel verdicts + meta/AC stance signals.
Eval: 73 held-out items with full four-family votes; reference = four-family majority.
Outputs: calibrated / rebuttal / conflict accuracy, subgroup cells, and the eps sweep.
"""
import csv, json, os
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# ---- anchor fit set ----
mc80 = [json.loads(l) for l in open(os.path.join(DATA, "anchor_consensus.jsonl"), encoding="utf-8")]
key80 = {json.loads(l)["item_id"]: json.loads(l)
         for l in open(os.path.join(DATA, "anchor_key.jsonl"), encoding="utf-8")}
st80 = {json.loads(l)["key"]: json.loads(l) for l in open(os.path.join(DATA, "stance_results.jsonl"), "rb")}
acl = {json.loads(l)["key"]: json.loads(l) for l in open(os.path.join(DATA, "stance_ac_localized.jsonl"), "rb")}

fit_rows = []
for r in mc80:
    if r["consensus"] is None:
        continue
    k = key80[r["item_id"]]
    y = np.full(4, np.nan); v = np.zeros(4, bool)
    y[0] = 1 if k["verdict_conflict"] == "incorrect" else 0; v[0] = True
    y[1] = 1 if k["verdict_rebuttal"] == "incorrect" else 0; v[1] = True
    ms = (st80.get(r["key"]) or {}).get("stance")
    if ms in ("SUPPORTS", "OPPOSES"):
        y[2] = 1 if ms == "OPPOSES" else 0; v[2] = True
    a = acl.get(r["key"])
    if a and str(a.get("is_response")) == "true" and a.get("stance") in ("SUPPORTS", "OPPOSES"):
        y[3] = 1 if a["stance"] == "OPPOSES" else 0; v[3] = True
    fit_rows.append({"z": 1 if r["consensus"] == "INCORRECT" else 0, "y": y, "v": v})

EPS = 1e-6
def fit_theta(rows):
    out = []
    for ch in range(4):
        t = np.zeros((2, 2)) + EPS
        for row in rows:
            if row["v"][ch]:
                t[int(row["y"][ch]), row["z"]] += 1
        out.append(t / t.sum(axis=0, keepdims=True))
    return out
theta = fit_theta(fit_rows)

# ---- held-out eval rows ----
held = defaultdict(dict)
for line in open(os.path.join(DATA, "cross_family_heldout.jsonl"), "rb"):
    r = json.loads(line)
    if r.get("majority"):
        held[r["item_id"]][r["family"]] = r["majority"]
fams = ("DS-PRO", "GLM", "QWEN", "KIMI")
full = [i for i, fv in held.items() if all(fv.get(f) for f in fams)]
items = {r["item_id"]: r for r in csv.DictReader(open(os.path.join(DATA, "heldout_items.csv"), encoding="utf-8"))}

rows = []
for iid in full:
    it = items.get(iid)
    if it is None:
        continue
    cons = Counter([held[iid][f] for f in fams]).most_common(1)[0][0]
    y = np.full(4, np.nan); v = np.zeros(4, bool)
    y[0] = 1 if it["verdict_conflict"] == "incorrect" else 0; v[0] = True
    y[1] = 1 if it["verdict_rebuttal"] == "incorrect" else 0; v[1] = True
    ms = (st80.get(it["key"]) or {}).get("stance")
    if ms in ("SUPPORTS", "OPPOSES"):
        y[2] = 1 if ms == "OPPOSES" else 0; v[2] = True
    a = acl.get(it["key"])
    if a and str(a.get("is_response")) == "true" and a.get("stance") in ("SUPPORTS", "OPPOSES"):
        y[3] = 1 if a["stance"] == "OPPOSES" else 0; v[3] = True
    rows.append({"cons": 1 if cons == "INCORRECT" else 0, "y": y, "v": v, "sub": it["subgroup"]})

def pred(th, r):
    lg = 0.0
    for ch in range(4):
        if r["v"][ch]:
            yv = int(r["y"][ch])
            lg += np.log(max(th[ch][yv, 1], 1e-12)) - np.log(max(th[ch][yv, 0], 1e-12))
    return 1 if lg > 0 else 0

Z = np.array([r["cons"] for r in rows])
P = np.array([pred(theta, r) for r in rows])
RB = np.array([int(r["y"][1]) for r in rows])
CF = np.array([int(r["y"][0]) for r in rows])
print(f"held-out n={len(rows)}: calibrated {(P==Z).mean():.4f}, rebuttal {(RB==Z).mean():.4f}, "
      f"conflict {(CF==Z).mean():.4f}, consensus incorrect {Z.mean():.4f}, majority class {max(Z.mean(),1-Z.mean()):.4f}")
for s in ("A", "B"):
    m = np.array([r["sub"] == s for r in rows])
    print(f"  sub{s}: n={m.sum()} cal={(P[m]==Z[m]).mean():.4f} rb={(RB[m]==Z[m]).mean():.4f} "
          f"cf={(CF[m]==Z[m]).mean():.4f} cons_inc={Z[m].mean():.4f}")

rng = np.random.RandomState(2027)
print("\neps sweep (20 trials, seed 2027):")
for eps in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
    accs = []
    for _ in range(20):
        n_flip = int(round(eps * len(fit_rows)))
        idx = set(rng.choice(len(fit_rows), n_flip, replace=False)) if n_flip else set()
        rows_f = [{"z": (1 - r["z"] if i in idx else r["z"]), "y": r["y"], "v": r["v"]}
                  for i, r in enumerate(fit_rows)]
        th = fit_theta(rows_f)
        preds = np.array([pred(th, r) for r in rows])
        accs.append((preds == Z).mean())
    print(f"  eps={eps:.2f}: {np.mean(accs):.4f} (std {np.std(accs):.4f})")
