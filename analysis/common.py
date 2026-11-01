"""Shared loaders and model for the released calibrated-adjudication artifacts.

All inputs come from ../artifact/data (released package). No network, no server.
Conventions: z=1 means "opinion incorrect"; four signals are
[conflict, rebuttal, meta-review stance, AC stance].
"""
import csv, json, os
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "artifact", "data")
if not os.path.isdir(DATA):
    DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

FAMS = ("DS-PRO", "GLM", "QWEN", "KIMI")


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_stance():
    meta = {r["key"]: r for r in _jsonl(os.path.join(DATA, "stance_results.jsonl"))}
    ac = {r["key"]: r for r in _jsonl(os.path.join(DATA, "stance_ac_localized.jsonl"))}
    return meta, ac


def signal_row(key, verdicts, subgroup, meta, ac):
    y = np.full(4, np.nan)
    v = np.zeros(4, bool)
    y[0] = 1 if verdicts[0] == "incorrect" else 0
    v[0] = True
    y[1] = 1 if verdicts[1] == "incorrect" else 0
    v[1] = True
    ms = (meta.get(key) or {}).get("stance")
    if ms in ("SUPPORTS", "OPPOSES"):
        y[2] = 1 if ms == "OPPOSES" else 0
        v[2] = True
    a = ac.get(key)
    if a and str(a.get("is_response")) == "true" and a.get("stance") in ("SUPPORTS", "OPPOSES"):
        y[3] = 1 if a["stance"] == "OPPOSES" else 0
        v[3] = True
    return {"key": key, "sub": subgroup, "y": y, "v": v}


def load_anchor():
    meta, ac = load_stance()
    cons = {r["item_id"]: r for r in _jsonl(os.path.join(DATA, "anchor_consensus.jsonl"))}
    key = {r["item_id"]: r for r in _jsonl(os.path.join(DATA, "anchor_key.jsonl"))}
    rows = []
    for iid, c in cons.items():
        if c.get("consensus") not in ("CORRECT", "INCORRECT"):
            continue
        k = key[iid]
        row = signal_row(k["key"], (k["verdict_conflict"], k["verdict_rebuttal"]), k["subgroup"], meta, ac)
        row.update({"item_id": iid, "z": 1 if c["consensus"] == "INCORRECT" else 0,
                    "golden": k["golden_majority"], "vconf": k["verdict_conflict"],
                    "vreb": k["verdict_rebuttal"], "eps": k.get("eps_meta", "")})
        rows.append(row)
    return rows


def load_heldout():
    meta, ac = load_stance()
    votes = defaultdict(dict)
    for r in _jsonl(os.path.join(DATA, "cross_family_heldout.jsonl")):
        if r.get("majority"):
            votes[r["item_id"]][r["family"]] = r["majority"]
        else:
            votes[r["item_id"]].setdefault(r["family"], None)
    items = {r["item_id"]: r for r in csv.DictReader(open(os.path.join(DATA, "heldout_items.csv"), encoding="utf-8"))}
    rows = []
    for iid, fv in votes.items():
        if not all(fv.get(f) for f in FAMS):
            continue
        it = items.get(iid)
        if it is None:
            continue
        cons = Counter([fv[f] for f in FAMS]).most_common(1)[0][0]
        row = signal_row(it["key"], (it["verdict_conflict"], it["verdict_rebuttal"]), it["subgroup"], meta, ac)
        row.update({"item_id": iid, "z": 1 if cons == "INCORRECT" else 0,
                    "vconf": it["verdict_conflict"], "vreb": it["verdict_rebuttal"]})
        rows.append(row)
    return rows


EPS = 1e-6


def fit_theta(rows, signals=(0, 1, 2, 3)):
    theta = []
    for ch in range(4):
        t = np.zeros((2, 2)) + EPS
        if ch not in signals:
            theta.append(t / t.sum(axis=0, keepdims=True))
            continue
        for r in rows:
            if r["v"][ch]:
                t[int(r["y"][ch]), r["z"]] += 1
        theta.append(t / t.sum(axis=0, keepdims=True))
    return theta


def predict_proba(theta, row, signals=(0, 1, 2, 3), tau=0.0):
    lg = tau
    for ch in signals:
        if row["v"][ch]:
            yv = int(row["y"][ch])
            lg += np.log(max(theta[ch][yv, 1], 1e-12)) - np.log(max(theta[ch][yv, 0], 1e-12))
    return 1.0 / (1.0 + np.exp(-lg))


def evaluate(theta, rows, signals=(0, 1, 2, 3), tau=0.0):
    p = np.array([predict_proba(theta, r, signals, tau) for r in rows])
    y = np.array([r["z"] for r in rows])
    return p, y


def ece(p, y, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    table = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        n = int(m.sum())
        if n == 0:
            table.append({"lo": float(lo), "hi": float(hi), "n": 0, "mean_p": None, "frac_pos": None})
            continue
        mp = float(p[m].mean())
        fp = float(y[m].mean())
        total += n / len(p) * abs(mp - fp)
        table.append({"lo": float(lo), "hi": float(hi), "n": n, "mean_p": mp, "frac_pos": fp})
    return float(total), table


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def acc(p, y):
    return float(((p >= 0.5).astype(int) == y).mean())


def kappa(pred, y):
    pred = np.asarray(pred).astype(int)
    y = np.asarray(y).astype(int)
    n = len(y)
    po = float((pred == y).mean())
    p1 = float(pred.mean()); q1 = float(y.mean())
    pe = p1 * q1 + (1 - p1) * (1 - q1)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def bootstrap(p, y, reps=1000, seed=2026):
    rng = np.random.RandomState(seed)
    vals = {"acc": [], "brier": [], "ece": [], "kappa": []}
    n = len(y)
    for _ in range(reps):
        idx = rng.randint(0, n, n)
        vals["acc"].append(acc(p[idx], y[idx]))
        vals["brier"].append(brier(p[idx], y[idx]))
        vals["ece"].append(ece(p[idx], y[idx])[0])
        vals["kappa"].append(kappa((p[idx] >= 0.5), y[idx]))
    return {k: {"mean": float(np.mean(v)), "lo": float(np.percentile(v, 2.5)),
                "hi": float(np.percentile(v, 97.5))} for k, v in vals.items()}


def save(name, obj):
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print("saved", os.path.relpath(path, HERE))
