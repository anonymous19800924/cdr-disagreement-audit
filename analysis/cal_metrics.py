"""B1: calibration measurement on the held-out set (accuracy, Brier, ECE, reliability bins)."""
import json, sys
import numpy as np
from common import (load_anchor, load_heldout, fit_theta, evaluate, ece, brier, acc,
                    bootstrap, save)

anchor = load_anchor()
held = load_heldout()
theta = fit_theta(anchor)
p, y = evaluate(theta, held)

res = {
    "n_anchor": len(anchor), "n_heldout": len(held),
    "posterior": {
        "acc": acc(p, y), "brier": brier(p, y), "ece": ece(p, y)[0],
        "reliability_bins": ece(p, y)[1],
        "bootstrap": bootstrap(p, y, reps=1000, seed=2026),
    },
}
# hard channel predictors
for name, ch in (("conflict", 0), ("rebuttal", 1)):
    ph = np.array([1.0 if r["y"][ch] == 1 else 0.0 for r in held])
    res[name] = {"acc": acc(ph, y), "brier": brier(ph, y), "ece": ece(ph, y)[0]}
# majority class
pm = np.full(len(y), 1.0 if y.mean() >= 0.5 else 0.0)
res["majority"] = {"acc": acc(pm, y), "brier": brier(pm, y), "ece": ece(pm, y)[0]}
res["label_rate"] = float(y.mean())

save("calibration.json", res)

ok = (abs(res["posterior"]["acc"] - 0.7123) < 1e-4
      and abs(res["rebuttal"]["acc"] - 0.6849) < 1e-4
      and abs(res["conflict"]["acc"] - 0.3151) < 1e-4)
print(f"acc={res['posterior']['acc']:.4f} brier={res['posterior']['brier']:.4f} ece={res['posterior']['ece']:.4f}")
print(f"rebuttal={res['rebuttal']['acc']:.4f} conflict={res['conflict']['acc']:.4f}")
print("SANITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
