"""B2: base-rate / tau sweep on the held-out set."""
import numpy as np
from common import load_anchor, load_heldout, fit_theta, evaluate, ece, brier, acc, save

anchor = load_anchor()
held = load_heldout()
theta = fit_theta(anchor)
grid = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
rows = []
for tau in grid:
    p, y = evaluate(theta, held, tau=tau)
    rows.append({"tau": tau, "acc": acc(p, y), "brier": brier(p, y), "ece": ece(p, y)[0]})
base = float(np.mean([r["z"] for r in held]))
res = {"label_rate_heldout": base,
       "tau_logit_baserate": float(np.log(base / (1 - base))),
       "grid": rows,
       "anchor_label_rate": float(np.mean([r["z"] for r in anchor]))}
save("tau_sweep.json", res)
for r in rows:
    print(f"tau={r['tau']:+.2f} acc={r['acc']:.4f} brier={r['brier']:.4f} ece={r['ece']:.4f}")
print("held-out positive rate:", round(base, 4))
