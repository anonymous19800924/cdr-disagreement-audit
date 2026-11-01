"""B3: signal ablation on the held-out set."""
import numpy as np
from common import load_anchor, load_heldout, fit_theta, evaluate, ece, brier, acc, save

anchor = load_anchor()
held = load_heldout()

variants = {
    "channels_only": (0, 1),
    "channels_meta": (0, 1, 2),
    "all_four": (0, 1, 2, 3),
    "leave_conflict": (1, 2, 3),
    "leave_rebuttal": (0, 2, 3),
    "leave_meta": (0, 1, 3),
    "leave_ac": (0, 1, 2),
}
out = {}
for name, sig in variants.items():
    theta = fit_theta(anchor, signals=sig)
    p, y = evaluate(theta, held, signals=sig)
    out[name] = {"signals": list(sig), "acc": acc(p, y), "brier": brier(p, y), "ece": ece(p, y)[0]}
    print(f"{name:16s} acc={out[name]['acc']:.4f} brier={out[name]['brier']:.4f} ece={out[name]['ece']:.4f}")
save("ablation.json", out)
