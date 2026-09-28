# -*- coding: utf-8 -*-
"""U1 配对统计：同一题集在 C/W 两条件下的判官采纳（McNemar 精确 + Newcombe）。
纯 numpy/math 实现（本机无 scipy）。"""
import json
import math
from pathlib import Path

SRC = Path("D:/cdr_research/out/u1_implanted.jsonl")
OUT = Path("D:/cdr_research/out/_u1_stats.txt")


def mcnemar_exact(b, c):
    """b,c = discordant counts (C-adopt&W-not, W-adopt&C-not)。binomial two-sided."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
    return min(1.0, p)


def wilson_up(k, n, z=1.96):
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c + h


def newcombe(k1, n1, k0, n0):
    """Newcombe paired-2? 用非配对 difference interval (method 10) 近似独立样本。"""
    z = 1.96
    def wilson(k, n):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return c - h, c + h
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    p1, p0 = k1 / n1, k0 / n0
    d = p1 - p0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    lo = d - z * se - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = d + z * se + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return lo, hi


rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
out = []
out.append("=" * 76)
out.append("U1 IMPLANTED-CONSENSUS: paired adoption stats (same 80 items, C vs W)")
out.append("=" * 76)
for judge in sorted({r["judge"] for r in rows}):
    rs = [r for r in rows if r["judge"] == judge]
    by = {(r["index"], r["condition"]): r for r in rs}
    idxs = sorted({r["index"] for r in rs})
    aC = [by[(i, "C")]["adopted"] for i in idxs]
    aW = [by[(i, "W")]["adopted"] for i in idxs]
    n = len(idxs)
    kC, kW = sum(aC), sum(aW)
    b = sum(1 for x, y in zip(aC, aW) if x and not y)   # C adopt, W not
    c = sum(1 for x, y in zip(aC, aW) if y and not x)   # W adopt, C not
    p = mcnemar_exact(b, c)
    lo, hi = newcombe(kC, n, kW, n)
    # judge_correct in W (rejection quality)
    jc = [by[(i, "W")] for i in idxs]
    rej = [r for r in jc if not r["adopted"]]
    rej_correct = sum(1 for r in rej if r["judge_correct"])
    out.append(f"\nJudge {judge}  (n={n})")
    out.append(f"  P(adopt|C) = {kC}/{n} = {kC/n:.3f}")
    out.append(f"  P(adopt|W) = {kW}/{n} = {kW/n:.3f}")
    out.append(f"  delta_disc = {(kC-kW)/n:+.3f}   Newcombe 95% CI [{lo:+.3f}, {hi:+.3f}]")
    out.append(f"  paired discordant: C-only={b}, W-only={c}  exact McNemar p = {p:.2e}")
    out.append(f"  W condition rejections: {len(rej)}/{n}; judge answered correctly after rejection: {rej_correct}")
    # template fallback count
    tf = sum(1 for r in rs if r.get("template_fallback"))
    out.append(f"  template-fallback records: {tf}")

txt = "\n".join(out)
OUT.write_text(txt, encoding="utf-8")
print(txt)
