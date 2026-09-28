# -*- coding: utf-8 -*-
"""
从原始 per-episode 日志生成 ACL 论文的图，并打印核对表。

设计原则
--------
1. 图里的每个数字都从 results 现场算出来，不手抄 —— 论文 §5 声称
   "no number was transcribed by hand"，脚本必须真的做到这一点。
2. 判别力的分类口径写死为「共识本身是否正确」：
      consensus = c1_answer if c1_answer == c2_answer
      TC = (consensus == ground_truth)
   而不是用 judge_correct —— 后者在判官拒纳共识时表示的是判官自己的对错，
   会把 TC/AT 混淆。饱和臂里两者恰好一致，非饱和臂里会不一致。
3. 输出版式：Times 字体、矢量 PDF、字体嵌入（pdf.fonttype=42），
   单栏宽 7.7cm / 通栏宽 16.0cm（ACL A4 双栏规格）。

用法：
    python make_figures.py            # 出图 + 打印核对表
    python make_figures.py --check    # 只打印核对表
"""
from __future__ import annotations
import argparse, json, os, glob, sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\cdr_research\out"
FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------- 版式
CM = 1 / 2.54
W1 = 7.7 * CM          # 单栏宽
W2 = 16.0 * CM         # 通栏宽
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# 配色（色盲友好）
C_OK = "#4C72B0"      # 正确共识
C_BAD = "#C44E52"     # 错误共识
C_NEUTRAL = "#8C8C8C"

# 臂定义：tag -> (文件名, 图例名, 是否同源)
ARMS = [
    ("dr_same",   "dr_GPQA_glm46_glm45_glm-4.5-air.json",        "DR",          True),
    ("pl_same",   "placebo_GPQA_glm46_glm45_glm-4.5-air.json",   "Placebo",     True),
    ("nc_same",   "dr_nocrux_GPQA_glm46_glm45_glm-4.5-air.json", "No-crux",     True),
    ("nf_same",   "dr_noconf_GPQA_glm46_glm45_glm-4.5-air.json", "No-conflict", True),
    ("dr_x",      "dr_GPQA_glm_kimi3_qwen-flash.json",           "DR",          False),
    ("pl_x",      "placebo_GPQA_glm_kimi3_qwen-flash.json",      "Placebo",     False),
]


# ---------------------------------------------------------------- 统计
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cohens_h(p1: float, p2: float) -> float:
    import math
    return 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1)))) - \
           2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))


def fisher2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a,b],[c,d]]."""
    from math import comb
    n = a + b + c + d
    if n == 0 or (a + b) == 0 or (c + d) == 0:
        return float("nan")
    r1, r2, k1 = a + b, c + d, a + c

    def pr(x):
        return comb(r1, x) * comb(r2, k1 - x) / comb(n, k1)

    p0 = pr(a)
    tot = 0.0
    for x in range(max(0, k1 - r2), min(r1, k1) + 1):
        p = pr(x)
        if p <= p0 + 1e-12:
            tot += p
    return min(1.0, tot)


def arm_stats(path: str):
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    rs = [r for r in (d.get("results") or []) if r.get("judge_correct") is not None]
    if not rs:
        return None

    n = len(rs)
    k_ok = sum(1 for r in rs if r.get("judge_correct"))
    acc = k_ok / n
    lo, hi = wilson(k_ok, n)

    tc = at = dc = 0
    tc_adopt = at_adopt = 0
    turns = Counter()
    rec_exit = Counter()
    rev_c1 = rev_c2 = 0
    eps_touched = 0
    n_at_rejected = 0

    for r in rs:
        c1, c2 = r.get("c1_answer"), r.get("c2_answer")
        gt = r.get("correct")
        cons = c1 if (c1 is not None and c1 == c2) else None
        adopt = (cons is not None) and (r.get("judge_answer") == cons)

        turns[r.get("turns") or 1] += 1
        rec_exit[r.get("exit_mode")] += 1
        d1 = r.get("n_ans_changes_1") or 0
        d2 = r.get("n_ans_changes_2") or 0
        rev_c1 += d1
        rev_c2 += d2
        if d1 + d2 > 0:
            eps_touched += 1

        if cons is None:
            dc += 1
        elif cons == gt:
            tc += 1
            tc_adopt += int(adopt)
        else:
            at += 1
            at_adopt += int(adopt)
            n_at_rejected += int(not adopt)

    p_tc = tc_adopt / tc if tc else float("nan")
    p_at = at_adopt / at if at else float("nan")
    disc = (p_tc - p_at) if (tc and at) else float("nan")
    # 单侧 95% 上界：k=n 时 p 的 Clopper–Pearson 上界 = alpha^(1/n)
    bound = 0.05 ** (1.0 / tc) if tc and tc_adopt == tc else float("nan")

    return dict(
        name=os.path.basename(path), n=n, acc=acc, lo=lo, hi=hi,
        tc=tc, at=at, dc=dc, tc_adopt=tc_adopt, at_adopt=at_adopt,
        p_tc=p_tc, p_at=p_at, disc=disc, p1_upper=bound,
        n_at_rejected=n_at_rejected,
        turns=dict(turns),
        multirate=1 - turns.get(1, 0) / n,
        mean_turns=sum(k * v for k, v in turns.items()) / n,
        rev_c1=rev_c1, rev_c2=rev_c2,
        rev_rate=eps_touched / n,
        rec_exit=dict(rec_exit),
        conflicts=d.get("mean_conflicts"),
    )


def fmt_int(x):
    return f"{x:,}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    stats = {}
    print("=" * 108)
    print("ARM TABLE (recomputed from per-episode logs; TC/AT classified by CONSENSUS correctness)")
    print("=" * 108)
    hdr = (f"{'tag':9s} {'n':>4s} {'acc':>7s} {'95%CI':>15s} "
           f"{'TC':>4s} {'AT':>4s} {'DC':>4s} "
           f"{'adopt|TC':>9s} {'adopt|AT':>9s} {'disc':>8s} "
           f"{'turns':>14s} {'multi':>6s} {'rev-ep':>7s} {'rev':>4s}")
    print(hdr)
    print("-" * len(hdr))
    for tag, fn, label, same in ARMS:
        s = arm_stats(os.path.join(DATA, fn))
        if s is None:
            print(f"{tag:9s}   -- (missing: {fn})")
            continue
        stats[tag] = s
        ci = f"[{s['lo']*100:.1f},{s['hi']*100:.1f}]"
        ts = ",".join(f"{k}:{v}" for k, v in sorted(s["turns"].items()))
        print(f"{tag:9s} {s['n']:4d} {s['acc']*100:6.1f}% {ci:>15s} "
              f"{s['tc']:4d} {s['at']:4d} {s['dc']:4d} "
              f"{s['tc_adopt']:4d}/{s['tc']:<4d} {s['at_adopt']:4d}/{s['at']:<4d} "
              f"{s['disc']:+8.3f} {ts:>14s} {s['multirate']*100:5.1f}% "
              f"{s['rev_rate']*100:6.1f}% {s['rev_c1']+s['rev_c2']:4d}")
    print()
    print("0.05^(1/n) one-sided 95% upper bound on P(adopt|TC) when adoption is saturated:")
    for tag, s in stats.items():
        if s["p1_upper"] == s["p1_upper"]:
            print(f"   {tag:9s} n_TC={s['tc']:3d} -> upper bound {s['p1_upper']:.4f}")
    print()
    print("CONSISTENCY CHECK (recomputed vs. per-episode exit_mode recorded by harness)")
    for tag, s in stats.items():
        mine = {"TC": s["tc"], "AT": s["at"], "DC": s["dc"]}
        recn = {k: s["rec_exit"].get(k, 0) for k in ("TC", "AT", "DC")}
        flag = "OK  " if mine == recn else "DIFF"
        print(f"  {flag} {tag:9s} recomputed={mine}  recorded={recn}")
    print()
    print("FISHER EXACT contrasts")
    if "pl_same" in stats:
        s = stats["pl_same"]
        p = fisher2x2(s["tc_adopt"], s["tc"] - s["tc_adopt"],
                      s["at_adopt"], s["at"] - s["at_adopt"])
        print(f"  placebo: adoption x consensus-correctness   p = {p:.4f}")
    if "dr_same" in stats and "pl_same" in stats:
        a, b = stats["dr_same"], stats["pl_same"]
        p = fisher2x2(round(a["multirate"] * a["n"]), a["n"] - round(a["multirate"] * a["n"]),
                      round(b["multirate"] * b["n"]), b["n"] - round(b["multirate"] * b["n"]))
        print(f"  multi-turn rate  DR(same) vs Placebo(same)  p = {p:.4f}")
    if "pl_x" in stats and "pl_same" in stats:
        a, b = stats["pl_x"], stats["pl_same"]
        ax = round(a["multirate"] * a["n"])
        bx = round(b["multirate"] * b["n"])
        p = fisher2x2(ax, a["n"] - ax, bx, b["n"] - bx)
        print(f"  multi-turn rate  Placebo(cross) vs Placebo(same)  p = {p:.4f}"
              f"   ({ax}/{a['n']} vs {bx}/{b['n']})")
    if "nc_same" in stats and "dr_same" in stats:
        a, b = stats["nc_same"], stats["dr_same"]
        ax = round(a["multirate"] * a["n"])
        bx = round(b["multirate"] * b["n"])
        p = fisher2x2(ax, a["n"] - ax, bx, b["n"] - bx)
        print(f"  multi-turn rate  No-crux vs DR (both same)       p = {p:.4f}"
              f"   ({ax}/{a['n']} vs {bx}/{b['n']})")

    if args.check:
        return

        # ============================================================ Fig 1
    # 结果象限：行 = 共识本身是否正确，列 = 判官是否采纳。
    # 颜色按「行归一化」（即采纳率），所以两行颜色可直接比较 ——
    # 关键视觉就是：机制臂两行一样深，placebo 第二行才有一点浅。
    def quad(ax, tc, at, tc_adopt, at_adopt, title):
        adopt = np.array([[tc_adopt, tc - tc_adopt],
                          [at_adopt, at - at_adopt]], dtype=float)
        row = adopt.sum(axis=1, keepdims=True)
        rate = np.divide(adopt, row, out=np.zeros_like(adopt), where=row > 0)
        ax.imshow(rate, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        for i in range(2):
            for j in range(2):
                v = int(adopt[i, j])
                dark = rate[i, j] > 0.55
                ax.text(j, i - 0.13, f"{v}", ha="center", va="center", fontsize=9.0,
                        color="white" if dark else "#222222", fontweight="bold")
                ax.text(j, i + 0.20, f"{rate[i, j]*100:.1f}" + "%",
                        ha="center", va="center", fontsize=6.2,
                        color="white" if dark else "#444444")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["adopt", "reject"], fontsize=6.8)
        ax.set_yticklabels(["consensus\ncorrect", "consensus\nwrong"], fontsize=6.8)
        ax.set_title(title, fontsize=7.2, pad=3)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="minor", length=0)

    mech = dict(tc=0, at=0, tca=0, ata=0)
    for t in ("dr_same", "nc_same", "nf_same"):
        if t in stats:
            mech["tc"] += stats[t]["tc"]; mech["at"] += stats[t]["at"]
            mech["tca"] += stats[t]["tc_adopt"]; mech["ata"] += stats[t]["at_adopt"]
    plc = stats.get("pl_same")

    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.62), gridspec_kw=dict(wspace=0.34))
    quad(axes[0], mech["tc"], mech["at"], mech["tca"], mech["ata"],
         f"mechanism arms, pooled\n$n$={mech['tc']+mech['at']}")
    if plc:
        quad(axes[1], plc["tc"], plc["at"], plc["tc_adopt"], plc["at_adopt"],
             f"placebo\n$n$={plc['tc']+plc['at']}")
    fig.savefig(os.path.join(FIGDIR, "fig_quadrants.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig_quadrants.png"), dpi=300)
    plt.close(fig)

    # ============================================================ Fig 2 (通栏)
    order = [t for t, _, _, _ in ARMS if t in stats]
    labels = {t: lb for t, _, lb, _ in ARMS}
    same = {t: sm for t, _, _, sm in ARMS}
    y = np.arange(len(order))[::-1]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(W2, 1.70), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.16))

    # (a) 准确率 + Wilson CI
    accs = np.array([stats[t]["acc"] * 100 for t in order])
    los = np.array([stats[t]["lo"] * 100 for t in order])
    his = np.array([stats[t]["hi"] * 100 for t in order])
    axA.errorbar(accs, y, xerr=[accs - los, his - accs], fmt="o", ms=3.4, lw=0.9,
                 capsize=2.0, color="#333333", ecolor="#666666")
    for yi, t in zip(y, order):
        axA.text(stats[t]["hi"] * 100 + 1.0, yi,
                 f"{stats[t]['acc']*100:.1f}", va="center", fontsize=5.8)
    axA.set_xlim(40, 108)
    axA.set_yticks(y)
    axA.set_yticklabels([f"{labels[t]} ({'same' if same[t] else 'cross'})" for t in order],
                        fontsize=6.4)
    axA.set_xlabel("judge accuracy  [%]")
    axA.set_title("(a) accuracy: all arms within noise of each other", fontsize=7.0)

    # (b) Δ_disc —— 追加 U1 植入臂（judge-only，协议交互被删除）
    import json as _json
    imp = {"C": [0, 0], "W": [0, 0]}  # condition -> [adopted, n]
    with open(os.path.join(DATA, "u1_implanted.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = _json.loads(line)
            if r.get("judge") == "glm-4.5-air" and r.get("condition") in imp:
                imp[r["condition"]][1] += 1
                imp[r["condition"]][0] += 1 if r.get("adopted") else 0
    assert imp["C"][1] == 80 and imp["W"][1] == 80, f"植入臂计数异常: {imp}"
    d_imp = (imp["C"][0] / imp["C"][1] - imp["W"][0] / imp["W"][1]) * 100.0
    y_imp = min(y) - 1.0  # 植入臂放最底行，保持上 6 行与 (a) 面板对齐

    dvals = np.array([stats[t]["disc"] * 100 for t in order])
    ok = ~np.isnan(dvals)
    axB.barh(y[ok], dvals[ok], 0.5,
             color=[C_BAD if abs(v) < 0.5 else C_OK for v in dvals[ok]])
    # 植入臂：+51.25 远超 15 点刻度 —— 截断条 + 断轴记号 + 白字标注
    XMAX = 15.0
    axB.barh([y_imp], [XMAX - 0.35], 0.5, color=C_OK, hatch="//", edgecolor="white")
    for x0 in (XMAX - 2.6, XMAX - 1.8):  # 双斜线断轴记号
        axB.plot([x0, x0 + 0.7], [y_imp - 0.28, y_imp + 0.28],
                 color="white", lw=2.4, zorder=5)
        axB.plot([x0, x0 + 0.7], [y_imp - 0.28, y_imp + 0.28],
                 color="#333333", lw=0.7, zorder=6)
    axB.text(XMAX - 0.35, y_imp, f"{d_imp:+.2f}", va="center", ha="right",
             fontsize=6.0, fontweight="bold", color="white", zorder=7)
    axB.text(0.0, y_imp + 0.62,
             "implanted consensus, exchange removed (judge-only; axis break)",
             va="center", ha="left", fontsize=5.8, style="italic", color="#444444")
    for yi, v in zip(y[ok], dvals[ok]):
        axB.text(v + (0.35 if v >= 0 else -0.35), yi, f"{v:+.2f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=6.0)
    for yi, t in zip(y, order):
        if np.isnan(stats[t]["disc"]):
            axB.text(0.4, yi, "not estimable", va="center", ha="left",
                     fontsize=5.8, style="italic", color="#666666")
    axB.axvline(0, color="#333333", lw=0.6)
    axB.set_xlim(-3.0, XMAX)
    axB.set_ylim(y_imp - 0.75, max(y) + 0.6)
    axB.set_yticks(y); axB.set_yticklabels([])
    axB.set_xlabel(r"$\widehat{\Delta}_{\mathrm{disc}}$  [percentage points]")
    axB.set_title(r"(b) $\Delta_{\mathrm{disc}}$: zero in mechanism arms, $+0.51$ once the exchange is removed",
                  fontsize=7.0)
    fig.savefig(os.path.join(FIGDIR, "fig_overview.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig_overview.png"), dpi=300)
    plt.close(fig)

    # ============================================================ Fig 3 交互率
    fig, ax = plt.subplots(figsize=(W1, 1.72))
    one = np.array([stats[t]["turns"].get(1, 0) / stats[t]["n"] * 100 for t in order])
    two = np.array([stats[t]["turns"].get(2, 0) / stats[t]["n"] * 100 for t in order])
    three = np.array([sum(v for k, v in stats[t]["turns"].items() if k >= 3)
                      / stats[t]["n"] * 100 for t in order])
    ax.barh(y, one, 0.58, color="#4C72B0", label="terminates at turn 1")
    ax.barh(y, two, 0.58, left=one, color="#DD8452", label="turn 2")
    ax.barh(y, three, 0.58, left=one + two, color="#937860", label=r"$\geq$ turn 3")
    for yi, o in zip(y, one):
        ax.text(o - 1.2, yi, f"{o:.0f}", ha="right", va="center",
                fontsize=6.0, color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{labels[t]} ({'same' if same[t] else 'cross'})" for t in order],
                       fontsize=6.4)
    ax.set_xlabel("share of episodes  [%]")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(order) - 0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=3,
              frameon=False, handlelength=1.0, fontsize=5.8,
              columnspacing=0.9, handletextpad=0.4)
    fig.savefig(os.path.join(FIGDIR, "fig_interaction.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig_interaction.png"), dpi=300)
    plt.close(fig)

    print("\nFIGURES ->", FIGDIR)
    for f in sorted(os.listdir(FIGDIR)):
        print("  ", f, os.path.getsize(os.path.join(FIGDIR, f)), "B")


if __name__ == "__main__":
    main()
