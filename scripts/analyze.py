# -*- coding: utf-8 -*-
"""
统计分析与假设检验 —— 检验报告 §6.5 的 H1-H4
================================================
输入: out/*.json （由 proto_run.py 产出，每个协议一个文件）
输出: 配对检验表 + 多种子方差 + 鉴别力（Discriminability）+ 安慰剂归因分解

用法:
  python exp/analyze.py                                  # 扫描 out/ 下全部
  python exp/analyze.py --glob 'placebo_*.json' --baseline dr_GPQA_gpt4o_claude_*.json
"""
from __future__ import annotations
import argparse, glob, json, math, os, random
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("CDR_ROOT", _HERE.parent))
OUT = ROOT / "out"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def mcnemar(a: dict, b: dict):
    """配对 McNemar 检验：两协议在共同 index 上的判官正误配对"""
    ia = {r["index"]: r for r in a["results"] if "judge_correct" in r}
    ib = {r["index"]: r for r in b["results"] if "judge_correct" in r}
    common = sorted(set(ia) & set(ib))
    n01 = sum(1 for i in common if (not ia[i]["judge_correct"]) and ib[i]["judge_correct"])
    n10 = sum(1 for i in common if ia[i]["judge_correct"] and (not ib[i]["judge_correct"]))
    n = n01 + n10
    # 准确率必须在任何分支前算好——退化数据（n==0）时下游仍要读 acc_a/acc_b
    if common:
        acc_a = sum(1 for i in common if ia[i]["judge_correct"]) / len(common) * 100
        acc_b = sum(1 for i in common if ib[i]["judge_correct"]) / len(common) * 100
    else:
        acc_a = acc_b = 0.0
    if n == 0:
        # 两协议在共同样本上判别完全一致：McNemar 无信息，p=1（不能拒绝任何假设）
        return dict(n=len(common), acc_a=acc_a, acc_b=acc_b, delta=acc_b - acc_a,
                    b=n01, c=n10, p=1.0, chi2=0.0, degenerate=True)
    chi2 = (abs(n01 - n10) - 1) ** 2 / n          # 连续性校正
    # 卡方(1) 的双尾 p 值，用 erfc 近似
    p = math.erfc(math.sqrt(chi2 / 2))
    return dict(n=len(common), acc_a=acc_a, acc_b=acc_b, delta=acc_b - acc_a,
                b=n01, c=n10, chi2=chi2, p=p, degenerate=False)


def bootstrap_ci(flags, n=10000, seed=42, alpha=0.05):
    if not flags:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    N = len(flags)
    sims = sorted(sum(rng.choice(flags) for _ in range(N)) / N * 100 for _ in range(n))
    return (sum(flags) / N * 100, sims[int(alpha / 2 * n)], sims[int((1 - alpha / 2) * n)])


def discriminability(rows):
    """
    ★ 报告 §6.4 视角2 提出的新度量（可从已有数据直接估计）
      Disc = P(判官判定共识可信 | 共识正确) − P(判官判定共识可信 | 共识错误)
    判官"判定可信" = 采纳共识（judge_answer == 共识答案，且 exit_mode == agree）
    """
    tc = [r for r in rows if r.get("consensus") and r.get("judge_exit") == "agree"
          and r.get("judge_answer") == r.get("c1_answer")]
    at = [r for r in rows if r.get("consensus") and r.get("judge_exit") == "agree"
          and r.get("judge_answer") != r.get("c1_answer")]
    # 用 exit_mode 分：TC = 共识且正确；AT = 共识且错误
    tc_rows = [r for r in rows if r.get("exit_mode") == "TC"]
    at_rows = [r for r in rows if r.get("exit_mode") == "AT"]
    def adopt(rs):
        if not rs: return None
        return sum(1 for r in rs if r.get("judge_answer") == r.get("c1_answer")) / len(rs)
    a_tc, a_at = adopt(tc_rows), adopt(at_rows)
    disc = None if (a_tc is None or a_at is None) else a_tc - a_at
    return dict(n_TC=len(tc_rows), n_AT=len(at_rows), adopt_TC=a_tc, adopt_AT=a_at, discriminability=disc)


def exit_action_profile(rows):
    ex = Counter(r.get("exit_mode") for r in rows if r.get("exit_mode"))
    ac = Counter()
    for r in rows:
        for k in ("action_1", "action_2"):
            if r.get(k): ac[r[k]] += 1
    tot = sum(ac.values()) or 1
    return dict(exit_modes={k: round(v / max(sum(ex.values()), 1) * 100, 1) for k, v in ex.items()},
                actions={k: round(v / tot * 100, 1) for k, v in ac.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--baseline", default=None, help="作为对照的协议文件 glob")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--json", default=None, help="将分析结果另存为 JSON 供报告引用")
    args = ap.parse_args()

    files = sorted(glob.glob(str(OUT / args.glob)))
    runs = [load(f) for f in files]
    runs = [r for r in runs if "results" in r]
    if not runs:
        print(f"out/ 下没有可分析的结果（glob={args.glob}）。先跑 proto_run.py。")
        return

    print("=" * 78); print("A. 各协议总览"); print("=" * 78)
    print(f"{'protocol':12s} {'dataset':10s} {'pair':14s} {'judge':22s} {'N':>4s} {'acc':>7s} {'95%CI':>16s}")
    for r in runs:
        ok = [x for x in r["results"] if "judge_correct" in x]
        m, lo, hi = bootstrap_ci([x["judge_correct"] for x in ok], args.boot)
        print(f"{r['protocol']:12s} {r['dataset']:10s} {r['pair']:14s} {r['judge']:22s} "
              f"{len(ok):4d} {m:6.1f}% [{lo:5.1f},{hi:5.1f}]")

    print()
    print("=" * 78); print("B. ★ 关键对照：安慰剂归因（H1）"); print("=" * 78)
    by = defaultdict(dict)
    for r in runs:
        by[(r["dataset"], r["pair"], r["judge"])][r["protocol"]] = r
    for k, d in sorted(by.items()):
        if "dr" in d and "placebo" in d:
            res = mcnemar(d["placebo"], d["dr"])
            if res.get("degenerate"):
                print(f"{k[0]}/{k[1]}/{k[2]}: DR={res['acc_b']:.1f}% placebo={res['acc_a']:.1f}% "
                      f"-> 两协议判别完全一致（n=0），无法做配对检验")
            else:
                verdict = "crux 机制无贡献（H1 成立）" if res["p"] > 0.05 else "crux 机制有贡献"
                print(f"{k[0]}/{k[1]}/{k[2]}: DR={res['acc_b']:.1f}% placebo={res['acc_a']:.1f}% "
                      f"Δ={res['delta']:+.1f} p={res['p']:.4f}  -> {verdict}")
        if "dr" in d and "dr_nocrux" in d:
            res = mcnemar(d["dr_nocrux"], d["dr"])
            tag = "（退化：完全一致）" if res.get("degenerate") else f"p={res['p']:.4f}"
            print(f"   消融 dr_nocrux: DR={res['acc_b']:.1f}% nocrux={res['acc_a']:.1f}% "
                  f"Δ={res['delta']:+.1f} {tag}")

    print()
    print("=" * 78); print("C. DR vs Debate（协议对称版，H2 参照）"); print("=" * 78)
    for k, d in sorted(by.items()):
        if "dr" in d and "debate" in d:
            res = mcnemar(d["debate"], d["dr"])
            if res.get("degenerate"):
                print(f"{k[0]}/{k[1]}/{k[2]}: Debate={res['acc_a']:.1f}% DR={res['acc_b']:.1f}% "
                      f"-> 判别完全一致（n=0），无法做配对检验")
            else:
                print(f"{k[0]}/{k[1]}/{k[2]}: Debate={res['acc_a']:.1f}% DR={res['acc_b']:.1f}% "
                      f"Δ={res['delta']:+.1f} McNemar p={res['p']:.4f} (n={res['n']}, b={res['b']}, c={res['c']})")

    print()
    print("=" * 78); print("D. ★ 可鉴别性 Discriminability（H3，报告 §6.4 视角2）"); print("=" * 78)
    print("Disc = P(采纳共识 | 共识正确TC) − P(采纳共识 | 共识错误AT)；论文从未报告此量")
    for k, d in sorted(by.items()):
        for proto in ("dr", "placebo", "dr_nocrux"):
            if proto in d:
                ok = [x for x in d[proto]["results"] if "judge_correct" in x]
                r = discriminability(ok)
                disc = r["discriminability"]
                flag = ""
                if disc is not None:
                    flag = " <- 判官无法区分共识质量" if abs(disc) < 0.1 else " <- 有鉴别力"
                print(f"{k[0]}/{k[1]}/{k[2]} [{proto}]: TC={r['n_TC']} AT={r['n_AT']} "
                      f"采纳率TC={r['adopt_TC']} 采纳率AT={r['adopt_AT']} Disc={disc}{flag}")

    print()
    print("=" * 78); print("E. 退出模式与行为分布（对齐论文 Def.3/4）"); print("=" * 78)
    for k, d in sorted(by.items()):
        for proto, run in sorted(d.items()):
            ok = [x for x in run["results"] if "judge_correct" in x]
            p = exit_action_profile(ok)
            print(f"{k[0]}/{k[1]}/{k[2]} [{proto:10s}] exit={p['exit_modes']}")
            print(f"{'':42s} actions={p['actions']}")

    print()
    print("=" * 78); print("F. 归一化准确率（H4 参照，报告 §6.5 新度量）"); print("=" * 78)
    for r in runs:
        ok = [x for x in r["results"] if "judge_correct" in x]
        nch = 4 if r["dataset"] == "GPQA" else 5
        rnd = 100.0 / nch
        acc = sum(1 for x in ok if x["judge_correct"]) / len(ok) * 100 if ok else 0
        norm = (acc - rnd) / (100 - rnd) * 100
        print(f"{r['protocol']:12s} {r['dataset']:10s} {r['pair']:14s} "
              f"acc={acc:5.1f}% random={rnd:4.1f}% -> norm={norm:5.1f}%")

    print()
    print("=" * 78); print("G. ★ crux 机制是否真的在运行（报告 §6.3 痛点 4）"); print("=" * 78)
    print("若 DR 的 crux 占比与 placebo 无差异，则『识别 crux』这一机制是装饰性的")
    print(f"{'protocol':12s} {'N':>4s} {'冲突/样本':>9s} {'出现crux%':>10s} {'改答案次数/样本':>14s} {'平均轮数':>8s}")
    for r in runs:
        ok = [x for x in r["results"] if "judge_correct" in x]
        if not ok:
            continue
        nc = sum(x.get("n_conflicts_1", 0) for x in ok) / len(ok)
        has = sum(1 for x in ok if x.get("n_conflicts_1", 0) > 0) / len(ok) * 100
        ch = sum(x.get("n_ans_changes_1", 0) for x in ok) / len(ok)
        tn = sum(x.get("turns", 0) for x in ok) / len(ok)
        print(f"{r['protocol']:12s} {len(ok):4d} {nc:9.2f} {has:9.0f}% {ch:14.2f} {tn:8.2f}")

    print()
    print("=" * 78); print("H. 终止原因分布（判断多轮机制是否为『空转』）"); print("=" * 78)
    for r in runs:
        ok = [x for x in r["results"] if "judge_correct" in x]
        if not ok:
            continue
        sr = Counter(x.get("stop_reason") for x in ok if x.get("stop_reason"))
        print(f"{r['protocol']:12s} {dict(sr)}")


if __name__ == "__main__":
    main()
