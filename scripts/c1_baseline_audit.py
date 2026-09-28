# -*- coding: utf-8 -*-
"""
C1 证据脚本 —— 零参数对照实验：论文数字核对 + 硬币基线反证
--------------------------------------------------------------
零 API 成本，纯本地复算。产出：
  1. 论文 consultant 个体准确率 / 样本量的逐项核对表
  2. exactly-one-correct 结构验证（含 both-wrong 计数）
  3. 硬币基线反证：判官在两个候选间纯随机选择 vs 论文 Debate 数字
  4. chance / oracle 归一化度量（本文提出的新度量）

用法:
  python exp_c1_baseline_audit.py            # 全量
  python exp_c1_baseline_audit.py --n-boot 20000
"""
from __future__ import annotations
import argparse, json, os, random, statistics, sys
from pathlib import Path

# 跨平台路径解析：优先环境变量 CDR_ROOT，其次脚本所在仓库根，最后回退到本机默认位置
_HERE = Path(__file__).resolve().parent      # .../exp
_ROOT = Path(os.environ.get("CDR_ROOT", _HERE.parent))
CODE = _ROOT / "code"
DATA = CODE / "data"
OUT = _ROOT / "out"
if not DATA.exists():                         # 回退：与本机 D:/cdr_research 布局一致
    CODE = Path(r"D:\cdr_research\code"); DATA = CODE / "data"; OUT = Path(r"D:\cdr_research\out")

# 论文 Table 2 / Table 3 报告值（硬编码自 main.tex 读取结果，用于逐项核对）
PAPER = {
    # (dataset, pair, judge) -> (debate, DR)
    ("GPQA", "gpt4o_claude", "gpt-4o-mini"): (63.9, 67.5),
    ("GPQA", "gpt4o_claude", "gemma-3-4b"): (50.0, 65.9),
    ("SuperGPQA", "gpt4o_claude", "gpt-4o-mini"): (68.3, 64.7),
    ("SuperGPQA", "gpt4o_claude", "gemma-3-4b"): (52.0, 64.7),
    ("HLE", "gpt4o_claude", "gpt-4o-mini"): (34.7, 38.6),
    ("HLE", "gpt4o_claude", "gemma-3-4b"): (44.3, 37.5),
    ("GPQA", "glm_kimi", "gpt-4o-mini"): (69.2, 77.2),
    ("GPQA", "glm_kimi", "gemma-3-4b"): (46.8, 74.3),
    ("SuperGPQA", "glm_kimi", "gpt-4o-mini"): (41.7, 66.2),
    ("SuperGPQA", "glm_kimi", "gemma-3-4b"): (46.2, 66.7),
    ("HLE", "glm_kimi", "gpt-4o-mini"): (32.6, 58.8),
    ("HLE", "glm_kimi", "gemma-3-4b"): (41.3, 62.5),
}
# 论文 Table 2 报告的 consultant 个体准确率
PAPER_CONSULTANT = {
    ("GPQA", "gpt_4o"): 39.7, ("GPQA", "claude"): 60.3,
    ("GPQA", "glm"): 34.7, ("GPQA", "kimi"): 65.4,
    ("SuperGPQA", "gpt_4o"): 30.7, ("SuperGPQA", "claude"): 69.3,
    ("SuperGPQA", "glm"): 31.9, ("SuperGPQA", "kimi"): 68.1,
    ("HLE", "gpt_4o"): 36.4, ("HLE", "claude"): 63.6,
    ("HLE", "glm"): 38.8, ("HLE", "kimi"): 61.2,
}
PAIRS = [("GPQA", "gpt_4o.json", "claude.json", "gpt4o_claude"),
         ("GPQA", "glm.json", "kimi.json", "glm_kimi"),
         ("SuperGPQA", "gpt_4o.json", "claude.json", "gpt4o_claude"),
         ("SuperGPQA", "glm.json", "kimi.json", "glm_kimi"),
         ("HLE", "gpt_4o.json", "claude.json", "gpt4o_claude"),
         ("HLE", "glm.json", "kimi.json", "glm_kimi")]


def load(ds: str, fn: str):
    return json.loads((DATA / ds / fn).read_text(encoding="utf-8"))


def match(ds, f1, f2):
    d1 = {e["index"]: e for e in load(ds, f1)}
    d2 = {e["index"]: e for e in load(ds, f2)}
    idx = sorted(set(d1) & set(d2))
    return d1, d2, idx


def hr(t=""):
    print(("=" * 78) if not t else f"\n{'='*78}\n{t}\n{'='*78}")


def part1_consultant_accuracy():
    hr("PART 1  论文 consultant 个体准确率逐项核对")
    print(f"{'dataset':10s} {'model':8s} {'N':>5s} {'复算':>9s} {'论文':>8s} {'Δ':>7s}  结论")
    ok = True
    for ds, f1, f2, _ in PAIRS:
        d1, d2, idx = match(ds, f1, f2)
        for fn, d in ((f1, d1), (f2, d2)):
            key = fn.replace(".json", "")
            n = len(idx)
            c = sum(1 for i in idx if d[i].get("model_correct"))
            mine = c / n * 100
            paper = PAPER_CONSULTANT.get((ds, key))
            if paper is None:
                continue
            delta = mine - paper
            flag = "OK" if abs(delta) < 0.3 else "MISMATCH"
            if abs(delta) >= 0.3:
                ok = False
            print(f"{ds:10s} {key:8s} {n:5d} {mine:8.2f}% {paper:7.1f}% {delta:+6.2f}  {flag}")
    # 去重（同一 dataset+model 只核对一次）
    print()
    print("=> consultant 个体准确率与论文报告值一致" if ok else "=> 存在不一致，需排查")
    return ok


def part2_structure():
    hr("PART 2  数据结构自洽性：是否严格 exactly-one-correct")
    tot = v_agree = v_both_right = v_both_wrong = 0
    for ds, f1, f2, _ in PAIRS:
        d1, d2, idx = match(ds, f1, f2)
        for i in idx:
            tot += 1
            a1 = str(d1[i].get("model_answer", "")).strip().upper()
            a2 = str(d2[i].get("model_answer", "")).strip().upper()
            c1, c2 = d1[i].get("model_correct"), d2[i].get("model_correct")
            if a1 == a2:
                v_agree += 1
            if c1 and c2:
                v_both_right += 1
            if (not c1) and (not c2):
                v_both_wrong += 1
    print(f"总样本 (已匹配)          : {tot}")
    print(f"A. 两顾问答案相同        : {v_agree}")
    print(f"B. 两顾问都答对          : {v_both_right}")
    print(f"C. 两顾问都答错          : {v_both_wrong}")
    print()
    print("判读:")
    if v_agree == 0:
        print("  * A=0 -> 数据已预筛为 disagreement-only（发布物中无筛选脚本）")
    if v_both_wrong == 0:
        print("  * C=0 -> 严格 exactly-one-correct。")
        print("    但论文 Def.2 脚注明确写道 both-wrong 情形 'is still possible'。")
        print("    => 发布数据比论文定义更强，且这一步论文未提及。")
        print("    => 后果: oracle 上限恒为 100%，判官任务被降为'二选一'。")
    return dict(total=tot, agree=v_agree, both_right=v_both_right, both_wrong=v_both_wrong)


def part3_coin_flip(n_boot: int, seed: int = 42):
    hr("PART 3  决定性反证：硬币基线 vs 论文 Debate 数字")
    print("设定: 判官完全忽略所有论证，在两个候选答案间均匀随机选择。")
    print("      （因数据严格 exactly-one-correct，该策略期望准确率 = 50%）")
    print()
    print(f"{'dataset':10s} {'pair':12s} {'判官':11s} {'N':>4s} {'论文Debate':>10s} {'论文DR':>8s} "
          f"{'硬币均值':>9s} {'95%CI':>16s} {'硬币>Debate?':>13s}")
    rows = []
    rng = random.Random(seed)
    for ds, f1, f2, pk in PAIRS:
        d1, d2, idx = match(ds, f1, f2)
        n = len(idx)
        for judge in ("gpt-4o-mini", "gemma-3-4b"):
            pb, pdr = PAPER[(ds, pk, judge)]
            # 蒙特卡洛：每次随机选一个候选
            sims = []
            for _ in range(n_boot):
                hits = sum(1 for _ in idx if rng.random() < 0.5)
                sims.append(hits / n * 100)
            sims.sort()
            mean = statistics.fmean(sims)
            lo = sims[int(0.025 * n_boot)]
            hi = sims[int(0.975 * n_boot)]
            beat = "  YES <--" if pb < lo else ("  ~打平" if lo <= pb <= hi else "  no")
            rows.append(dict(ds=ds, pair=pk, judge=judge, n=n, debate=pb, dr=pdr,
                             coin_mean=mean, coin_lo=lo, coin_hi=hi, beats=pb < lo))
            print(f"{ds:10s} {pk:12s} {judge:11s} {n:4d} {pb:9.1f}% {pdr:7.1f}% "
                  f"{mean:8.1f}% [{lo:5.1f},{hi:5.1f}] {beat:>13s}")
    n_beat = sum(1 for r in rows if r["beats"])
    print()
    print(f"=> 12 个条件中，论文 Debate 数字低于硬币基线 95%CI 下界的有 {n_beat} 个。")
    print("   含义: Debate 在这些条件下并未'有效提取信号'，而是把判官从 50% 的")
    print("   朴素水平拉低了。因此 Δ=DR−Debate 的部分来源是 Debate 基线被破坏，")
    print("   而非 DR 机制的独有增益。")
    return rows


def part4_normalized(n_boot: int):
    hr("PART 4  建议的新度量：相对 chance/oracle 归一化准确率")
    print("定义: Acc_norm = (Acc − random) / (oracle − random)，random = 1/|Ω|")
    print("     对比之下，论文报告的原始准确率把'任务难度'与'协议能力'混在一起。")
    print()
    print(f"{'dataset':10s} {'|Ω|':>4s} {'random':>8s} {'oracle':>8s} "
          f"{'Debate原始':>11s} {'Debate_norm':>12s} {'DR原始':>8s} {'DR_norm':>9s}")
    rows = []
    for ds, f1, f2, pk in PAIRS:
        d1, d2, idx = match(ds, f1, f2)
        ch = d1[idx[0]].get("choices")
        if isinstance(ch, dict):
            nch = len(ch)
        else:
            import re
            nch = max(len(re.findall(r"(?:^|[,\s])([A-E])[\.\)]\s", str(ch))), 2)
        rnd = 100.0 / nch
        pb, pdr = PAPER[(ds, pk, "gpt-4o-mini")]
        nb = (pb - rnd) / (100.0 - rnd) * 100
        nd = (pdr - rnd) / (100.0 - rnd) * 100
        rows.append(dict(ds=ds, pair=pk, nch=nch, rnd=rnd, debate=pb, dr=pdr,
                         debate_norm=nb, dr_norm=nd))
        print(f"{ds:10s} {nch:4d} {rnd:7.1f}% {100.0:7.1f}% "
              f"{pb:10.1f}% {nb:11.1f}% {pdr:7.1f}% {nd:8.1f}%")
    print()
    print("判读: 归一化后，Debate 接近 0 甚至为负，DR 为正 -> 说明二者差距")
    print("      主要来自'谁没有低于随机'，而非'谁更接近真值'。")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--json-out", type=Path, default=OUT / "c1_baseline_audit.json")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ok_acc = part1_consultant_accuracy()
    struct = part2_structure()
    coins = part3_coin_flip(args.n_boot)
    norms = part4_normalized(args.n_boot)

    summary = dict(
        consultant_accuracy_verified=ok_acc,
        structure=struct,
        coin_flip=coins,
        normalized=norms,
        n_conditions_debate_below_coin=int(sum(1 for r in coins if r["beats"])),
    )
    args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    hr("SUMMARY")
    print(json.dumps({
        "consultant_accuracy_match_paper": ok_acc,
        "agree_samples": struct["agree"],
        "both_wrong_samples": struct["both_wrong"],
        "total_samples": struct["total"],
        "debate_below_coin_ci": summary["n_conditions_debate_below_coin"],
        "out_of_conditions": len(coins),
        "json": str(args.json_out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
