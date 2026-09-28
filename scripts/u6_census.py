# -*- coding: utf-8 -*-
"""U6: construction-artifact census.

对被审论文释放的全部公开工件（3 数据集 × 2 顾问对）逐条统计初始答案的
正确性平衡与分歧结构。判据与稿件 §5 完全一致：
  - agree:    c1_answer == c2_answer
  - one-correct-disagree: 答案不同且恰好一方正确（必然分歧 + 一对一错）
  - both-right / both-wrong: 答案相同
  - other:    答案不同但双方全错或数据异常
纯本地分析，零 API 成本。
"""
import json
from pathlib import Path

DATA = Path("D:/cdr_research/code/data")
OUT_TXT = Path("D:/cdr_research/out/_u6_census.txt")

PAIRS = [("glm", "kimi"), ("gpt_4o", "claude")]
DATASETS = ["GPQA", "SuperGPQA", "HLE"]


def load(ds, stem):
    p = DATA / ds / f"{stem}.json"
    return {e["index"]: e for e in json.loads(p.read_text(encoding="utf-8"))}


def is_correct(e):
    a = str(e.get("model_answer", "")).strip().upper()
    c = str(e.get("correct_choice") or e.get("Correct Answer") or "").strip().upper()
    return a == c


rows = []
for ds in DATASETS:
    for s1, s2 in PAIRS:
        d1, d2 = load(ds, s1), load(ds, s2)
        idx = sorted(set(d1) & set(d2))
        stat = dict(n=len(idx), agree=0, one_correct_disagree=0,
                    both_right=0, both_wrong_disagree=0, other=0)
        for i in idx:
            a, b = d1[i], d2[i]
            aa = str(a.get("model_answer", "")).strip().upper()
            bb = str(b.get("model_answer", "")).strip().upper()
            ca, cb = is_correct(a), is_correct(b)
            if aa == bb:
                if ca and cb:
                    stat["both_right"] += 1
                elif not ca and not cb:
                    stat["both_wrong_disagree"] += 1
                else:
                    stat["agree"] += 1
            else:
                if ca != cb:            # 恰好一方正确
                    stat["one_correct_disagree"] += 1
                elif not ca and not cb:
                    stat["both_wrong_disagree"] += 1
                else:
                    stat["both_right"] += 1
        rows.append((ds, f"{s1}×{s2}", stat))

lines = []
lines.append("=" * 78)
lines.append("U6 CONSTRUCTION-ARTIFACT CENSUS (released artifacts, %d dataset×pair cells)" % len(rows))
lines.append("=" * 78)
lines.append(f"{'dataset':10s} {'pair':12s} {'n':>5s} {'1c-disagree':>12s} {'both-right':>11s} "
             f"{'both-wrong':>11s} {'agree':>7s} {'%forced':>8s}")
tot_n = tot_f = 0
for ds, pair, s in rows:
    forced = s["one_correct_disagree"]
    pct = 100.0 * forced / s["n"] if s["n"] else 0.0
    lines.append(f"{ds:10s} {pair:12s} {s['n']:5d} {forced:12d} {s['both_right']:11d} "
                 f"{s['both_wrong_disagree']:11d} {s['agree']:7d} {pct:7.1f}%")
    tot_n += s["n"]; tot_f += forced
lines.append("-" * 78)
lines.append(f"TOTAL n={tot_n}  one-correct-disagree={tot_f} ({100.0*tot_f/tot_n:.1f}%)")
txt = "\n".join(lines)
OUT_TXT.write_text(txt, encoding="utf-8")
print(txt)
