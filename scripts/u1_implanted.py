# -*- coding: utf-8 -*-
"""U1: implanted-consensus arm (judge-only).

设计（审稿意见 U1 / 第三轮 CCFA idea 审查）：
  终端共识由实验者**植入**，只跑判官。每个 GPQA 题目构造两个条件：
    C 条件: 两顾问"一致"给出正确答案（推理 = 工件中正确顾问的推理）
    W 条件: 两顾问"一致"给出错误答案（推理 = 工件中错误顾问的推理）
  判官输入 = JUDGE_BODY（placebo 视角：无 crux/conflict 块），agree 状态。
  ⇒ 共识正确性是唯一受控变量，把"判官采纳不依赖正确性"从观察升级为因果判定。

度量（与 _r2_stats.py 完全一致）:
  adopted := judge_answer == 植入共识答案
  Δ_disc  := P(adopt|C) - P(adopt|W)

运行:
  python3 u1_implanted.py --limit 2        # 冒烟
  python3 u1_implanted.py --limit 80       # 全量（与主实验同题集）
输出:
  out/u1_implanted.jsonl   逐条追加（断点续跑）
  out/u1_implanted.json    结束时写汇总
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proto_run as pr   # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "out"
JSONL = OUT / "u1_implanted.jsonl"
SUMMARY = OUT / "u1_implanted.json"

JUDGES = [("glm-4.5-air",), ("glm-4-flash",)]   # 双判官（主判官 + 弱判官对照）
# 注：2026-09-24 12:59 实测阿里 MaaS 全端账号级拒绝（免费额度耗尽），
#     qwen-flash 不可用，第二判官改 glm-4-flash（智谱，实测 OK）。

# 无工件推理时的对称兜底模板（C/W 两条件同一文本，仅答案标签不同 —— 对称性保持）
TEMPLATE = [
    "Restate the question and identify the quantity or relation asked for.",
    "Eliminate options inconsistent with the stated conditions.",
    "Cross-check the remaining option against the given constraints.",
    "The analysis converges on {ans}.",
]

_print_lock = threading.Lock()


def log(msg: str):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_case_pair(case: dict):
    """返回 (C, W) 两个条件各自的 (answer, reasoning list)。
    若任一侧工件推理为空，两条件都退回对称模板（保持对称）。"""
    c1a, c2a = case["c1_answer"], case["c2_answer"]
    c1r = [str(x) for x in (case["c1_reasoning"] or []) if str(x).strip()]
    c2r = [str(x) for x in (case["c2_reasoning"] or []) if str(x).strip()]
    correct = case["correct"]
    if c1a == correct:
        ans_c, rea_c = c1a, c1r
        ans_w, rea_w = c2a, c2r
    elif c2a == correct:
        ans_c, rea_c = c2a, c2r
        ans_w, rea_w = c1a, c1r
    else:
        return None    # 工件异常（无一正确）：跳过该题并记录
    if not rea_c or not rea_w:
        rea_c = [s.format(ans=ans_c) for s in TEMPLATE]
        rea_w = [s.format(ans=ans_w) for s in TEMPLATE]
    return (ans_c, rea_c), (ans_w, rea_w)


def judge_prompt(case: dict, ans: str, rea) -> str:
    return pr.JUDGE_BODY.format(
        a1=ans, a2=ans,
        r1="\n".join(f"{i+1}. {x}" for i, x in enumerate(rea)),
        r2="\n".join(f"{i+1}. {x}" for i, x in enumerate(rea)),
        turns=1, maxturns=pr.BUDGET["max_turns"], agree="agree",
        question=case["question"], choices=case["choices"],
        conflict_block="")


def run_one(cl, case: dict, cond: str, ans: str, rea) -> dict:
    jp = judge_prompt(case, ans, rea)
    raw = cl.chat(case["judge_model"], [{"role": "user", "content": jp}], role="judge")
    p = pr.jparse(raw)
    selected = str(p.get("selected_answer") or "").strip().upper()
    return dict(
        index=case["index"], condition=cond, judge=case["judge_model"],
        planted_answer=ans, correct=case["correct"],
        judge_answer=selected,
        adopted=(selected == ans),
        judge_winner=p.get("winner"), judge_exit=p.get("exit_mode"),
        judge_correct=(selected == case["correct"]),
        judge_reason=str(p.get("reason") or "")[:400],
        template_fallback=bool(case["_template_fallback"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = pr.load_env(pr.CODE / ".env")
    KEYS = {"zhipu": cfg.get("ZHIPU_API_KEY", ""),
            "alibaba": cfg.get("ALIBABA_API_KEY", ""),
            "openrouter": cfg.get("OPENROUTER_API_KEY", "")}
    if not any(KEYS.values()) and not args.dry_run:
        print("[FATAL] code/.env 缺少 API key"); return 1
    cl = pr.Client("", pr.CACHE, dry_run=args.dry_run, provider="zhipu",
                   keys=KEYS, think="auto")

    cases = pr.load_cases("GPQA", "glm46_glm45", limit=args.limit, offset=args.offset)
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["index"], r["condition"], r["judge"]))
            except Exception:
                pass
    log(f"cases={len(cases)} done={len(done)}")

    jobs = []
    skipped = []
    for c in cases:
        pair = build_case_pair(c)
        if pair is None:
            skipped.append(c["index"]); continue
        (ans_c, rea_c), (ans_w, rea_w) = pair
        c["_template_fallback"] = not ((rea_c and rea_w))
        for (jmodel,) in JUDGES:
            cc = dict(c); cc["judge_model"] = jmodel
            if (c["index"], "C", jmodel) not in done:
                jobs.append((cc, "C", ans_c, rea_c))
            if (c["index"], "W", jmodel) not in done:
                jobs.append((cc, "W", ans_w, rea_w))
    log(f"jobs={len(jobs)} skipped_cases={skipped}")

    t0 = time.time()
    n_ok = n_err = 0
    with JSONL.open("a", encoding="utf-8") as fh, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, cl, c, cond, ans, rea): (c["index"], cond, c["judge_model"])
                for (c, cond, ans, rea) in jobs}
        for fu in as_completed(futs):
            key = futs[fu]
            try:
                r = fu.result()
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                n_ok += 1
                if n_ok % 20 == 0:
                    el = time.time() - t0
                    log(f"progress {n_ok}/{len(jobs)} ok={n_ok} err={n_err} elapsed={el:.0f}s")
            except Exception as e:
                n_err += 1
                log(f"ERR {key}: {type(e).__name__}: {str(e)[:150]}")

    # ---- 汇总 ----
    results = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = dict(job="u1_implanted", limit=args.limit, offset=args.offset,
                   n_new_ok=n_ok, n_new_err=n_err, elapsed_s=round(time.time() - t0, 1),
                   total_records=len(results))
    by = {}
    for r in results:
        k = (r["condition"], r["judge"])
        b = by.setdefault(k, dict(n=0, adopt=0, jcorrect=0))
        b["n"] += 1
        b["adopt"] += int(r["adopted"])
        b["jcorrect"] += int(r["judge_correct"])
    disc = {}
    for (cond, j), b in by.items():
        pass
    judges = sorted({k[1] for k in by})
    for j in judges:
        c, w = by.get(("C", j)), by.get(("W", j))
        if c and w and c["n"] and w["n"]:
            p1, p0 = c["adopt"] / c["n"], w["adopt"] / w["n"]
            disc[j] = dict(p_C=round(p1, 4), p_W=round(p0, 4),
                           delta_disc=round(p1 - p0, 4),
                           n_C=c["n"], n_W=w["n"],
                           judge_correct_C=c["jcorrect"], judge_correct_W=w["jcorrect"])
    summary["by_condition"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(by.items())}
    summary["delta_disc"] = disc
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"DONE ok={n_ok} err={n_err} -> {SUMMARY.name}")
    print(json.dumps(summary["delta_disc"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
