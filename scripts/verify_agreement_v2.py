#!/usr/bin/env python3
"""Verify human annotation agreement against machine consensus and channels.

After two annotators fill annotator1.csv and annotator2.csv (from annotate_form.csv),
run:

    python verify_agreement_v2.py --a1 annotator1.csv --a2 annotator2.csv --key answer_key.jsonl

Outputs paper-ready numbers:
  1. Human-human agreement + Cohen's kappa
  2. Human-majority vs MACHINE CONSENSUS agreement + kappa
  3. Human-majority vs CONFLICT channel / REBUTTAL channel agreement   <- core check
  4. Breakdown by epistemic type
"""
import argparse
import csv
import json
from collections import Counter


def norm_verdict(v):
    v = (v or "").strip().upper()
    if v in ("CORRECT", "C"): return "CORRECT"
    if v in ("INCORRECT", "I"): return "INCORRECT"
    return None


def load_csv(path):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vid = (row.get("item_id") or "").strip()
            v = norm_verdict(row.get("verdict(CORRECT/INCORRECT)") or row.get("verdict"))
            if vid and v:
                out[vid] = v
    return out


def load_key(path):
    key = {}
    with open(path, "rb") as f:
        for line in f:
            r = json.loads(line)
            key[r["item_id"]] = r
    return key


def cohen_kappa(pairs):
    """pairs: list of (label_a, label_b) with labels in {CORRECT, INCORRECT}."""
    n = len(pairs)
    if n == 0: return float("nan")
    agree = sum(1 for a, b in pairs if a == b) / n
    pa = Counter(a for a, _ in pairs)
    pb = Counter(b for _, b in pairs)
    pe = sum(pa[k] * pb[k] for k in ("CORRECT", "INCORRECT")) / (n * n)
    if pe == 1.0: return 1.0
    return (agree - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1", required=True)
    ap.add_argument("--a2", required=True)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()

    a1 = load_csv(args.a1)
    a2 = load_csv(args.a2)
    key = load_key(args.key)

    common = sorted(set(a1) & set(a2) & set(key))
    print(f"items annotated by both and in key: {len(common)} / {len(key)}")

    # 1) human-human
    hh = [(a1[i], a2[i]) for i in common]
    hh_agree = sum(1 for x, y in hh if x == y) / len(hh)
    kappa_hh = cohen_kappa(hh)
    print(f"\n[1] Human-human agreement: {hh_agree*100:.1f}%  kappa={kappa_hh:.3f}")

    # human majority (ties -> keep both, report tie rate)
    majority, ties = {}, 0
    for i in common:
        if a1[i] == a2[i]:
            majority[i] = a1[i]
        else:
            ties += 1
            majority[i] = None  # unresolved; excluded from vs-machine comparison
    print(f"    disagreements (to arbitrate): {ties}/{len(common)}")

    resolved = [i for i in common if majority[i]]
    print(f"    resolved items (after majority): {len(resolved)}")

    # 2) human majority vs machine consensus
    hm_pairs, agree = [], 0
    for i in resolved:
        gm = key[i]["golden_majority"]  # CORRECT / INCORRECT
        hm_pairs.append((majority[i], gm))
    hm_agree = sum(1 for x, y in hm_pairs if x == y) / len(hm_pairs)
    kappa_hm = cohen_kappa(hm_pairs)
    print(f"\n[2] Human-majority vs MACHINE consensus: {hm_agree*100:.1f}%  kappa={kappa_hm:.3f}")

    # 3) human majority vs each channel
    for ch_name, ch_field in (("conflict", "verdict_conflict"), ("rebuttal", "verdict_rebuttal")):
        pairs = []
        for i in resolved:
            cv = key[i][ch_field].upper()
            pairs.append((majority[i], cv))
        agree = sum(1 for x, y in pairs if x == y) / len(pairs)
        k = cohen_kappa(pairs)
        print(f"[3] Human-majority vs {ch_name:8s} channel: {agree*100:.1f}%  kappa={k:.3f}")

    # 4) by epistemic type
    print("\n[4] Human-majority vs machine consensus, by epistemic type:")
    by_epi = {}
    for i in resolved:
        e = key[i].get("epistemic_meta") or "n/a"
        by_epi.setdefault(e, []).append((majority[i], key[i]["golden_majority"]))
    for e, pairs in sorted(by_epi.items(), key=lambda x: -len(x[1])):
        ag = sum(1 for x, y in pairs if x == y) / len(pairs)
        print(f"    {e:6s}: n={len(pairs)} agreement={ag*100:.1f}%")

    # save paper-ready summary
    summary = {
        "n_resolved": len(resolved),
        "human_human_agreement": hh_agree,
        "human_human_kappa": kappa_hh,
        "human_machine_agreement": hm_agree,
        "human_machine_kappa": kappa_hm,
    }
    with open("agreement_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("\nsaved agreement_summary.json (paper Sec. 6.2 regenerates from these values)")


if __name__ == "__main__":
    main()