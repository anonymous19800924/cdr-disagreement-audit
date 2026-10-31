#!/usr/bin/env python3
"""Regenerate the channel-reliability audit of Sec. 6.1 from released artifacts.

Inputs (same directory):
  - anchor_votes.jsonl      per-item per-family majority verdicts (4 families x 3 votes)
  - anchor_key.jsonl  item_id <-> key mapping + golden majority + channel verdicts
  - anchor_consensus.jsonl       the 5-vote consensus (golden + 4 family majorities)

Outputs (stdout): rebuttal/conflict agreement, subgroup-A incorrect rate,
pairwise inter-family agreement, full unanimity, leave-one-family-out stability.

Expected (paper Sec. 6.1 / Table 1): rebuttal 59/80 = 73.8%, conflict 21/80 = 26.2%,
subgroup A 31/40 = 77.5%, unanimity 39/80 = 48.8%, max leave-one-out change 6/80.
"""
import json, os, sys
from collections import Counter, defaultdict

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

fam_votes = defaultdict(dict)
with open(os.path.join(HERE, "anchor_votes.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        fam_votes[r["item_id"]][r["family"]] = r.get("majority")

with open(os.path.join(HERE, "anchor_key.jsonl"), encoding="utf-8") as f:
    keymap = {json.loads(l)["item_id"]: json.loads(l) for l in f}

with open(os.path.join(HERE, "anchor_consensus.jsonl"), encoding="utf-8") as f:
    saved = {json.loads(l)["item_id"]: json.loads(l)["consensus"] for l in f}

fams = ["DS-PRO", "GLM", "QWEN", "KIMI"]
items = sorted(k for k in keymap if k in fam_votes and all(f in fam_votes[k] for f in fams))
assert len(items) == 80, f"expected 80 items, got {len(items)}"

consensus = {}
for k in items:
    votes = [keymap[k]["golden_majority"]] + [fam_votes[k][f] for f in fams]
    consensus[k] = Counter(votes).most_common(1)[0][0]

mismatch = sum(1 for k in items if consensus[k] != saved[k])
print(f"consensus recompute vs anchor_consensus.jsonl: {'MATCH' if mismatch == 0 else str(mismatch)+' mismatches'}")

n = len(items)
for ch, label in (("verdict_rebuttal", "rebuttal channel"), ("verdict_conflict", "conflict channel")):
    agr = sum(1 for k in items
              if (1 if keymap[k][ch] == "incorrect" else 0) == (1 if consensus[k] == "INCORRECT" else 0))
    print(f"{label} agreement: {agr}/{n} = {agr/n:.4f}")

subA = [k for k in items if keymap[k]["subgroup"] == "A"]
a_inc = sum(1 for k in subA if consensus[k] == "INCORRECT")
print(f"subgroup A judged incorrect: {a_inc}/{len(subA)} = {a_inc/len(subA):.4f}")

print("\npairwise family agreement (item-level majorities):")
for i in range(len(fams)):
    for j in range(i + 1, len(fams)):
        a, b = fams[i], fams[j]
        agr = sum(1 for k in items if fam_votes[k][a] == fam_votes[k][b])
        print(f"  {a}-{b}: {agr}/{n} = {agr/n:.3f}")

unanim = sum(1 for k in items if len({fam_votes[k][f] for f in fams}) == 1)
print(f"full unanimity across 4 families: {unanim}/{n} = {unanim/n:.3f}")

print("\nleave-one-family-out consensus stability:")
for drop in fams:
    rest = [f for f in fams if f != drop]
    changed = sum(1 for k in items
                  if Counter([keymap[k]["golden_majority"]] + [fam_votes[k][f] for f in rest]).most_common(1)[0][0] != consensus[k])
    print(f"  drop {drop}: {changed}/{n} labels change")


