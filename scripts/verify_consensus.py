#!/usr/bin/env python3
"""Regenerate the channel-reliability audit of Sec. 6.1 from released artifacts.

Inputs:
  - anchor_votes.jsonl      per-item per-family majority verdicts (up to 4 families x 3 votes)
  - anchor_key.jsonl        item_id <-> key, golden majority, channel verdicts
  - anchor_consensus.jsonl  stored five-vote consensus

Outputs:
  - channel agreement against the stored consensus (complementary by construction)
  - pairwise family agreement on commonly voted items, and with missing-as-disagreement
  - unanimity among complete four-family and five-vote records
  - leave-one-family-out: items without a majority, and label changes after golden tie-break
"""
import json, os
from collections import Counter, defaultdict

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

fam_votes = defaultdict(dict)
with open(os.path.join(HERE, "anchor_votes.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("majority"):
            fam_votes[r["item_id"]][r["family"]] = r["majority"]

with open(os.path.join(HERE, "anchor_key.jsonl"), encoding="utf-8") as f:
    keymap = {}
    for l in f:
        r = json.loads(l)
        keymap[r["item_id"]] = r

with open(os.path.join(HERE, "anchor_consensus.jsonl"), encoding="utf-8") as f:
    saved = {json.loads(l)["item_id"]: json.loads(l)["consensus"] for l in f}

fams = ["DS-PRO", "GLM", "QWEN", "KIMI"]
items = sorted(k for k in keymap if k in fam_votes)
assert len(items) == 80, f"expected 80 items, got {len(items)}"


def strict(vals):
    c = Counter(vals).most_common()
    if not c:
        return None
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None
    return c[0][0]


# Stored consensus is authoritative. Report two recomputations: with the golden vote
# (should match) and with the four families only (the held-out label rule; differs on 5/80).
mismatch5 = 0
for k in items:
    vals = [keymap[k]["golden_majority"]] + [fam_votes[k][f] for f in fams if f in fam_votes[k]]
    m = strict(vals)
    if m is not None and m != saved[k]:
        mismatch5 += 1
print(f"documented five-vote rule vs stored consensus: {mismatch5}/80 differ (stored labels used below)")
mismatch4 = 0
for k in items:
    vals = [fam_votes[k][f] for f in fams if f in fam_votes[k]]
    m = strict(vals)
    if m is not None and m != saved[k]:
        mismatch4 += 1
print(f"four-family rule vs stored consensus: {mismatch4}/80 differ")

n = len(items)
for ch, label in (("verdict_rebuttal", "rebuttal channel"), ("verdict_conflict", "conflict channel")):
    agr = sum(1 for k in items
              if (1 if keymap[k][ch].lower() == "incorrect" else 0) == (1 if saved[k] == "INCORRECT" else 0))
    print(f"{label} agreement: {agr}/{n} = {agr/n:.4f}")

subA = [k for k in items if keymap[k]["subgroup"] == "A"]
a_inc = sum(1 for k in subA if saved[k] == "INCORRECT")
print(f"subgroup A judged incorrect: {a_inc}/{len(subA)} = {a_inc/len(subA):.4f}")

print("\npairwise family agreement (item-level majorities; missing votes excluded):")
for i in range(len(fams)):
    for j in range(i + 1, len(fams)):
        a, b = fams[i], fams[j]
        both = [k for k in items if a in fam_votes[k] and b in fam_votes[k]]
        agr = sum(1 for k in both if fam_votes[k][a] == fam_votes[k][b])
        all80 = sum(1 for k in items if k in both and fam_votes[k][a] == fam_votes[k][b])
        print(f"  {a}-{b}: {agr}/{len(both)} = {agr/len(both):.3f} common; {all80}/80 = {all80/80:.3f} missing-as-disagreement")

complete4 = [k for k in items if all(f in fam_votes[k] for f in fams)]
uni4 = sum(1 for k in complete4 if len({fam_votes[k][f] for f in fams}) == 1)
uni5 = sum(1 for k in complete4
           if len({keymap[k]["golden_majority"]} | {fam_votes[k][f] for f in fams}) == 1)
print(f"unanimity: four-family {uni4}/{len(complete4)} = {uni4/len(complete4):.3f}; "
      f"five-vote {uni5}/{len(complete4)} = {uni5/len(complete4):.3f}")

print("\nleave-one-family-out (golden vote retained):")
for drop in fams:
    ties = changed = skipped = 0
    for k in items:
        vals = [keymap[k]["golden_majority"]] + [fam_votes[k][f] for f in fams if f != drop and f in fam_votes[k]]
        if len(vals) < 4:
            skipped += 1
            continue
        m = strict(vals)
        if m is None:
            ties += 1
            m = keymap[k]["golden_majority"]
        if m != saved[k]:
            changed += 1
    print(f"  drop {drop}: {ties} of {80-skipped} complete items without a majority; "
          f"{changed} labels change after golden tie-break ({skipped} incomplete skipped)")
