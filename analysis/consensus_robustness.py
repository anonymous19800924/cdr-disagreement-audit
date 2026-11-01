"""B4: consensus-definition robustness for the channel audit (Table 1)."""
import json
from collections import Counter
from common import DATA, FAMS, _jsonl, save

key = {r["item_id"]: r for r in _jsonl(DATA + r"\anchor_key.jsonl")}
votes = {}
for r in _jsonl(DATA + r"\anchor_votes.jsonl"):
    votes.setdefault(r["item_id"], {})[r["family"]] = r["majority"]
saved = {r["item_id"]: r["consensus"] for r in _jsonl(DATA + r"\anchor_consensus.jsonl")}

def label(majority_votes):
    c = Counter(majority_votes)
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]

def metrics(labels):
    items = [i for i in key if labels.get(i) in ("CORRECT", "INCORRECT")]
    reb = sum(1 for i in items if (1 if key[i]["verdict_rebuttal"] == "incorrect" else 0)
              == (1 if labels[i] == "INCORRECT" else 0))
    con = sum(1 for i in items if (1 if key[i]["verdict_conflict"] == "incorrect" else 0)
              == (1 if labels[i] == "INCORRECT" else 0))
    subA = [i for i in items if key[i]["subgroup"] == "A"]
    a_inc = sum(1 for i in subA if labels[i] == "INCORRECT")
    return {"n": len(items), "rebuttal_agree": reb / len(items) if items else None,
            "conflict_agree": con / len(items) if items else None,
            "subA_incorrect": a_inc / len(subA) if subA else None,
            "subA_n": len(subA)}

variants = {"five_vote_saved": {i: saved[i] for i in saved}}
# four-family, no golden
four = {}
for i in key:
    if i in votes and all(f in votes[i] for f in FAMS):
        four[i] = label([votes[i][f] for f in FAMS])
variants["four_family_ties_drop"] = four
variants["four_family_ties_golden"] = {i: (four[i] if four[i] else key[i]["golden_majority"]) for i in four}
variants["four_family_ties_dspro"] = {i: (four[i] if four[i] else votes[i].get("DS-PRO")) for i in four}
# leave-one-family-out from five votes (golden + 4 families)
for drop in FAMS:
    d = {}
    for i in key:
        if i in votes and all(f in votes[i] for f in FAMS if f != drop):
            d[i] = label([key[i]["golden_majority"]] + [votes[i][f] for f in FAMS if f != drop])
    variants[f"loo_{drop}_ties_drop"] = d

out = {name: metrics(v) for name, v in variants.items()}
save("consensus_robustness.json", out)
for name, m in out.items():
    print(f"{name:26s} n={m['n']:2d} reb={m['rebuttal_agree']:.3f} con={m['conflict_agree']:.3f} subA={m['subA_incorrect']:.3f}")
