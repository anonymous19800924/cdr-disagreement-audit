"""Wilson CIs for the Table-1 channel audit (five-vote consensus)."""
import math
from common import DATA, _jsonl, save

key = {r["item_id"]: r for r in _jsonl(DATA + r"\anchor_key.jsonl")}
cons = {r["item_id"]: r["consensus"] for r in _jsonl(DATA + r"\anchor_consensus.jsonl")}

def wilson(k, n, z=1.96):
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]

items = [i for i in key if cons.get(i) in ("CORRECT", "INCORRECT")]
out = {"overall": {}, "subgroup_A": {}, "subgroup_B": {}}
for name, subset in (("overall", items),
                     ("subgroup_A", [i for i in items if key[i]["subgroup"] == "A"]),
                     ("subgroup_B", [i for i in items if key[i]["subgroup"] == "B"])):
    for ch, verdict in (("conflict", "verdict_conflict"), ("rebuttal", "verdict_rebuttal")):
        hit = sum(1 for i in subset
                  if (1 if key[i][verdict] == "incorrect" else 0) == (1 if cons[i] == "INCORRECT" else 0))
        out[name][ch] = {"k": hit, "n": len(subset), "rate": hit / len(subset) if subset else None,
                         "ci": wilson(hit, len(subset)) if subset else None}
save("audit_cis.json", out)
print("overall:", out["overall"])
