"""B5: score the released posterior against human labels on the agreed anchor items."""
import csv, json
import numpy as np
from common import DATA, _jsonl, kappa, save

post = {}
with open(DATA + r"\surface_items.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        post[r["key"]] = float(r["posterior_incorrect"])

key = {r["item_id"]: r for r in _jsonl(DATA + r"\anchor_key.jsonl")}
cons = {r["item_id"]: r["consensus"] for r in _jsonl(DATA + r"\anchor_consensus.jsonl")}

def load_ann(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("verdict(CORRECT/INCORRECT)") or "").strip().upper()
            if v in ("CORRECT", "INCORRECT"):
                out[r["item_id"]] = 1 if v == "INCORRECT" else 0
    return out

a1 = load_ann(DATA + r"\anchor_annotator1.csv")
a2 = load_ann(DATA + r"\anchor_annotator2.csv")
agreed = [i for i in sorted(set(a1) & set(a2)) if a1[i] == a2[i]]
human = {i: a1[i] for i in agreed}

rows = []
for iid in agreed:
    k = key[iid]["key"]
    if k not in post:
        continue
    rows.append({"item_id": iid, "p": post[k], "human": human[iid],
                 "machine": 1 if cons[iid] == "INCORRECT" else 0,
                 "type": (key[iid].get("epistemic_meta") or "").upper(),
                 "conflict": 1 if key[iid]["verdict_conflict"] == "incorrect" else 0,
                 "rebuttal": 1 if key[iid]["verdict_rebuttal"] == "incorrect" else 0})
p = np.array([r["p"] for r in rows]); h = np.array([r["human"] for r in rows])
m = np.array([r["machine"] for r in rows])
pred = (p >= 0.5).astype(int)
by_type = {}
for typ in ("FACT", "NORM", "VALUE", "MIXED"):
    idx = [i for i, r in enumerate(rows) if r["type"] == typ]
    by_type[typ] = {"n": len(idx),
                    "machine_agree": float((m[idx] == h[idx]).mean()) if idx else None}
channels = {}
for ch in ("conflict", "rebuttal"):
    c = np.array([r[ch] for r in rows])
    channels[ch] = {"agreement": float((c == h).mean()), "kappa": kappa(c, h)}
res = {
    "n_agreed": len(rows),
    "posterior_vs_human": {"agreement": float((pred == h).mean()), "kappa": kappa(pred, h)},
    "machine_vs_human": {"agreement": float((m == h).mean()), "kappa": kappa(m, h)},
    "channels_vs_human": channels,
    "by_type_machine_vs_human": by_type,
    "confusion": {"post_inc_human_inc": int(((pred == 1) & (h == 1)).sum()),
                  "post_inc_human_cor": int(((pred == 1) & (h == 0)).sum()),
                  "post_cor_human_inc": int(((pred == 0) & (h == 1)).sum()),
                  "post_cor_human_cor": int(((pred == 0) & (h == 0)).sum())},
    "posterior_mean_by_human": {"human_correct": float(p[h == 0].mean()) if (h == 0).any() else None,
                                "human_incorrect": float(p[h == 1].mean()) if (h == 1).any() else None},
}
save("human_check.json", res)
print(f"n={res['n_agreed']}  posterior-vs-human={res['posterior_vs_human']['agreement']:.3f} "
      f"(kappa={res['posterior_vs_human']['kappa']:.3f})  machine-vs-human={res['machine_vs_human']['agreement']:.3f}")
