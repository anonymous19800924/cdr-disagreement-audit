# Artifact — Calibrated Peer-Review Reference Labels

Self-contained package for the paper *Calibrated Reference Labels for AI-Reviewer Benchmarks* (NeurIPS D&B submission, anonymized).

**Repository:** https://anonymous.4open.science/r/anonymous19800924

**Maintenance:** see `MAINTENANCE.md`; release history in `CHANGELOG.md`. The package is versioned; run `python run_all.py` (or `make repro`) before every push.

## Contents

| Path | Content |
|---|---|
| `data/surface_items.csv` | 10,762 disputed opinions: key, paper id, venue-year, decision, channel verdicts, taxonomy category, calibrated posterior, calibrated label, calibration-set flag, epistemic type. |
| `data/anchor_votes.jsonl` | 320 item–family vote records (up to 3 votes each) for the 80 anchor items. |
| `data/anchor_consensus.jsonl` | 80-item five-vote consensus. |
| `data/anchor_key.jsonl` | item_id ↔ key mapping, channel verdicts, subgroup, epistemic type, golden majority. |
| `data/anchor_annotator1.csv`, `anchor_annotator2.csv` | Two blind human annotation sheets (anonymized). |
| `data/anchor_agreement.json` | Human agreement statistics. |
| `data/heldout_items.csv` | 80 held-out items with channel verdicts and subgroup; the 73 with full four-family votes are the evaluation set. |
| `data/cross_family_heldout.jsonl`, `answer_key_heldout.jsonl` | Held-out arbitration votes and keys. |
| `data/heldout_report.json` | Held-out per-cell accuracy, McNemar statistics, bootstrap CI, and the sensitivity curve (matches Table 2 and the sensitivity numbers). |
| `data/calibration_set.tsv`, `calibration_votes.jsonl` | 280 calibration items (retained from 300 voted items with a confident majority) and the 300×3 vote records. |
| `data/dual_scores.jsonl`, `dual_decisions.json`, `ai_opinions.jsonl` | Downstream dual-reference scoring log, 30-paper decisions, 745-opinion generation manifest. |
| `data/stance_results.jsonl`, `stance_ac_localized.jsonl` | Meta-review and area-chair stance signals (raw JSONL). |
| `data/corpus_counts.json`, `signal_coverage.json`, `vision_report.json`, `vote_completeness.json`, `cal_report.json` | Audit statistics behind the paper. |
| `scripts/verify_consensus.py` | Regenerates the channel-reliability audit (Sec. 6.1). |
| `scripts/verify_agreement_v2.py` | Regenerates the human-verification statistics (Sec. 6.2). |
| `scripts/sensitivity.py` | Regenerates the held-out table and noise-sensitivity curve (Secs. 6.3–6.4). |
| `scripts/verify_theory.py`, `scripts/verify_symmetry.py` | Symbolic/Monte-Carlo theory checks (Sec. 4). |
| `croissant.json` | Croissant metadata for the core tables. |
| `DATASHEET.md`, `LICENSE-CODE`, `LICENSE-DATA` | Documentation and licensing. |

## Reproduce

```
python scripts/verify_consensus.py          # 73.8% / 26.2% / 77.5% / robustness
python scripts/verify_agreement_v2.py --a1 data/anchor_annotator1.csv \
    --a2 data/anchor_annotator2.csv --key data/anchor_key.jsonl
python scripts/sensitivity.py               # 71.2 / 68.5 / 31.5 + eps sweep
python scripts/verify_theory.py             # identities + synthetic EM
python scripts/verify_symmetry.py           # S,T,U algebra + ranks
```

Requirements: Python 3.9+, NumPy, SymPy. No network access and no model calls are needed.

## Notes

- The corpus itself is not redistributed; `surface_items.csv` contains only derived metadata and public opinion identifiers.
- `heldout_report.json` matches the paper's Tables 2–3 exactly. The legacy file `sensitivity_n4.json` (older 110-item evaluation set) is **not** part of this package.
- A public repository URL will replace this package location at camera-ready; the Croissant file will point to the hosted copy.
