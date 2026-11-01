# Artifact — Reliability-Weighted Peer-Review Reference Labels

Self-contained package for the paper *Reliability-Weighted Reference Labels for AI-Reviewer Benchmarks* (NeurIPS D&B submission, anonymized).

**Repository:** https://anonymous.4open.science/r/anonymous19800924

**Maintenance:** see `MAINTENANCE.md`; release history in `CHANGELOG.md`. Run `python run_all.py` (or `make repro`) before every push.

## Contents

| Path | Content |
|---|---|
| `data/surface_items.csv` | 10,762 disputed opinions: key, paper id, venue-year, decision, channel verdicts, taxonomy category, posterior, released label, calibration-set flag, epistemic type. |
| `data/anchor_votes.jsonl` | 320 item–family vote records (up to 3 votes each) for the 80 anchor items. |
| `data/anchor_consensus.jsonl` | 80-item five-vote consensus. |
| `data/anchor_key.jsonl` | item_id ↔ key mapping, channel verdicts, subgroup, epistemic type, golden majority. |
| `data/anchor_annotator1.csv`, `anchor_annotator2.csv` | Two blind human annotation sheets (anonymized). |
| `data/anchor_agreement.json` | Human agreement statistics. |
| `data/heldout_items.csv` | 80 held-out items with channel verdicts and subgroup; the 73 with full four-family votes are the evaluation set. |
| `data/cross_family_heldout.jsonl`, `answer_key_heldout.jsonl` | Held-out arbitration votes and keys. |
| `data/heldout_report.json` | Held-out per-cell accuracy, McNemar statistics, bootstrap CI, and the sensitivity curve. |
| `data/calibration_set.tsv`, `calibration_votes.jsonl` | 280 calibration items and the 300×3 vote records. |
| `data/cross_judge_scores.jsonl`, `cross_judge_report.json` | Second-judge re-scoring log and summary (Sec. 6). |
| `data/dual_scores.jsonl`, `dual_decisions.json`, `ai_opinions.jsonl` | Downstream dual-reference scoring log, 30-paper decisions, 745-opinion generation manifest. |
| `data/stance_results.jsonl`, `stance_ac_localized.jsonl` | Meta-review and area-chair stance signals. |
| `data/corpus_counts.json`, `signal_coverage.json`, `vision_report.json`, `vote_completeness.json`, `cal_report.json` | Audit statistics. |
| `analysis/` | Calibration (ECE/Brier), τ sweep, signal ablation, consensus robustness, human-subset checks, Wilson CIs; `results/*.json` holds every number reported in Sec. 6.7 and App. B. |
| `scripts/` | Legacy reproduction scripts for the channel audit, human agreement, sensitivity, theory, and symmetry. |
| `croissant.json` | Croissant metadata for the core tables. |
| `DATASHEET.md`, `DATA_STATEMENT.md`, `LICENSE-CODE`, `LICENSE-DATA` | Documentation and licensing. |

## Reproduce

```
python run_all.py                          # audit, agreement, sensitivity, theory, symmetry
make analysis                              # calibration, tau, ablation, consensus, human checks
```

or individually:

```
python scripts/verify_consensus.py
python scripts/verify_agreement_v2.py --a1 data/anchor_annotator1.csv \
    --a2 data/anchor_annotator2.csv --key data/anchor_key.jsonl
python scripts/sensitivity.py
python scripts/verify_theory.py
python scripts/verify_symmetry.py
cd analysis && python cal_metrics.py && python tau_sweep.py && python ablation.py \
    && python consensus_robustness.py && python human_check.py && python audit_cis.py
```

Requirements: Python 3.9+, NumPy, SymPy. No network access and no model calls are needed.

## Notes

- The corpus itself is not redistributed; `surface_items.csv` contains only derived metadata and public opinion identifiers.
- Every evaluation item lies on the disagreement surface, so conflict and rebuttal accuracies are exact complements; only the sign is empirical.
- The 80 anchor keys are inside the 280-item calibration set; 12 of the 73 held-out keys are also in that set. The overlap is disclosed in the paper.
- `heldout_report.json` matches the held-out and sensitivity numbers. The legacy file `sensitivity_n4.json` (older 110-item evaluation set) is **not** part of this package.
- A public repository URL will replace this package location at camera-ready; the Croissant file will point to the hosted copy.
