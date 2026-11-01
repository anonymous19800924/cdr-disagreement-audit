# Changelog

## v1.1.0 — 2026-11-01

Revision accompanying the resubmission. Adds the calibration/robustness analysis and corrects denominator and overlap statements.

- Analysis: `analysis/` adds ECE/Brier/reliability (`cal_metrics.py`), τ sweep (`tau_sweep.py`), signal ablation (`ablation.py`), consensus variants (`consensus_robustness.py`), human-subset checks (`human_check.py`), Wilson CIs (`audit_cis.py`), and held-out CI/McNemar/tie sensitivity (`heldout_repair.py`); `analysis/results/*.json` holds the numbers.
- Held-out: added `heldout_predictions.csv` (73 anchor-fit posteriors behind Table 2), `bootstrap_ci95_cal_minus_rb`/`mcnemar` fields, and the 2:2-tie sensitivity; removed an internal audit JSON that conflicted with `human_check.json`.
- Data: added `cross_judge_scores.jsonl` and `cross_judge_report.json`; synchronized `heldout_report.json` with the regenerated sensitivity curve.
- Corrections: inter-family agreement is 70.1–83.9% on commonly voted items (60.0–73.8% only if missing votes count as disagreements); unanimity is 37/60 (61.7%); dropping one family leaves 5–14 items without a majority; anchor value/mixed share is 20/43 (46.5%); held-out overlap with the 280-item set is 12/73.
- Clarifications: conflict and rebuttal accuracies are exact complements on the surface; held-out ECE is 0.203 at τ=0 and the lower τ-grid values are post-hoc; the 85.8% figure is a model prediction rate, not an observed label rate.
- Verification: all scripts reproduce the reported numbers from the package alone.

## v1.0.0 — 2026-10-30

Initial public release accompanying the NeurIPS D&B submission.

- Data: `surface_items.csv` (10,762 opinions with IDs and posteriors), anchor votes/consensus/key, human annotation sheets and agreement summary, 280-item calibration set and votes, 80 held-out items with votes and keys, dual-reference scoring logs and decisions, 745-opinion generation manifest, stance signals, audit statistics.
- Scripts: `verify_consensus.py`, `verify_agreement_v2.py`, `sensitivity.py`, `verify_theory.py`, `verify_symmetry.py`; `run_all.py` and `Makefile` (`make repro`).
- Metadata: Croissant, datasheet, data statement, dual license, citation file.
- Verification: all five scripts reproduce the reported statistics from the package alone (73.75/26.25, 87.3%/κ=0.745, 71.2/68.5/31.5, ε-curve, ranks 3/7).
- Known exclusions: source corpus text beyond the two annotation sheets; corpus EM likelihood trace (covered by theorem + synthetic replication); legacy `sensitivity_n4.json` (older 110-item run, not part of this release).
