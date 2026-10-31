# Changelog

## v1.0.0 — 2026-10-30

Initial public release accompanying the NeurIPS D&B submission.

- Data: `surface_items.csv` (10,762 opinions with IDs and calibrated posteriors), anchor votes/consensus/key, human annotation sheets and agreement summary, 280-item calibration set and votes, 80 held-out items with votes and keys, dual-reference scoring logs and decisions, 745-opinion generation manifest, stance signals, audit statistics.
- Scripts: `verify_consensus.py`, `verify_agreement_v2.py`, `sensitivity.py`, `verify_theory.py`, `verify_symmetry.py`; `run_all.py` and `Makefile` (`make repro`).
- Metadata: Croissant, datasheet, data statement, dual license, citation file.
- Verification: all five scripts reproduce the reported statistics from the package alone (73.75/26.25, 87.3%/κ=0.745, 71.2/68.5/31.5, ε-curve, ranks 3/7).
- Known exclusions: source corpus text beyond the two annotation sheets; corpus EM likelihood trace (covered by theorem + synthetic replication); legacy `sensitivity_n4.json` (older 110-item run, not part of this release).
