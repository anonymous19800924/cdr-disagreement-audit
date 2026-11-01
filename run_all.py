#!/usr/bin/env python3
"""Run the full reproduction suite without make (Windows-friendly)."""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
steps = [
    ("consensus", [PY, os.path.join(HERE, "scripts", "verify_consensus.py")]),
    ("agreement", [PY, os.path.join(HERE, "scripts", "verify_agreement_v2.py"),
                   "--a1", os.path.join(HERE, "data", "anchor_annotator1.csv"),
                   "--a2", os.path.join(HERE, "data", "anchor_annotator2.csv"),
                   "--key", os.path.join(HERE, "data", "anchor_key.jsonl")]),
    ("sensitivity", [PY, os.path.join(HERE, "scripts", "sensitivity.py")]),
    ("theory", [PY, os.path.join(HERE, "scripts", "verify_theory.py")]),
    ("symmetry", [PY, os.path.join(HERE, "scripts", "verify_symmetry.py")]),
    ("calibration", [PY, os.path.join(HERE, "analysis", "cal_metrics.py")]),
    ("tau", [PY, os.path.join(HERE, "analysis", "tau_sweep.py")]),
    ("ablation", [PY, os.path.join(HERE, "analysis", "ablation.py")]),
    ("consensus_variants", [PY, os.path.join(HERE, "analysis", "consensus_robustness.py")]),
    ("human_check", [PY, os.path.join(HERE, "analysis", "human_check.py")]),
    ("audit_cis", [PY, os.path.join(HERE, "analysis", "audit_cis.py")]),
    ("heldout_repair", [PY, os.path.join(HERE, "analysis", "heldout_repair.py")]),
]
failed = 0
for name, cmd in steps:
    print(f"\n=== {name} ===")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"{name} FAILED with code {r.returncode}")
        failed += 1
print(f"\n{len(steps) - failed}/{len(steps)} steps succeeded")
sys.exit(1 if failed else 0)
