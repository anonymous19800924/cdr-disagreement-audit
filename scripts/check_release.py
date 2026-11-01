#!/usr/bin/env python3
"""Release self-check: metadata integrity, required files, anonymity hygiene."""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://anonymous.4open.science/r/cdr-disagreement-audit-C1E7"
problems = []

required = [
    "README.md", "DATASHEET.md", "DATA_STATEMENT.md", "MAINTENANCE.md", "CHANGELOG.md",
    "LICENSE-CODE", "LICENSE-DATA", "CITATION.cff", "croissant.json", "Makefile", "run_all.py",
    "requirements.txt", "data/surface_items.csv", "data/anchor_votes.jsonl",
    "data/anchor_consensus.jsonl", "data/anchor_key.jsonl", "data/heldout_items.csv",
    "data/heldout_report.json", "data/dual_scores.jsonl", "data/ai_opinions.jsonl",
    "scripts/verify_consensus.py", "scripts/verify_agreement_v2.py", "scripts/sensitivity.py",
    "scripts/verify_theory.py", "scripts/verify_symmetry.py",
]
for rel in required:
    if not os.path.exists(os.path.join(ROOT, rel)):
        problems.append(f"missing required file: {rel}")

with open(os.path.join(ROOT, "croissant.json"), encoding="utf-8") as f:
    try:
        cr = json.load(f)
    except Exception as e:
        problems.append(f"croissant.json does not parse: {e}")
        cr = {}
if cr.get("url") != URL:
    problems.append("croissant.json url mismatch")
if cr.get("license") != "https://creativecommons.org/licenses/by/4.0/":
    problems.append("croissant.json license mismatch")

for rel in ("README.md", "DATASHEET.md"):
    txt = open(os.path.join(ROOT, rel), encoding="utf-8").read()
    if URL not in txt:
        problems.append(f"{rel} does not mention the repository URL")

leak_patterns = [
    (r"@[\w.-]+\.(edu|com|cn)\b", "email address"),
    (r"\bTKDE\b", "journal-version mention"),
    (r"/mnt/diskvol", "internal server path"),
    (r"\bShilin\b|\bHexuan\b|\bzhangshilin\b", "author name"),
]
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in filenames:
        if fn.endswith((".pyc", ".zip")):
            continue
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        if rel == "scripts/check_release.py":
            continue
        try:
            txt = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for pat, label in leak_patterns:
            if re.search(pat, txt):
                # allow the source-corpus venue references inside README notes only if legitimate
                problems.append(f"possible {label} in {rel}")

if problems:
    print("RELEASE CHECK FAILED")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("RELEASE CHECK PASSED: required files present, metadata consistent, no obvious anonymity leaks.")
