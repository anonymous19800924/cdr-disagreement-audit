# Discriminability of LLM-Judge Adoption in Disagreement Resolution — Anonymous Artifact

Artifact release accompanying the ARR 2026-10 submission (double-blind review).

## What this repository contains

This release contains everything needed to verify the paper's quantitative claims:

1. **The harness** (`audited-release/`) — protocol code for the disagreement-resolution
   interaction (consultant rounds, judge verdicts), prompts, and baseline protocols
   (naive judge, double consultancy), plus the three ablations (no-crux, no-conflict, placebo).
2. **Per-episode result logs** (`results/`) — every episode with consultant actions,
   consensus state, and judge verdict, exactly as used in the paper's tables.
3. **Analysis scripts** (`scripts/`) — statistics, discriminability estimation,
   figures, and the U6 construction census.
4. **Question data** (`audited-release/data/`) — per-model answer files for GPQA, HLE, SuperGPQA.

## Key quantities to verify

| Claim | Where |
|---|---|
| Main arms: Δ_disc = +0.000 across DR / no-crux / no-conflict | `results/dr_*.json`, `results/drnocrux_*.json`, `results/drnoconf_*.json` |
| Placebo arm: Δ_disc = +0.111 | `results/placebo_GPQA_glm46_glm45_judge_glm45air.json` |
| U1 implanted-consensus: judge-only, Δ = +0.512 (glm-4.5-air) / −0.013 (glm-4-flash) | `results/u1_implanted.json(.jsonl)` |
| W-5 cross-family rejudge: qwen-flash Δ = +0.000 on DR arm, +0.325 on implanted arm | `results/u1_implanted_w5_qwenflash.json(.jsonl)`, `results/dr_*` |
| U6 construction census: 620/620 forced one-right-one-wrong | `scripts/u6_census.py` + result logs |

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .                                     # see pyproject.toml
cp env.example .env                                  # fill in your own API keys
python audited-release/run.py --help                 # protocol runner
python scripts/analyze.py --help                     # statistics over result logs
python scripts/u1_stats.py results/u1_implanted.jsonl
python scripts/u6_census.py --help
```

All model calls go through cached endpoints; reruns reuse per-episode artifacts so
costs are limited to uncached calls.

## Provenance note

`audited-release/` is the released codebase of the disagreement-resolution protocol
that this paper audits, included unmodified (see `audited-release/README.upstream.md`)
so that reviewers can distinguish upstream protocol behaviour from our instrumentation.
The `results/` directory and `scripts/u1_*`/`u6_*` files are this paper's own artifacts.

## Anonymity

This repository is maintained anonymously for double-blind review. Upon acceptance,
the de-anonymized release will supersede it.
