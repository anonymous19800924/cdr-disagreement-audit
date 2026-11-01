# Maintenance Guide

This package is the artifact for the paper *Reliability-Weighted Reference Labels for AI-Reviewer Benchmarks*. It is hosted at:

**https://anonymous.4open.science/r/anonymous19800924**

## 1. Repository model

- `anonymous.4open.science` mirrors a GitHub repository. To update the public link, push to the linked private repository; the anonymized URL updates automatically.
- Keep the artifact directory (`artifact/`) as the repository root. Do not re-root files; the scripts resolve `data/` relative to `scripts/..`.
- The paper's `sections/8_artifacts.tex` and `artifact/croissant.json` both cite the URL above. If the URL changes, update both.

## 2. Versioning policy

- Semantic versioning: `MAJOR.MINOR.PATCH`.
  - MAJOR: a released file is removed or renamed, or a statistic changes.
  - MINOR: new files or new statistics added.
  - PATCH: documentation, typo, or script robustness fixes with unchanged outputs.
- Every release appends an entry to `CHANGELOG.md` (date, version, files touched, reason, whether any paper number changes).
- Never delete a released file in a PATCH/MINOR release; deprecate it in `CHANGELOG.md` and keep it for one MAJOR cycle.

## 3. What must be checked before every push

1. `python run_all.py` (or `make repro`) exits 0 and prints:
   - rebuttal 59/80 = 73.75%, conflict 21/80 = 26.25%, subgroup A 31/40 = 77.5%;
   - human–human 88.8% (κ=0.774), human–machine 87.3% (κ=0.745), 80.3% vs 19.7%;
   - held-out 71.2 / 68.5 / 31.5 and the ε-curve (71.2, 69.9, 69.2, 67.2, 61.0, 48.1);
   - k=2 rank 3 and k=3 rank 7.
2. `grep -riE "author|affiliation|@.*\\.(edu|com)|tkde|ieee" artifact/` returns only legitimate reference venues (no author names, no internal paths, no journal-version mentions).
3. Both license files are present and every new file is covered by one of them.
4. `croissant.json` parses as JSON and its `url` matches the hosted link.
5. `README.md` file table lists every new file; `UPLOAD_MANIFEST.txt` is regenerated.

## 4. Updating numbers (paper revision)

Numbers in the artifact are generated from the server-side pipeline, not hand-edited. When a statistic changes:

1. Re-run the corresponding server export (`export_audit_artifacts.py`, `export_surface_items.py`, `export_heldout_items.py`) to regenerate `data/*.json|csv`.
2. Copy the regenerated files into `data/`; do not edit them by hand.
3. Bump MINOR or MAJOR in `CHANGELOG.md`, rerun `run_all.py`, and update the paper if a headline number changed.
4. If a caption or table in the paper changes, update `README.md` and the datasheet's "Missing data" section if coverage changed.

## 5. Data updates and provenance

- Pin the source-corpus snapshot used for every release (record the corpus commit/date in `CHANGELOG.md`). Do not silently change the 10,762-opinion surface.
- If the corpus adds venue-years, re-export `surface_items.csv` and `corpus_counts.json`; the anchor/held-out sets are independent and should not be regenerated unless a new arbitration round is run.
- Stance dumps (`stance_results.jsonl`, `stance_ac_localized.jsonl`) are the largest files; keep them gzipped if repository size becomes an issue, and update the scripts' `open(...)` calls accordingly.

## 6. Issue handling

- During anonymity, use the linked private repository's issue tracker; do not create public issues referencing the paper.
- After acceptance, make the repository public, enable issues, and keep the anonymized URL as a redirect note in the README.
- Triage: reproducibility failures first (script error → PATCH), then documentation gaps (PATCH), then data corrections (MINOR/MAJOR).

## 7. Release checklist

- [ ] `run_all.py` passes 5/5
- [ ] no anonymity leaks (grep above)
- [ ] `CHANGELOG.md` updated with version/date/files/reason
- [ ] `README.md` table and `UPLOAD_MANIFEST.txt` regenerated
- [ ] `croissant.json` URL and version match
- [ ] licenses unchanged and applicable
- [ ] paper §8 URL matches the hosted link
- [ ] zip archive rebuilt (`artifact_v<version>.zip`) and uploaded, if the venue requires a single file
