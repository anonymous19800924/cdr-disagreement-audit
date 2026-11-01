# Datasheet — Reliability-Weighted Peer-Review Reference Labels

Following *Datasheets for Datasets* (Gebru et al.).

## Motivation

- **Purpose.** Provide auditable correctness labels and calibrated posteriors for atomic review opinions, together with the arbitration votes and human annotations needed to reproduce every statistic in the accompanying paper.
- **Tasks supported.** Measuring AI-reviewer quality against reliable references; auditing adjudicated benchmark labels; studying latent-truth identifiability; calibrating adjudication pipelines.
- **Non-goals.** Evaluating individual reviewers or venues; automated decision-making about people; redistributing review text.

## Composition

- **Instances.** 10,762 disputed opinions (the disagreement surface of a 3,900-paper, 152,240-opinion review corpus), each with: opinion key, paper id, venue-year, paper decision, the two channel verdicts, corpus taxonomy category, calibrated posterior, calibrated label, calibration-set membership, and epistemic type where available. 80 anchor items additionally carry five-vote arbitration records and two human annotations; 80 held-out items carry arbitration votes and channel verdicts, of which the 73 with full four-family votes are used for evaluation.
- **Label sources.** (i) channel verdicts from the source corpus's released pipeline; (ii) machine arbitration by four model families plus a DeepSeek three-vote arbiter; (iii) two human annotators (blind; anonymized); (iv) calibrated posteriors from a one-shot logistic combination of channel and stance signals.
- **Missing data.** 20 of 80 anchor items have at least one family with fewer than three retained votes (10 of them GLM); 50/80 carry the full 4×3 record. 16/80 anchor items and 56/280 calibration items are epistemically untyped ("n/a"). Stance signals cover 13.6% (meta) and 10.8% (AC) of surface opinions.
- **Sensitive content.** No reviewer identities or non-public metadata. Opinion keys are the corpus's public identifiers. The package includes two human annotation sheets that reproduce the review excerpts annotators saw (title, abstract, opinion, meta-review, AC comments), under the source corpus's CC BY 4.0 terms; no additional review text is included.

## Collection process

- Channel labels are produced by the source corpus's documented pipeline (including LLM-assisted validation); we use them unchanged.
- Arbitration: four families (DeepSeek-v4, GLM-5, Qwen3.8, Kimi-k2.7) × up to three temperature-varied votes (0.2/0.4/0.7), blind to channel verdicts; consensus is the majority over five votes (DeepSeek arbiter majority + four family majorities).
- Human annotation: two volunteers from the authors' institution, blind to channel verdicts and to each other; 80 items; 88.8% inter-annotator agreement (κ=0.774); 9 disagreements dropped rather than third-party arbitrated.

## Preprocessing and quality

- No text normalization; identifiers preserved. Retained votes are recorded verbatim; family majorities are used where individual votes were not retained.
- Verification: released scripts regenerate the channel audit, human agreement, held-out table, sensitivity curve, and theory checks; deterministic seeds are fixed.

## Uses

- **Intended.** Meta-scientific measurement; benchmark construction; reliability auditing.
- **Caution.** Labels are silver-standard machine consensus, not ground truth; calibrated posteriors are model outputs with uncertainty and should be thresholded per use case.

## Distribution and maintenance

- **License.** Derived data CC BY 4.0; code MIT.
- **Hosting.** Anonymized repository: https://anonymous.4open.science/r/anonymous19800924 (public mirror at camera-ready).
- **Maintenance.** Versioned releases (`CHANGELOG.md`); issues tracked in the repository; corpus updates are independent. See `MAINTENANCE.md`.

## Ethics

- Uses public, license-cleared review metadata; no IRB approval required for annotating public metadata; annotators consented to participate and are anonymized; no compensation was provided (volunteer annotation).
