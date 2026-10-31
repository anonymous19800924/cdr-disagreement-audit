# Data Statement

Following Bender & Friedman (2018).

- **Language(s):** English (review text, meta-reviews, author responses).
- **Provenance:** derived from a publicly released peer-review corpus (CC BY 4.0 for OpenReview-derived content); opinion identifiers and channel verdicts come from that corpus's pipeline; arbitration votes, consensus labels, calibrated posteriors, and human annotations were produced in this work.
- **Speaker demographics:** reviewers, authors, and area chairs of ICLR (2017–2025) and NeurIPS (2021–2024); demographics are not available and not inferred.
- **Annotation:** two volunteer annotators (graduate ML background), blind to channel verdicts; 80 items; inter-annotator agreement 88.8% (κ = 0.774); no compensation; annotators anonymized.
- **Content notes:** the package contains metadata plus two annotation sheets that reproduce the review excerpts annotators saw (title, abstract, opinion, meta-review, AC comments) under the source corpus's license; it does not add review text beyond those excerpts.
- **Limitations:** machine consensus is silver-standard; epistemic-type coverage is partial (336/10,762 surface opinions); stance signals are sparse (13.6% / 10.8% of the surface).
- **Intended use:** meta-scientific measurement and benchmark construction; not for evaluating individuals.
