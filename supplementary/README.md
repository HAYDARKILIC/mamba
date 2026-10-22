# Supplementary material

`supplementary.pdf` (12 pages, built from `supplementary.tex`) holds the
material the paper refers to as released with the archived artefact. The
paper is kept to the journal's fifteen-page limit; nothing was discarded
to reach it, and everything removed is here or in the result files this
document points to.

| Section | Contents |
|---|---|
| S1 | Derivations and definitions omitted from the paper's methodology: the class-weighted risk, standardisation and the projection operator, the continuous-time formulation and its zero-order-hold discretisation, the selective projection and block dimensions, the detector head, the closed-form recurrence used in the second validation check, the optimisation schedule and the balanced-accuracy selection criterion, the per-category detection rates, the latency and throughput definitions, and the paired test with the Holm–Bonferroni step-down |
| S2 | The two pseudo-code listings: leakage-aware partitioning, and the selective-scan forward pass |
| S3 | Extended statement of scope: the full record counts, the NSL-KDD label re-count that verifies the 17 unseen attack types and the 16.63% of test records they cover, the near-zero-variance feature counts and their explanation, the pre-sanitisation counts, the analysable-category criterion, and the split verification |
| S4 | The NF-UNSW-NB15 leakage episode in full: the defect, why the single-threshold audit scores 0.9679 where a membership test scores 1.0000 (both reproducible from the deposited subsample), the detection route, the feature-importance analysis and the feature-removal ablation |
| S5 | The padding-masked re-evaluation in full, with the harness validation and the per-cell dependence |
| S6 | What the aggregate metrics conceal about severity, per attack category |
| S7 | Adversarial training: effective, but insufficiently verified |
| S8 | Threat-model realism and the constraint question |
| S9 | The capacity ablation in full |
| S10 | Index of the deposited tables and figures, and of the superseded pre-correction frame kept so the diagnostic-(5) claim can be checked |

Every quantity in the document is reproduced from the released result
files, and the file each comes from is named where it is used. The two
values that are plain arithmetic rather than measurements are the
padding fraction 23/64 = 35.94% and the parameter shares quoted from
`results/param_counts.json`.

## Building

```bash
cd supplementary
pdflatex supplementary.tex && pdflatex supplementary.tex
```

Requires `algorithm2e`, `booktabs`, `multirow`, `xurl` and `mathptmx`.
