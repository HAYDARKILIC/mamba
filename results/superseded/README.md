# Superseded pre-correction results

`master_results_20260710_193652.csv` is the result frame of the **first**
execution of the evaluation grid, before the preprocessing defect described
in the paper (§5.4) and in `supplementary/` (S4) was corrected. In that
execution the `Attack` column of NF-UNSW-NB15-v2 — the attack-family name —
was label-encoded and admitted into the feature matrix, so every
NF-UNSW-NB15 figure in this file is contaminated by target leakage.

**It is not the source of any number reported in the paper.** Every
published figure comes from `../master_results.csv` (run `20261021_031658`).
This frame is deposited for one reason only: the paper states that the
obfuscated-gradient diagnostic (5), insensitivity to the distortion bound,
fired in the earlier execution and no longer does, and a reader should be
able to check that claim rather than take it on trust.

Reproducing the two figures the paper quotes:

```python
import pandas as pd
d = pd.read_csv("master_results_20260710_193652.csv")
nf = d[(d.dataset == "NF-UNSW-NB15") & (d.model == "Transformer")]
atk = nf[nf.attack.isin(["FGSM", "PGD"])]
print(atk.groupby(["attack", "eps"]).accuracy.mean().min())   # 0.997629
clean = nf[nf.attack == "clean"].accuracy.mean()              # 0.999979
print(clean - atk.groupby(["attack", "eps"]).accuracy.mean().min())  # 0.002351
```

The Transformer's accuracy on that corpus never fell below 0.9976 at any
budget, so a tenfold increase in epsilon cost it at most 0.0024 — the
signature of insensitivity to the distortion bound. The signature was
produced by the leaking feature, not by the architecture, which is why the
paper reports it as a caution: a leaking feature can imitate gradient
masking on this diagnostic.

NSL-KDD and CSE-CIC-IDS2018 were never reached by the defect; their rows in
this file agree with the corrected run.

## The data-health log

`../logs/data_validation_20260710_193652.log` is the automated
pre-training health audit, and it was written by this same first
execution. It is kept under `results/logs/` rather than here because
most of what it reports is unaffected by the correction: NSL-KDD and
CSE-CIC-IDS2018 were preprocessed identically in both runs
(`data_version` `v3_groupsplit`, see `../logs/config.json`), so their
NaN/Inf findings, class distributions and near-zero-variance counts
(24 and 8) describe the published run as well. Its record counts and
class proportions for NF-UNSW-NB15 are also unchanged, because the
defect removed a column and not a row.

One line in it is superseded. The near-zero-variance count for
NF-UNSW-NB15 reads 22, which is 64 sequence positions minus the 42
feature columns of the leaking matrix. With `Attack` withheld the
corpus contributes 41 features and therefore 23 constant padded
positions. The audit cell was not re-run after the correction, so no
post-correction count is deposited; `supplementary/` (S3) states this
where it quotes the figure.
