# Adversarial Robustness of Mamba-Based Network Intrusion Detection — Code and Results

Reproducibility repository accompanying the manuscript under review.
**The manuscript itself is not included here**; this repository contains only
code, configuration and the numerical outputs on which the reported figures
and tables are based.

> **Reviewers:** you do not need a GPU, the datasets, or several hours to
> check the paper's arithmetic. One command reproduces every headline number
> from the released result files in a few seconds — see
> [Tier 0](#tier-0--verify-the-papers-numbers-2-minutes-no-gpu) below.

---

## What the study does

Five detectors — a selective state-space (Mamba/S6) model, a Transformer, a
1-D CNN, a bidirectional LSTM and a Random Forest — are compared on three
public NIDS corpora under an architectural parity constraint, and evaluated
under white-box (FGSM, PGD) and black-box transfer attacks over seven random
seeds. The full grid is 945 evaluation records, plus a defence arm and a
capacity ablation.

---

## A note on language

`analysis/` and `scripts/` — everything needed for Tier 0 and Tier 1 — are in
English, as are all printed outputs, figure labels and result files. The three
Colab notebooks carry **Turkish comments and markdown**, since they were
written for the authors' own workflow. Their code, variable names and console
output are English. If the editor prefers fully English notebooks we will
supply translated versions on request; the scripts in `scripts/` are
line-for-line equivalents of the verification notebooks and require no
Turkish.

## Repository layout

```
.
├── notebooks/
│   ├── 01_main_experiment.ipynb   Full pipeline: preprocessing, training,
│   │                              attacks, defence, ablation, tables, figures
│   ├── 02_verification.ipynb      S6 implementation validation, split
│   │                              verification, single-threshold leakage audit
│   ├── 03_ttl_ablation.ipynb      NF-UNSW-NB15 feature-removal ablation
│   └── 04_padding_mask.ipynb      Re-runs the white-box grid with the
│                                  perturbation confined to the informative
│                                  coordinates (padding held at zero)
├── scripts/                       Standalone equivalents of the verification
│   ├── validate_s6.py             notebooks, for running outside Colab
│   ├── verify_splits.py
│   ├── check_leakage.py
│   └── ttl_ablation.py
├── analysis/
│   ├── reproduce_claims.py        Recompute every headline number (no GPU)
│   ├── check_published_artifacts.py  Regenerate tables/figures and diff them
│   │                              against the published copies
│   ├── gen_tables.py              Regenerate the LaTeX tables
│   └── make_figures.py            Regenerate the figures
├── supplementary/                 The material the paper refers to as
│   ├── supplementary.tex          released with the artefact: the omitted
│   └── supplementary.pdf          derivations, pseudo-code and extended analyses
├── published/                     The exact tables and figures used in the paper
│   ├── figures/                   20 vector PDFs
│   ├── figures_png/               the same figures as PNG, for browsing on GitHub
│   └── tables_tex/                9 LaTeX table bodies
├── results/                       Outputs of the reported run
│   ├── master_results.csv         945 rows: dataset x model x attack x eps x seed
│   ├── defense_results.csv        Adversarial-training arm
│   ├── ablation_results.csv       Capacity ablation
│   ├── e3_padding_mask.{json,csv,tex}  Padding-masked re-evaluation
│   ├── param_counts.json          Trainable parameters per architecture
│   ├── tables/                    T1-T6 and the post-hoc T2b (csv, md, tex)
│   ├── logs/                      Run configuration and the run's report
│   ├── verification/              S6 validation, split and TTL-ablation output
│   └── superseded/                Pre-correction frame of the first execution
│                                  and its run report; source of no published
│                                  figure (see its README)
├── .gitattributes                 Pins LF line endings on all platforms
├── CITATION.cff                   Citation metadata
├── .zenodo.json                   Zenodo deposition metadata
└── requirements.txt
```

Everything the paper reports is therefore present in one of four forms: the
raw per-run records (`results/*.csv`), the aggregated tables
(`results/tables/`), the rendered artefacts as they appear in the
manuscript (`published/`), and the prose and derivations the paper omits
for length (`supplementary/`). The paper is held to its journal's
fifteen-page limit, so its methodology section cites derivations rather
than restating them and several analyses are reported in summary; the
supplement carries both in full, with the result file behind every
quantity named in place.

The supplement's sections are cited in the paper as S1-S10, so a pointer in
the paper resolves to a section of `supplementary/supplementary.pdf` without
further lookup.

One frame in `results/superseded/` is deliberately not part of that set.
It records the first execution of the grid, in which the `Attack` column of
NF-UNSW-NB15-v2 leaked the label into the feature matrix. No published
number comes from it. It is kept so that the paper's claim about how that
defect imitated a gradient-masking signature can be checked against data
rather than taken on trust.

Not included in this repository, by size: preprocessed tensor caches
(162 MB), model checkpoints (234 MB) and the corpus files actually processed
(156 MB, one of them 110 MB and so above GitHub's hard per-file limit). The
caches and checkpoints are regenerated by `01_main_experiment.ipynb`.

The corpus files require a word of warning. NSL-KDD is used as distributed.
The NF-UNSW-NB15-v2 and CSE-CIC-IDS2018 files are **not** the full public
releases but class-rebalanced subsamples: 139,264 flows in an exactly equal
benign/attack split for the former, against a public release of 2,390,275
flows at a 3.98% attack share; and 313,648 records for the latter, obtained by
capping each class independently. Re-running the pipeline against the full
public releases will therefore not reproduce the reported numbers. The two
subsample files are deposited alongside this repository in the Zenodo record
(`datasets-subsamples.zip`) so that the figures remain reproducible.

---

## Reproduction, in three tiers of effort

### Tier 0 — verify the paper's numbers (2 minutes, no GPU)

```bash
pip install -r requirements.txt
python analysis/reproduce_claims.py
```

Recomputes 58 headline claims from `results/` and prints computed value
against value as printed in the manuscript. Exit status is non-zero if any
row reads `MISMATCH`. Expected output ends with:

```
58 claims checked, 0 mismatch(es).
```

The published tables and figures are committed under `published/`, so they
can be inspected without running anything. To check that *this* code actually
produces *those* artefacts:

```bash
python analysis/check_published_artifacts.py
```

This regenerates everything from `results/` and diffs it against
`published/`. The figure comparison additionally needs `pdftotext` and
`pdftoppm` from poppler-utils, which pip cannot install; without them the
figures report `SKIP` and only the table comparison runs. LaTeX table bodies are compared byte-for-byte; figures are
compared on extracted content, since PDF writers stamp a fresh timestamp and
document ID on every run. Expected output ends with:

```
29 artefacts checked. Table bodies: 0 not reproduced. Figures: 0 flagged for review.
```

To regenerate without diffing:

```bash
python analysis/gen_tables.py     # -> build/tab_T*.tex
python analysis/make_figures.py   # -> build/figures/*.pdf
```

### Tier 1 — verify the implementation and the data handling (~15 minutes, no GPU)

`check_published_artifacts.py` regenerates the nine LaTeX table bodies and the
twenty figures from `results/` and compares them with the published copies.
Table bodies are compared byte-for-byte and must match exactly. Figures are
compared by rasterising both copies and measuring the mean grey-level
deviation, because matplotlib and poppler differ between releases in
antialiasing and in mathtext glyph embedding: the published figures sit
between 0.6 and 6.1 out of 255 against a regeneration on a different
toolchain, with no plotted value changed. Only the table comparison gates the
exit status; a figure above tolerance is reported for inspection rather than
treated as a failure.

```bash
python scripts/validate_s6.py      # is the S6 reimplementation correct?
python scripts/verify_splits.py    # are the train/test partitions disjoint?
python scripts/check_leakage.py    # single-threshold separability audit
python scripts/ttl_ablation.py     # feature-removal ablation (~8 min)
```

The last three need the raw CSVs; set the path at the top of each script.
`validate_s6.py` needs nothing but PyTorch.

`validate_s6.py` performs three independent checks and is the one we consider
decisive, because the state-space detector is the only component not supplied
by a standard library:

| Check | What it compares | Reported in the paper |
|---|---|---|
| 1 | Our scan against the official `selective_scan_ref` | max abs. deviation 1.86e-08 (fp32) |
| 2 | Our sequential loop against a closed-form O(L^2) solution | 1.39e-17 (fp64) |
| 3 | Backward pass against finite differences | passes |

Check 1 does **not** require the `mamba-ssm` package to build: the reference
function is pure PyTorch and the script extracts it from the official source
if the package is unavailable.

### Tier 2 — rerun the full experiment (GPU, several hours)

Open `notebooks/01_main_experiment.ipynb` in Colab or Jupyter with an
NVIDIA GPU, place the datasets as described below, and run all cells. Set
`SMOKE_TEST = True` first for a reduced-scale end-to-end check.

The pipeline is resumable: completed (dataset, model, seed) jobs are detected
and skipped, and intermediate results are flushed after each model.

---

## Datasets

Public; not redistributed here.

| Corpus | Source |
|---|---|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html |
| NF-UNSW-NB15-v2 | https://staff.itee.uq.edu.au/marius/NIDS_datasets/ |
| CSE-CIC-IDS2018 | https://www.unb.ca/cic/datasets/ids-2018.html |

Expected layout:

```
<project>/datasets/NSL-KDD/{KDDTrain+.txt, KDDTest+.txt}
<project>/datasets/NF-UNSW-NB15/NF-UNSW-NB15-v2.csv
<project>/datasets/CICIDS2018/CICIDS2018_sample.csv
```

> **Note.** More than one copy of `NF-UNSW-NB15-v2.csv` may exist on a given
> machine (the full release and the subsample used here). The notebooks print
> every copy they find, flag the one selected, and abort the analysis if the
> record count does not match the 139,264 reported in the paper. Please keep
> this guard enabled.

---

## Credentials

The main notebook contains an optional e-mail notification block for long
runs. It reads credentials from Colab Secrets at run time
(`userdata.get(...)`); **no credential, token or personal address is stored in
this repository** — the address in the documentation is the placeholder
`sizinadresiniz@gmail.com`. The block can be deleted without affecting any
result.

## Environment

Experiments were run on Google Colab. The first execution of the grid used
an NVIDIA A100-SXM4-80 GB; the corrected NF-UNSW-NB15 execution and the
padding-masked re-evaluation used an A100-SXM4-40 GB. Each run's device is
recorded in its own report (`results/logs/final_report.md` and
`results/superseded/final_report_20260710_193652.md`) and, for the
padding-mask arm, in `results/e3_padding_mask.json`. `results/logs/config.json`
records the resolved run configuration; exact library versions are not
serialised there and appear only in the first cell's output of the main
notebook.

```
python >= 3.10
torch, numpy, pandas, scikit-learn, scipy, matplotlib, einops
```

`mamba-ssm` is optional and used only for one validation check.

---

## Determinism

All stochastic sources are seeded per run (seeds 13, 42, 123, 777, 2024,
2025, 31415) and the resolved configuration is serialised before training.
cuDNN autotuning and TF32 reduction are enabled for throughput, and GPU
atomic accumulation is not order-deterministic; results are therefore
reproducible **in distribution** across seeds rather than bitwise identical
across runs. Seed-level dispersion is reported alongside every mean.

---

## Known limitations of this artefact

Stated so that reviewers do not have to discover them:

1. Robustness is evaluated with FGSM, PGD and surrogate transfer. Attack
   ensembles resistant to gradient masking (e.g. AutoAttack) were not run;
   the manuscript reports the corresponding self-audit and treats the
   white-box ranking as an upper bound.
2. The adversarially trained checkpoints were not evaluated under transfer
   attack.
3. Perturbations are bounded in standardised feature space and are not
   constrained to correspond to emittable network flows.
4. The Random Forest baseline is excluded from the attack stage, which is
   gated on differentiability.
5. `CSE-CIC-IDS2018` has 79 features and the fixed pseudo-sequence length of
   64 truncates 15 of them; the corpus is retained as a declared negative
   control.

---

## Citing this artefact

The repository is archived on Zenodo, which mints a DOI for each tagged
release. Please cite the archived release rather than the GitHub URL, since
only the former is immutable:

```
Kilic, H. Adversarial Robustness of Mamba-Based Network Intrusion Detection:
Code and Results. Zenodo. https://doi.org/10.5281/zenodo.23497561
```

That is the concept DOI: it is stable across releases and always resolves to
the most recent one. Cite it unless you need to pin a specific release, in
which case use that release's own version DOI, which Zenodo lists under
*Versions* on the record page.

The corpus files the reported run consumed are deposited as a companion
record, <https://doi.org/10.5281/zenodo.23499680>, linked from this one under
*Related works*. Two of the four are not the full public releases, so
reproducing the published numbers requires those files rather than the
upstream distributions.

`CITATION.cff` carries the same metadata in machine-readable form, and
`.zenodo.json` supplies the deposition fields Zenodo reads when it imports a
release.

---

## License

Code is released under the MIT License (see `LICENSE`). The datasets remain
subject to the licences of their original providers.
