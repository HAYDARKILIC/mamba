"""Recompute every headline number reported in the paper, directly from the
released result frames, and compare it against the value as printed.

This is the fastest way for a reviewer to check the paper's arithmetic: it
needs no GPU, no datasets and no training -- only the CSV/JSON files in
results/. Runtime is a few seconds.

    python analysis/reproduce_claims.py

Any row marked MISMATCH is a discrepancy between the manuscript and the data.

The reported values correspond to the post-leakage-fix run 20261021_031658
(main grid) and 20261021_190939 (the padding-mask re-evaluation). The earlier
run 20260710_193652 carried a target leak in the NF-UNSW-NB15 preprocessing:
the `Attack` column reached the feature matrix and determined the label
exactly. NSL-KDD and CSE-CIC-IDS2018 were unaffected and their rows are
bit-identical across the two runs.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(_HERE), "results")

master = pd.read_csv(os.path.join(RESULTS, "master_results.csv"))
defense = pd.read_csv(os.path.join(RESULTS, "defense_results.csv"))
ablation = pd.read_csv(os.path.join(RESULTS, "ablation_results.csv"))
t4 = pd.read_csv(os.path.join(RESULTS, "tables", "T4_cikarim_verimliligi.csv"))
ver = json.load(open(os.path.join(RESULTS, "verification", "verification.json")))
ttl = json.load(open(os.path.join(RESULTS, "verification", "ttl_ablation.json")))

rows = []
def claim(section, text, computed, reported, dp=4):
    ok = round(float(computed), dp) == round(float(reported), dp)
    rows.append((ok, section, text, float(computed), float(reported)))

adv = master[master.attack != "clean"]
clean = master[master.attack == "clean"].groupby(["dataset", "model"])["accuracy"].mean()

# ---- Section 4.1 scope ----
claim("4.1", "evaluation records", len(master), 945, 0)

# ---- Section 4.2 clean performance ----
claim("4.2", "Random Forest, NSL-KDD, clean acc", clean[("NSL-KDD", "RandomForest")], 0.7699)
claim("4.2", "BiLSTM, NSL-KDD, clean acc", clean[("NSL-KDD", "LSTM")], 0.7999)
claim("4.2", "Mamba, CIC-IDS2018, clean AUC",
      master[(master.dataset == "CICIDS2018") & (master.model == "Mamba")
             & (master.attack == "clean")].auc.mean(), 0.4925)

# ---- Section 4.3 white-box ----
claim("4.3", "Transformer, mean robust acc", adv[adv.model == "Transformer"].accuracy.mean(), 0.7450)
claim("4.3", "Mamba, mean robust acc", adv[adv.model == "Mamba"].accuracy.mean(), 0.5868)
claim("4.3", "CNN1D, mean robust acc", adv[adv.model == "CNN1D"].accuracy.mean(), 0.5441)
claim("4.3", "BiLSTM, mean robust acc", adv[adv.model == "LSTM"].accuracy.mean(), 0.5315)

# ---- Section 4.4 black-box ----
# ASR is computed per (dataset, model, seed, eps) and then averaged over
# seeds, which is the convention used throughout the paper (Eq. 14 read
# together with the seed-averaging rule of Section 4.1). Forming the ratio
# from seed-averaged accuracies instead shifts the fourth decimal.
clean_seed = master[master.attack == "clean"].set_index(
    ["dataset", "model", "seed"])["accuracy"]
bb = master[master.attack == "BlackBoxTransfer"].copy()
bb["asr"] = bb.apply(
    lambda r: 1 - r.accuracy / clean_seed[(r.dataset, r.model, r.seed)], axis=1)
asr = bb.groupby(["dataset", "model", "eps"])["asr"].mean().reset_index()
claim("4.4", "max ASR, Mamba (transfer)", asr[asr.model == "Mamba"].asr.max(), 0.1258)
claim("4.4", "max ASR, BiLSTM (transfer)", asr[asr.model == "LSTM"].asr.max(), 0.8352)
claim("4.4", "max ASR, Transformer (transfer)",
      asr[asr.model == "Transformer"].asr.max(), 0.0567)

# ---- Section 4.5 statistics (pre-specified comparator) ----
for ds, atk, dz_rep in [("NSL-KDD", "FGSM", -2.11), ("NSL-KDD", "PGD", -8.16),
                        ("CICIDS2018", "PGD", -4.27)]:
    sub = master[(master.dataset == ds) & (master.attack == atk)]
    m = sub[sub.model == "Mamba"].groupby("seed")["accuracy"].mean()
    t = sub[sub.model == "Transformer"].groupby("seed")["accuracy"].mean()
    d = m - t
    claim("4.5", f"Cohen d_z, {ds}/{atk}", d.mean() / d.std(), dz_rep, 2)

# ---- Section 4.7 efficiency ----
lat = t4.set_index("model")
claim("4.7", "Mamba latency b=1 (ms)", lat.loc["Mamba", "latency_ms_bs1"], 25.7807)
claim("4.7", "Mamba/fastest latency ratio",
      lat.loc["Mamba", "latency_ms_bs1"] / lat["latency_ms_bs1"].min(), 50.42, 2)

# ---- Section 4.8 defence ----
wb = defense[defense.attack.isin(["FGSM", "PGD"])].groupby(["model", "defense"])["accuracy"].mean()
claim("4.8", "Mamba defence gain", wb[("Mamba", "adv_train")] - wb[("Mamba", "none")], 0.2057)
claim("4.8", "CNN1D defence gain", wb[("CNN1D", "adv_train")] - wb[("CNN1D", "none")], 0.2185)

# ---- Section 3.3 implementation validation ----
claim("3.3", "CHECK 1 deviation (1e-8 units)", ver["results"]["check1"] * 1e8, 1.86, 2)
claim("3.3", "CHECK 2 deviation (1e-17 units)", ver["results"]["check2"] * 1e17, 1.39, 2)

# ---- Section 4.1 / 5.4 split verification and TTL ----
claim("4.1", "NF-UNSW host-pair groups",
      ver["results"]["splits"]["NF-UNSW-NB15"]["n_groups"], 242, 0)
L = ver["results"]["leakage"]
claim("5.4", "MAX_TTL single-threshold acc", L["single_split"]["MAX_TTL"], 0.9948)
claim("5.4", "MIN_IP_PKT_LEN single-threshold acc", L["single_split"]["MIN_IP_PKT_LEN"], 0.9916)
a = ttl["ablation"]
claim("5.4", "accuracy, all features", a["tum ozellikler"]["accuracy"], 0.9976)
claim("5.4", "accuracy, three features removed",
      a["TTL + MIN_IP_PKT_LEN haric"]["accuracy"], 0.9975)
claim("5.4", "drop from removal (1e-5 units)",
      (a["tum ozellikler"]["accuracy"] - a["TTL + MIN_IP_PKT_LEN haric"]["accuracy"]) * 1e5, 7.3, 1)
claim("5.4", "majority-class baseline", a["_taban_cizgisi"], 0.5103)
claim("5.4", "SERVER_TCP_FLAGS residual", a["_kalan_tek_esik"]["SERVER_TCP_FLAGS"], 0.8573)

# ---- Section 5.4 padding-mask re-evaluation (E3) ----
with open(os.path.join(RESULTS, "e3_padding_mask.json")) as fh:
    e3 = json.load(fh)
ag = e3["aggregate"]
for m, rep in [("Transformer", 0.0044), ("CNN1D", 0.0892),
               ("Mamba", 0.0652), ("LSTM", 0.1592)]:
    claim("5.4", f"padding-mask delta, {m}", ag[m]["delta"], rep)
margin_unmasked = ag["Transformer"]["unmasked"] - ag["Mamba"]["unmasked"]
margin_masked = ag["Transformer"]["masked"] - ag["Mamba"]["masked"]
claim("5.4", "headline margin, unmasked", margin_unmasked, 0.1971)
claim("5.4", "headline margin, masked", margin_masked, 0.1363)
claim("5.4", "reduction in the margin (%)",
      100 * (1 - margin_masked / margin_unmasked), 31, 0)

# ---- Section 4.3 white-box ordering, and its resolution ----
# The revision that followed the leakage fix changed this ordering, which no
# earlier check in this file would have caught: the three claims below are the
# ones that moved, so they are asserted explicitly.
wbx = master[master.attack.isin(["FGSM", "PGD"])]
wb_mean = wbx.groupby("model")["accuracy"].mean()
for mo, rep in [("Transformer", 0.7390), ("LSTM", 0.5613),
                ("Mamba", 0.5462), ("CNN1D", 0.5079)]:
    claim("4.3", f"white-box mean robust acc, {mo}", wb_mean[mo], rep)
claim("5.2", "white-box gap, Transformer - Mamba (pp)",
      100 * (wb_mean["Transformer"] - wb_mean["Mamba"]), 19.3, 1)
# The nominal second place is not resolved by a paired test; the paper says so,
# and the p-value is asserted here so that a future run cannot quietly reverse
# the statement without failing this check.
_a = wbx[wbx.model == "LSTM"].groupby("seed")["accuracy"].mean()
_b = wbx[wbx.model == "Mamba"].groupby("seed")["accuracy"].mean()
claim("4.3", "BiLSTM vs Mamba white-box, paired p",
      stats.ttest_rel(_a, _b).pvalue, 0.3677)

# ---- Section 4.5 significance structure ----
def _holm(pvals):
    pv = np.asarray(pvals, float)
    order = np.argsort(pv)
    run, adj = 0.0, np.empty(len(pv))
    for rank, idx in enumerate(order):
        run = max(run, (len(pv) - rank) * pv[idx])
        adj[idx] = min(run, 1.0)
    return adj

_rows = []
for ds in ["NSL-KDD", "NF-UNSW-NB15", "CICIDS2018"]:
    for setting, atks in [("Clean", ["clean"]), ("FGSM", ["FGSM"]),
                          ("PGD", ["PGD"]), ("Transfer", ["BlackBoxTransfer"])]:
        sub = master[(master.dataset == ds) & (master.attack.isin(atks))]
        mm_ = sub[sub.model == "Mamba"].groupby("seed")["accuracy"].mean()
        tt_ = sub[sub.model == "Transformer"].groupby("seed")["accuracy"].mean()
        t_, p_ = stats.ttest_rel(mm_, tt_)
        _rows.append(dict(dataset=ds, setting=setting, t=t_, p=p_))
_stat = pd.DataFrame(_rows)
_stat["p_holm"] = _holm(_stat.p.values)
claim("4.5", "comparisons significant after Holm",
      int((_stat.p_holm < 0.05).sum()), 5, 0)
claim("4.5", "of those, number under white-box attack",
      int(((_stat.p_holm < 0.05) & _stat.setting.isin(["FGSM", "PGD"])).sum()), 5, 0)
# On clean NF-UNSW-NB15 the sign favours Mamba; the paper reports t = +1.84.
claim("4.5", "t, NF-UNSW-NB15 clean (sign matters)",
      _stat[(_stat.dataset == "NF-UNSW-NB15")
            & (_stat.setting == "Clean")].t.iloc[0], 1.84, 2)

# ---- Section 4.4: how often the detector records the lowest transfer ASR ----
# BiLSTM is excluded: its surrogate is itself (Section 3.6), so its rows are
# not transfer results.
_wins = 0
for ds in ["NSL-KDD", "NF-UNSW-NB15", "CICIDS2018"]:
    for e in [0.03, 0.1]:
        cell = asr[(asr.dataset == ds) & (np.isclose(asr.eps, e))
                   & (asr.model != "LSTM")]
        _wins += cell.loc[cell.asr.idxmin(), "model"] == "Mamba"
claim("4.4", "cells where Mamba has the lowest ASR", _wins, 4, 0)

# ---- Section 5.2: the two data-dependent obfuscated-gradient diagnostics ----
mu_ = master.groupby(["dataset", "model", "attack", "eps"])["accuracy"].mean()
_cells = [(ds, mo, e) for ds in ["NSL-KDD", "NF-UNSW-NB15", "CICIDS2018"]
          for mo in ["Mamba", "Transformer", "CNN1D", "LSTM"]
          for e in [0.01, 0.03, 0.05, 0.1]]
claim("5.2", "diagnostic 1: cells where PGD is weaker than FGSM",
      sum(mu_[(ds, mo, "PGD", e)] > mu_[(ds, mo, "FGSM", e)] for ds, mo, e in _cells),
      5, 0)
# Diagnostic 5 fires when a tenfold budget increase barely moves accuracy. The
# flattest curve in the grid loses this much, which is why it no longer fires.
claim("5.2", "diagnostic 5: flattest response over the budget sweep",
      min(mu_[(ds, mo, a, 0.01)] - mu_[(ds, mo, a, 0.1)]
          for ds in ["NSL-KDD", "NF-UNSW-NB15", "CICIDS2018"]
          for mo in ["Mamba", "Transformer", "CNN1D", "LSTM"]
          for a in ["FGSM", "PGD"]),
      0.0457)

# ---- Section 3.6: the surrogate for the recurrent target is that target ----
# The paper excludes the BiLSTM rows from every transfer conclusion because the
# surrogate resolves to the same cached checkpoint as the target. Two facts
# support that: its transfer ASR is an order of magnitude above the other three
# targets', and at eps = 0.03 it tracks its own white-box PGD ASR. The residual
# difference is the step size (eps/4 = 0.0075 against alpha = 0.01).
_D = ["NSL-KDD", "NF-UNSW-NB15", "CICIDS2018"]
def _asr(ds, mo, atk, eps=None):
    sub = master[(master.dataset == ds) & (master.model == mo)
                 & (master.attack == atk)]
    if eps is not None:
        sub = sub[np.isclose(sub.eps, eps)]
    return float(np.mean([1 - r.accuracy / clean_seed[(r.dataset, r.model, r.seed)]
                          for _, r in sub.iterrows()]))
claim("3.6", "BiLSTM mean transfer ASR",
      np.mean([_asr(d, "LSTM", "BlackBoxTransfer") for d in _D]), 0.4446)
claim("3.6", "mean transfer ASR, other three targets",
      np.mean([_asr(d, m, "BlackBoxTransfer") for d in _D
               for m in ["Mamba", "Transformer", "CNN1D"]]), 0.0422)
for d, tr_rep, pgd_rep in [("NSL-KDD", 0.1932, 0.1970),
                           ("NF-UNSW-NB15", 0.3115, 0.3174),
                           ("CICIDS2018", 0.1403, 0.1416)]:
    claim("3.6", f"BiLSTM transfer ASR @0.03, {d}",
          _asr(d, "LSTM", "BlackBoxTransfer", 0.03), tr_rep)
    claim("3.6", f"BiLSTM white-box PGD ASR @0.03, {d}",
          _asr(d, "LSTM", "PGD", 0.03), pgd_rep)

# ---- Sections 3.6 and 4.1: device agreement and the PGD random start ----
# The padding-mask re-execution ran on the second A100 and recomputed 24
# published quantities before its own arm was read. Deterministic quantities
# agree to numerical noise; PGD additionally carries the unseeded random start.
_val = e3["validity"]
_det = max(abs(v["delta"]) for v in _val if not v["quantity"].startswith("PGD"))
_pgd = max(abs(v["delta"]) for v in _val if v["quantity"].startswith("PGD"))
claim("4.1", "device agreement, clean and FGSM (1e-5 units)", _det * 1e5, 8.9, 1)
claim("3.6", "PGD reproduction bound (1e-3 units)", _pgd * 1e3, 3.4, 1)

# ---- Section 4.9: the pairwise relation across capacities ----
_abl = ablation.groupby(["dataset", "model", "d_model"])["accuracy"].mean()
claim("4.9", "ablation cells where Transformer exceeds Mamba",
      sum(_abl[(ds, "Transformer", dm)] > _abl[(ds, "Mamba", dm)]
          for ds in ["NSL-KDD", "NF-UNSW-NB15", "CICIDS2018"] for dm in [32, 64]),
      4, 0)

# ---- report ----
print(f"{'':4s} {'Sec':5s} {'claim':44s} {'computed':>12s} {'in paper':>10s}")
print("-" * 80)
bad = 0
for ok, sec, text, comp, rep in rows:
    if not ok:
        bad += 1
    print(f"{'OK' if ok else 'MISMATCH':4s} {sec:5s} {text:44s} {comp:12.4f} {rep:10.4f}")
print("-" * 80)
print(f"{len(rows)} claims checked, {bad} mismatch(es).")
raise SystemExit(1 if bad else 0)
