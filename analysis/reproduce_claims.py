"""Recompute every headline number reported in the paper, directly from the
released result frames, and compare it against the value as printed.

This is the fastest way for a reviewer to check the paper's arithmetic: it
needs no GPU, no datasets and no training -- only the CSV/JSON files in
results/. Runtime is a few seconds.

    python analysis/reproduce_claims.py

Any row marked MISMATCH is a discrepancy between the manuscript and the data.
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
claim("4.3", "Transformer, mean robust acc", adv[adv.model == "Transformer"].accuracy.mean(), 0.7502)
claim("4.3", "Mamba, mean robust acc", adv[adv.model == "Mamba"].accuracy.mean(), 0.6049)
claim("4.3", "CNN1D, mean robust acc", adv[adv.model == "CNN1D"].accuracy.mean(), 0.5425)
claim("4.3", "BiLSTM, mean robust acc", adv[adv.model == "LSTM"].accuracy.mean(), 0.4751)

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
claim("4.4", "max ASR, Mamba (transfer)", asr[asr.model == "Mamba"].asr.max(), 0.0194)
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
claim("4.7", "Mamba latency b=1 (ms)", lat.loc["Mamba", "latency_ms_bs1"], 26.2587)
claim("4.7", "Mamba/fastest latency ratio",
      lat.loc["Mamba", "latency_ms_bs1"] / lat["latency_ms_bs1"].min(), 51.29, 2)

# ---- Section 4.8 defence ----
wb = defense[defense.attack.isin(["FGSM", "PGD"])].groupby(["model", "defense"])["accuracy"].mean()
claim("4.8", "Mamba defence gain", wb[("Mamba", "adv_train")] - wb[("Mamba", "none")], 0.1892)
claim("4.8", "CNN1D defence gain", wb[("CNN1D", "adv_train")] - wb[("CNN1D", "none")], 0.2226)

# ---- Section 3.3 implementation validation ----
claim("3.3", "CHECK 1 deviation (1e-8 units)", ver["results"]["check1"] * 1e8, 1.86, 2)
claim("3.3", "CHECK 2 deviation (1e-17 units)", ver["results"]["check2"] * 1e17, 1.39, 2)

# ---- Section 4.1 / 5.7 split verification and TTL ----
claim("4.1", "NF-UNSW host-pair groups",
      ver["results"]["splits"]["NF-UNSW-NB15"]["n_groups"], 242, 0)
L = ver["results"]["leakage"]
claim("5.7", "MAX_TTL single-threshold acc", L["single_split"]["MAX_TTL"], 0.9948)
claim("5.7", "MIN_IP_PKT_LEN single-threshold acc", L["single_split"]["MIN_IP_PKT_LEN"], 0.9916)
a = ttl["ablation"]
claim("5.7", "accuracy, all features", a["tum ozellikler"]["accuracy"], 0.9976)
claim("5.7", "accuracy, three features removed",
      a["TTL + MIN_IP_PKT_LEN haric"]["accuracy"], 0.9975)
claim("5.7", "drop from removal (1e-5 units)",
      (a["tum ozellikler"]["accuracy"] - a["TTL + MIN_IP_PKT_LEN haric"]["accuracy"]) * 1e5, 7.3, 1)
claim("5.7", "majority-class baseline", a["_taban_cizgisi"], 0.5103)
claim("5.7", "SERVER_TCP_FLAGS residual", a["_kalan_tek_esik"]["SERVER_TCP_FLAGS"], 0.8573)

# ---- Section 5.7 padding-mask re-evaluation (E3) ----
with open(os.path.join(RESULTS, "e3_padding_mask.json")) as fh:
    e3 = json.load(fh)
ag = e3["aggregate"]
for m, rep in [("Transformer", 0.0036), ("CNN1D", 0.1263),
               ("Mamba", 0.1308), ("LSTM", 0.2645)]:
    claim("5.7", f"padding-mask delta, {m}", ag[m]["delta"], rep)
margin_unmasked = ag["Transformer"]["unmasked"] - ag["Mamba"]["unmasked"]
margin_masked = ag["Transformer"]["masked"] - ag["Mamba"]["masked"]
claim("5.7", "headline margin, unmasked", margin_unmasked, 0.2259)
claim("5.7", "headline margin, masked", margin_masked, 0.0987)
claim("5.7", "reduction in the margin (%)",
      100 * (1 - margin_masked / margin_unmasked), 56, 0)

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
