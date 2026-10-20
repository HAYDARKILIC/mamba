#!/usr/bin/env python3
"""
FOLLOWUP_ttl_ablation.py — the one experiment the leakage finding still needs.

The verification run showed that MAX_TTL, MIN_TTL and MIN_IP_PKT_LEN each
separate NF-UNSW-NB15 at >0.99 accuracy with a SINGLE threshold. Removing any
one of them changes nothing, because they are mutually redundant. The
informative ablation removes the GROUP, and evaluates on the pipeline's own
group-aware split rather than in-sample.

Run in Colab after Cell 6 of the main notebook (or standalone with BASE_DIR set).
Runtime: ~3 minutes on CPU.

Report the printed table in Section 5.7 of the manuscript.
"""
import glob, os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = ""            # bos birakirsaniz aranir
DROP_SETS = {
    "all features": [],
    "without MAX_TTL": ["MAX_TTL"],
    "without both TTL": ["MAX_TTL", "MIN_TTL"],
    "without TTL + MIN_IP_PKT_LEN": ["MAX_TTL", "MIN_TTL", "MIN_IP_PKT_LEN"],
}

def find_csv():
    roots = [BASE_DIR] if BASE_DIR else [
        "/content/drive/MyDrive", "/content/drive/My Drive", "/content", os.getcwd()]
    for r in roots:
        if not os.path.isdir(r):
            continue
        hits = sorted(glob.glob(os.path.join(r, "**", "NF-UNSW-NB15*.csv"), recursive=True))
        if hits:
            return hits[0]
    raise SystemExit("NF-UNSW-NB15 CSV bulunamadi; BASE_DIR degiskenini doldurun.")

path = find_csv()
print("veri:", path)
df = pd.read_csv(path, low_memory=False)
df.columns = [c.strip() for c in df.columns]
lab = "Label" if "Label" in df.columns else "label"
atk = "Attack" if "Attack" in df.columns else lab

groups = (df["IPV4_SRC_ADDR"].astype(str) + "__" + df["IPV4_DST_ADDR"].astype(str))
y = df[lab].astype(int)
X = df.drop(columns=[c for c in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", lab, atk]
                     if c in df.columns], errors="ignore")
for c in X.select_dtypes(include=["object", "string"]).columns:
    X[c] = pd.factorize(X[c].astype(str))[0]
X = X.replace([np.inf, -np.inf], np.nan)
keep = X.notna().all(axis=1)
X, y, groups = X[keep].astype(np.float32), y[keep].values, groups[keep]

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                random_state=42).split(X, groups=groups))
print(f"train={len(tr):,}  test={len(te):,}  "
      f"(grup-farkindali bolme, makaledeki ile ayni tohum)\n")

print(f"{'yapilandirma':32s} {'ozellik':>8s} {'tutulan-set dogrulugu':>22s}")
print("-" * 66)
for name, drop in DROP_SETS.items():
    cols = [c for c in X.columns if c not in drop]
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42,
                                class_weight="balanced")
    rf.fit(X.iloc[tr][cols], y[tr])
    acc = rf.score(X.iloc[te][cols], y[te])
    print(f"{name:32s} {len(cols):8d} {acc:22.4f}")

print("\nYORUM: dogruluk ancak TTL grubu BIRLIKTE cikarildiginda anlamli olarak")
print("duserse, saturasyonun bu artefakttan kaynaklandigi kanitlanmis olur.")
