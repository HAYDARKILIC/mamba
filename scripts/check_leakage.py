#!/usr/bin/env python3
"""
check_leakage.py — investigate the suspected label-encoding feature in
NF-UNSW-NB15 (Discussion, Section 5.7).

WHY THIS EXISTS
---------------
A 200-tree Random Forest attains 0.999963 accuracy on this corpus with ZERO
variance across all seven seeds, and five architecturally unrelated models
all exceed 0.9977. That pattern is characteristic of a single feature that
encodes the target, not of a hard problem being solved well. The group-aware
split closes the host-pair leakage channel; it does nothing about a leaking
column.

The manuscript currently voices the suspicion without evidence, which is the
worst position to be in at review. This script settles it in three steps:

  STEP 1  Impurity-based feature importances of the Random Forest, named.
  STEP 2  Per-feature single-split separability: for each feature, the best
          achievable accuracy using ONE threshold. A value near 1.0 for any
          single feature is direct proof of leakage.
  STEP 3  Leave-the-suspect-out retraining: if the top feature is removed,
          does performance collapse to a plausible level?

OUTCOME AND WHAT TO WRITE
-------------------------
  * If STEP 2 finds a feature with single-split accuracy > 0.99:
        the corpus is leaking. Name the feature in Section 5.7, state
        whether it is legitimately predictive (e.g. an aggregate the
        exporter derives from the label) or an artefact, and report the
        STEP 3 number. This converts a vulnerability into a contribution.
  * If no single feature exceeds ~0.95 and STEP 3 shows only a small drop:
        the task is genuinely easy under this feature set. Say so, and the
        suspicion is answered rather than left open.

HOW TO RUN
----------
Run in the Colab session with Drive mounted, after Cell 6 has been executed.
Runs on CPU in a few minutes.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set this to the project folder that contains datasets/ .
# Example (Colab): "/content/drive/MyDrive/<your-project-folder>"
BASE_DIR = ""   # <-- fill in before running
CSV = f"{BASE_DIR}/datasets/NF-UNSW-NB15/NF-UNSW-NB15-v2.csv"
SEED = 42
TOP_K = 10


def load():
    """Reproduce the manuscript's preprocessing for this corpus."""
    df = pd.read_csv(CSV, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    label_col = "Label" if "Label" in df.columns else "label"
    y = df[label_col].astype(int)
    attack_col = "Attack" if "Attack" in df.columns else label_col
    drop = [c for c in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", label_col, attack_col]
            if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")
    # integer-encode the string columns exactly as the pipeline does
    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = pd.factorize(X[c].astype(str))[0]
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    return X[keep].astype(np.float32), y[keep].values


def best_single_split_accuracy(x, y, n_thresholds=256):
    """Highest accuracy obtainable from one threshold on one feature."""
    qs = np.unique(np.quantile(x, np.linspace(0.001, 0.999, n_thresholds)))
    best = 0.0
    for t in qs:
        p = (x > t).astype(int)
        best = max(best, accuracy_score(y, p), accuracy_score(y, 1 - p))
    return best


def main():
    if not BASE_DIR:
        raise SystemExit("Set BASE_DIR at the top of this file first.")
    X, y = load()
    print(f"rows={len(X)}  features={X.shape[1]}  positive rate={y.mean():.4f}\n")

    # ---------------- STEP 1 ----------------
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                random_state=SEED, class_weight="balanced")
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("STEP 1 — Random Forest impurity importances (top %d)" % TOP_K)
    for name, v in imp.head(TOP_K).items():
        print(f"    {v:7.4f}  {name}")
    print(f"    top-1 share of total importance mass: {imp.iloc[0]:.4f}")
    print(f"    top-3 share: {imp.head(3).sum():.4f}\n")

    # ---------------- STEP 2 ----------------
    print("STEP 2 — best single-threshold accuracy per feature (top %d by importance)" % TOP_K)
    single = {}
    for name in imp.head(TOP_K).index:
        single[name] = best_single_split_accuracy(X[name].values, y)
        flag = "  <-- LEAKING" if single[name] > 0.99 else ""
        print(f"    {single[name]:7.4f}  {name}{flag}")
    worst = max(single, key=single.get)
    print(f"    maximum single-feature accuracy: {single[worst]:.4f} ({worst})\n")

    # ---------------- STEP 3 ----------------
    suspect = imp.index[0]
    rf2 = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                 random_state=SEED, class_weight="balanced")
    Xd = X.drop(columns=[suspect])
    rf2.fit(Xd, y)
    print(f"STEP 3 — retrained without '{suspect}'")
    print(f"    in-sample accuracy with    it: {rf.score(X, y):.6f}")
    print(f"    in-sample accuracy without it: {rf2.score(Xd, y):.6f}")
    print("\nNOTE: STEP 3 uses in-sample accuracy only as a fast diagnostic of")
    print("whether the signal survives feature removal. For the manuscript,")
    print("re-run the pipeline's own group-aware split with the feature")
    print("dropped and report held-out accuracy.")


if __name__ == "__main__":
    main()
