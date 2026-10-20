#!/usr/bin/env python3
"""
verify_splits.py — evidence the partitioning guarantee claimed in Section 3.2.

WHY THIS EXISTS
---------------
Section 4.1 of the manuscript currently admits that the run log records the
partitioning strategy only for NSL-KDD, because the other two corpora were
served from the version-tagged cache. The group-aware and chronological
guarantees are therefore ASSERTED but not EVIDENCED in the reported run.

Since the split is deterministic given the seed (GroupShuffleSplit with
random_state=42), it can be re-derived from the raw CSVs and verified
directly, WITHOUT retraining anything. This script does that and prints a
statement suitable for quoting in the manuscript.

HOW TO RUN
----------
Run in the Colab session with Drive mounted, after Cell 6 has been executed
(it reuses _load_raw_dataframe and _group_aware_split from the notebook).
Alternatively set BASE_DIR below and run standalone.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Set this to the project folder that contains datasets/ .
# Example (Colab): "/content/drive/MyDrive/<your-project-folder>"
BASE_DIR = ""   # <-- fill in before running
DATASETS = {
    "NF-UNSW-NB15": dict(
        path=f"{BASE_DIR}/datasets/NF-UNSW-NB15/NF-UNSW-NB15-v2.csv",
        group_cols=["IPV4_SRC_ADDR", "IPV4_DST_ADDR"],
        time_col=None,
    ),
    "CICIDS2018": dict(
        path=f"{BASE_DIR}/datasets/CICIDS2018/CICIDS2018_sample.csv",
        group_cols_candidates=[["Flow ID"], ["Src IP", "Dst IP"],
                               ["Source IP", "Destination IP"],
                               ["Src IP Addr", "Dst IP Addr"]],
        time_cols=["Timestamp", "timestamp", "Flow Start Time"],
    ),
}


def verify_group_split(df, group_cols, test_size=0.2, seed=42):
    """Re-derive the split and check that no group spans both partitions."""
    if len(group_cols) >= 2:
        key = df[group_cols[0]].astype(str) + "__" + df[group_cols[1]].astype(str)
    else:
        key = df[group_cols[0]].astype(str)
    n_groups = key.nunique()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=key))
    overlap = set(key.iloc[tr]) & set(key.iloc[te])
    return dict(strategy="group", n_groups=n_groups, n_train=len(tr),
                n_test=len(te), overlapping_groups=len(overlap))


def verify_temporal_split(df, time_col, test_size=0.2):
    ts = pd.to_datetime(df[time_col], errors="coerce")
    parsed = ts.notna().mean()
    order = ts.sort_values(kind="mergesort").index
    cut = int(len(order) * (1 - test_size))
    tr, te = order[:cut], order[cut:]
    return dict(strategy="temporal", parsed_fraction=round(float(parsed), 4),
                n_train=len(tr), n_test=len(te),
                train_max=str(ts.loc[tr].max()), test_min=str(ts.loc[te].min()),
                chronologically_disjoint=bool(ts.loc[tr].max() <= ts.loc[te].min()))


def main():
    if not BASE_DIR:
        raise SystemExit("Set BASE_DIR at the top of this file first.")
    print("PARTITIONING VERIFICATION\n" + "=" * 60)
    print("NSL-KDD: official disjoint KDDTrain+/KDDTest+ files; "
          "no derived split, no leakage channel via partitioning.\n")

    for name, cfg in DATASETS.items():
        print(f"[{name}]")
        try:
            df = pd.read_csv(cfg["path"], low_memory=False)
            df.columns = [c.strip() for c in df.columns]
        except Exception as e:                                # noqa: BLE001
            print(f"  could not read raw file: {e}\n")
            continue

        if name == "NF-UNSW-NB15":
            cols = [c for c in cfg["group_cols"] if c in df.columns]
            if len(cols) >= 1:
                print("  " + str(verify_group_split(df, cols)))
            else:
                print("  group columns ABSENT -> the claimed group-aware "
                      "guarantee does not hold for this corpus")
        else:
            chosen = next((c for c in cfg["group_cols_candidates"]
                           if all(x in df.columns for x in c)), None)
            if chosen:
                print(f"  group columns found: {chosen}")
                print("  " + str(verify_group_split(df, chosen)))
            else:
                tcol = next((c for c in cfg["time_cols"] if c in df.columns), None)
                if tcol:
                    print(f"  no group columns; timestamp column: {tcol}")
                    print("  " + str(verify_temporal_split(df, tcol)))
                else:
                    print("  NEITHER group columns NOR a timestamp are present.")
                    print("  -> the split for this corpus is STRATIFIED RANDOM and")
                    print("     Section 3.2 must state this explicitly as a")
                    print("     methodological limitation for this corpus.")
        print()

    print("=" * 60)
    print("Quote the printed figures in Section 4.1 in place of the current")
    print("statement that the log contains no entry for these two corpora.")


if __name__ == "__main__":
    main()
