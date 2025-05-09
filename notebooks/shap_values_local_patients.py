#%%
"""
print_shap_scores_no_types.py
-----------------------------
• Prints the time-summed SHAP contribution of every feature
  for patients 8, 661 and 941 in each dataset JSON.
• Prints the mean of those sums across all patients (global importance).

Adjust DATASET_FILES, PATIENT_IDX or LABEL_KEY to suit your setup.
"""

import json
import numpy as np
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────── #
DATASET_FILES = {
    "combined_tudd": Path("SHAP_scores/multi_channel_lstm_static_combined_tudd_1000.json"),
    "mimic_tudd"   : Path("SHAP_scores/multi_channel_lstm_static_mimic_tudd_1000.json"),
    "tudd_tudd"    : Path("SHAP_scores/multi_channel_lstm_static_tudd_tudd_1000.json"),
}

PATIENT_IDX = [8, 661, 941]   # 0-based indices
LABEL_KEY   = "label_1"       # choose "label_0" or "label_1"
# ───────────────────────────────────────────────────────────────────────────── #

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_feature_names(meta, key, n, prefix):
    """Return feature names, falling back to generic names if absent."""
    return meta.get(key, [f"{prefix}{i}" for i in range(n)])

def print_patient_sums(names, seq_sum, static_vals, patient):
    combined = np.concatenate([seq_sum, static_vals])
    print(f"\n  Patient {patient}:")
    for name, val in zip(names, combined):
        print(f"    {name:30s}: {val:+.6f}")

def print_mean_sums(names, mean_combined):
    print("\n  Mean summed SHAP over ALL patients:")
    for name, val in zip(names, mean_combined):
        print(f"    {name:30s}: {val:+.6f}")


for ds_name, path in DATASET_FILES.items():
    print(f"\n==================  {ds_name.upper()}  ==================")
    data = load_json(path)

    # SHAP arrays
    shap_seq   = np.asarray(data["shap_values"][LABEL_KEY]["sequential"])  # (N, T, F_seq)
    shap_static = np.asarray(data["shap_values"][LABEL_KEY]["static"])     # (N, F_static)

    n_samples, _, n_seq_feat = shap_seq.shape
    n_static_feat = shap_static.shape[1]

    meta = data.get("metadata", {})
    seq_names    = get_feature_names(meta, "sequential_feature_names", n_seq_feat,  "seq_feat_")
    static_names = get_feature_names(meta, "static_feature_names",     n_static_feat, "static_feat_")
    all_names    = seq_names + static_names

    # Per-patient output
    for patient in PATIENT_IDX:
        if patient >= n_samples:
            print(f"\n  ! Patient index {patient} out of range (dataset has {n_samples} samples) — skipped.")
            continue
        seq_sum    = shap_seq[patient].sum(axis=0)   # Σ over time
        static_val = shap_static[patient]
        print_patient_sums(all_names, seq_sum, static_val, patient)

    # Mean across all patients
    seq_sum_all  = shap_seq.sum(axis=1)              # (N, F_seq)
    mean_seq     = seq_sum_all.mean(axis=0)          # (F_seq,)
    mean_static  = shap_static.mean(axis=0)          # (F_static,)
    mean_combined = np.concatenate([mean_seq, mean_static])
    print_mean_sums(all_names, mean_combined)


# %%
