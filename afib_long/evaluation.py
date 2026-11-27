"""Patient-cluster bootstrap evaluation with one random index ECG per patient."""
import numpy as np
import pandas as pd
from numpy.random import SeedSequence, default_rng
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(np.asarray(y, int), np.asarray(p, float))
    return float(thr[int(np.argmax(tpr - fpr))])


def maxpool_auroc(y, p, patient_id):
    """Patient-level AUROC of each patient's highest-scoring ECG (used for model selection in training)."""
    d = pd.DataFrame({"y": y, "p": p, "pid": patient_id})
    top = d.loc[d.groupby("pid", sort=False)["p"].idxmax()]
    return float(roc_auc_score(top["y"], top["p"])) if top["y"].nunique() == 2 else float("nan")


def _draws(df, n_boot, seed):
    """Yields (row positions in df, labels) of each iteration's one-ECG-per-patient sample."""
    order = np.lexsort((df["ecg_id"].to_numpy(dtype=str), df["patient_id"].to_numpy()))
    _, start, length = np.unique(df["patient_id"].to_numpy()[order], return_index=True, return_counts=True)
    y = df["label"].to_numpy().astype(int)[order]
    n = len(start)
    for ss in SeedSequence(seed).spawn(n_boot):
        rng = default_rng(ss)
        pick = rng.integers(0, n, size=n)
        rows = start[pick] + (rng.random(n) * length[pick]).astype(np.int64)
        if y[rows].min() != y[rows].max():
            yield order[rows], y[rows]


def _summary(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def bootstrap_metrics(df, prob_col="prob", threshold=None, n_boot=1000, seed=42):
    """df columns: patient_id, ecg_id, label, prob_col. Returns {metric: (mean, lo, hi)}."""
    out = {k: [] for k in ("AUROC", "AUPRC", "Sensitivity", "Specificity", "F1", "Accuracy")}
    p_all = df[prob_col].to_numpy(float)
    for rows, y in _draws(df, n_boot, seed):
        p = p_all[rows]
        out["AUROC"].append(roc_auc_score(y, p))
        out["AUPRC"].append(average_precision_score(y, p))
        if threshold is not None:
            pred = p >= threshold
            tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
            fn, tn = np.sum(~pred & (y == 1)), np.sum(~pred & (y == 0))
            out["Sensitivity"].append(tp / (tp + fn))
            out["Specificity"].append(tn / (tn + fp))
            out["F1"].append(2 * tp / (2 * tp + fp + fn))
            out["Accuracy"].append((tp + tn) / len(y))
    return {k: _summary(v) for k, v in out.items() if v}


def paired_auroc_difference(df, col_a, col_b, n_boot=10000, seed=42):
    """AUROC(col_a) - AUROC(col_b) on the same draws. Returns (mean, lo, hi, two-sided P)."""
    a_all, b_all = df[col_a].to_numpy(float), df[col_b].to_numpy(float)
    delta = []
    for rows, y in _draws(df, n_boot, seed):
        delta.append(roc_auc_score(y, a_all[rows]) - roc_auc_score(y, b_all[rows]))
    delta = np.asarray(delta)
    p = min(1.0, max(2 * min((delta <= 0).mean(), (delta >= 0).mean()), 1 / len(delta)))
    return (*_summary(delta), float(p))
