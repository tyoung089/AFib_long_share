"""Stage 4: bootstrap evaluation of test-set predictions."""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from afib_long.evaluation import bootstrap_metrics, paired_auroc_difference, youden_threshold  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, help="test-set predictions CSV")
    ap.add_argument("--threshold", type=float, help="operating threshold, for example from the model file")
    ap.add_argument("--val-predictions", help="validation predictions CSV, to set the threshold by Youden's J")
    ap.add_argument("--compare", help="predictions CSV of a second model for the same index ECGs")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-paired", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions, dtype={"ecg_id": str})
    threshold = args.threshold
    if threshold is None and args.val_predictions:
        val = pd.read_csv(args.val_predictions)
        threshold = youden_threshold(val["label"], val["prob"])
    if args.compare:
        other = pd.read_csv(args.compare, dtype={"ecg_id": str})[["patient_id", "ecg_id", "prob"]]
        df = df.merge(other.rename(columns={"prob": "prob_other"}), on=["patient_id", "ecg_id"], validate="1:1")
    if threshold is not None:
        print(f"operating threshold {threshold:.4f}")

    subsets = [("All index ECGs", df)]
    if "history_length" in df:
        subsets.append(("Index ECGs with >= 1 prior ECG", df[df["history_length"] >= 2]))
    for name, d in subsets:
        d = d.reset_index(drop=True)
        print(f"\n{name}: {d['patient_id'].nunique()} patients, {len(d)} index ECGs, {int(d['label'].sum())} with AF")
        for metric, (mean, lo, hi) in bootstrap_metrics(d, "prob", threshold, args.n_boot, args.seed).items():
            print(f"  {metric:<12s} {mean:.4f} [{lo:.4f}, {hi:.4f}]")
        if args.compare:
            mean, lo, hi, p = paired_auroc_difference(d, "prob", "prob_other", args.n_boot_paired, args.seed)
            print(f"  paired AUROC difference {mean:+.4f} [{lo:+.4f}, {hi:+.4f}], P = {p:.4f}")


if __name__ == "__main__":
    main()
