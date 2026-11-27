"""Predict 1-year AF risk from raw 12-lead ECGs with the released weights.

  python scripts/predict.py --table my_ecgs.csv --out predictions.csv
  python scripts/predict.py --demo        # a few synthetic ECGs, to check the installation
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from afib_long.data import LongitudinalDataset, WaveformDataset, build_histories, read_table  # noqa: E402
from afib_long.encoder import load_encoder  # noqa: E402
from afib_long.models import load_longitudinal, predict  # noqa: E402


def make_demo_table(folder, fs=250):
    """Synthetic 10-s ECGs (noisy sinusoids, not physiological) for 3 patients with 1, 3 and 6 ECGs."""
    rng = np.random.default_rng(0)
    t = np.arange(10 * fs) / fs
    rows = []
    for pid, n_ecg, age, sex in [(1, 1, 45, "female"), (2, 3, 63, "male"), (3, 6, 78, "female")]:
        for k in range(n_ecg):
            ecg = rng.normal(1, 0.3, (12, 1)) * np.sin(2 * np.pi * 1.2 * t) ** 15 + rng.normal(0, 0.05, (12, len(t)))
            path = os.path.join(folder, f"patient{pid}_ecg{k}.npy")
            np.save(path, ecg.astype(np.float32))
            when = pd.Timestamp("2020-01-01") + pd.Timedelta(days=200 * k)
            rows.append(dict(patient_id=pid, ecg_id=f"p{pid}_e{k}", acquisition_datetime=when,
                             waveform_path=path, age=age - 2 * (n_ecg - 1 - k), sex=sex))
    table = os.path.join(folder, "demo_ecgs.csv")
    pd.DataFrame(rows).to_csv(table, index=False)
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", help="ECG table (see README)")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--fs", type=int, default=250, help="sampling rate of the raw waveforms (Hz)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.demo:
        demo_dir = tempfile.mkdtemp(prefix="afib_long_demo_")
        args.table = make_demo_table(demo_dir, args.fs)

    df = read_table(args.table)
    encoder = load_encoder(os.path.join(args.weights_dir, "encoder_ecgfm_finetuned.pt"), device)[0].eval()
    model, config = load_longitudinal(os.path.join(args.weights_dir, "longitudinal_model.pt"), device)

    emb = []  # one 768-d embedding per ECG: mean of its two 5-s window embeddings
    loader = DataLoader(WaveformDataset(df, np.arange(len(df)), fs=args.fs), batch_size=args.batch_size)
    with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
        for x, _ in loader:
            emb.append(encoder.embed_ecg(x.to(device)).float().cpu().numpy())
    emb = np.concatenate(emb).mean(axis=1)

    histories = build_histories(df, config["max_history"])
    prob = predict(model, LongitudinalDataset(df, histories, emb, np.arange(len(df))), device=device)
    out = pd.DataFrame({"patient_id": df["patient_id"], "ecg_id": df["ecg_id"],
                        "history_length": [len(h) for h in histories[0]], "prob": prob,
                        "high_risk": (prob >= config["threshold"]).astype(int)})
    if "label" in df:
        out.insert(2, "label", df["label"])
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False) if args.demo else f"wrote {len(out)} predictions to {args.out}")
    if args.demo:
        print("These ECGs are synthetic, so the risks have no clinical meaning. They only show that the pipeline runs.")
        shutil.rmtree(demo_dir)


if __name__ == "__main__":
    main()
