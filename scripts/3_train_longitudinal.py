"""Stage 3: train the longitudinal model with age and sex."""
import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from afib_long.data import LongitudinalDataset, build_histories, load_embeddings, read_table  # noqa: E402
from afib_long.evaluation import maxpool_auroc, youden_threshold  # noqa: E402
from afib_long.models import LongitudinalModel, predict  # noqa: E402

ARCH = dict(emb_dim=768, d_model=256, d_time=64, n_heads=4, n_layers=2, d_ff=512, dropout=0.3,
            demo_dim=2, demo_hidden=32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--embeddings", required=True, help="output of 2_extract_embeddings.py")
    ap.add_argument("--out-dir", default="runs/longitudinal")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--k-sample", type=int, default=5)
    ap.add_argument("--pos-weight-cap", type=float, default=10.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = read_table(args.table)
    histories = build_histories(df)
    emb = load_embeddings(args.embeddings, df["ecg_id"])
    rows = {s: np.flatnonzero(df["split"] == s) for s in ("train", "val", "test")}
    train_ds = LongitudinalDataset(df, histories, emb, rows["train"], train=True, k_sample=args.k_sample,
                                   seed=args.seed)
    val_ds = LongitudinalDataset(df, histories, emb, rows["val"])
    y = df["label"].to_numpy()[rows["train"]]
    pos_weight = float(min((y == 0).sum() / max((y == 1).sum(), 1), args.pos_weight_cap))

    model = LongitudinalModel(**ARCH).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # Batch norm fails on a final batch of one sample, so drop it only in that case.
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        drop_last=len(train_ds) % args.batch_size == 1)
    y_val, pid_val = df["label"].to_numpy()[rows["val"]], df["patient_id"].to_numpy()[rows["val"]]

    best_auc, best_max_pool, bad, best_state = -1.0, -1.0, 0, None
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        for emb_b, days, mask, age_sex, label in loader:
            optimizer.zero_grad()
            loss = criterion(model(emb_b.to(device), days.to(device), mask.to(device), age_sex.to(device)),
                             label.to(device))
            loss.backward()
            optimizer.step()
        p = predict(model, val_ds, batch_size=args.batch_size, device=device)
        auc, max_pool = roc_auc_score(y_val, p), maxpool_auroc(y_val, p, pid_val)
        print(f"epoch {epoch}: val per-ECG AUROC {auc:.4f}, max-pool AUROC {max_pool:.4f}", flush=True)
        if auc > best_auc:
            best_auc, best_state = auc, copy.deepcopy(model.state_dict())
        if max_pool > best_max_pool:
            best_max_pool, bad = max_pool, 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    preds = {}
    for split in ("val", "test"):
        ds = LongitudinalDataset(df, histories, emb, rows[split])
        r = rows[split]
        preds[split] = pd.DataFrame({"patient_id": df["patient_id"].to_numpy()[r], "ecg_id": df["ecg_id"].to_numpy()[r],
                                     "label": df["label"].to_numpy()[r],
                                     "history_length": [len(histories[0][i]) for i in r],
                                     "prob": predict(model, ds, batch_size=args.batch_size, device=device)})
        preds[split].to_csv(os.path.join(args.out_dir, f"{split}_predictions.csv"), index=False)
    threshold = youden_threshold(preds["val"]["label"], preds["val"]["prob"])
    config = {"architecture": ARCH, "max_history": 20, "threshold": threshold,
              "threshold_rule": "Youden J on the validation set, over all validation index ECGs"}
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"config": config, "state_dict": state}, os.path.join(args.out_dir, "longitudinal_model.pt"))
    print(f"best val per-ECG AUROC {best_auc:.4f}; threshold {threshold:.4f}; outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
