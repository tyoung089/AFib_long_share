"""Stage 1: fine-tune ECG-FM for 1-year AF with the single-ECG model."""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from afib_long.data import WaveformDataset, read_table  # noqa: E402
from afib_long.encoder import load_encoder, load_pretrained_ecgfm, save_encoder  # noqa: E402
from afib_long.evaluation import maxpool_auroc  # noqa: E402
from afib_long.models import SingleECGModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="ECG table (see README)")
    ap.add_argument("--ecgfm", help="original ECG-FM checkpoint, mimic_iv_ecg_physionet_pretrained.pt")
    ap.add_argument("--init-from", help="fine-tuned encoder file to start from instead of --ecgfm")
    ap.add_argument("--out-dir", default="runs/encoder")
    ap.add_argument("--fs", type=int, default=250, help="sampling rate of the raw waveforms (Hz)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr", type=float, default=5e-6)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accumulation-steps", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--k-sample", type=int, default=5)
    ap.add_argument("--pos-weight-cap", type=float, default=10.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not (args.ecgfm or args.init_from):
        ap.error("give --ecgfm or --init-from")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"

    df = read_table(args.table)
    train_rows, val_rows = np.flatnonzero(df["split"] == "train"), np.flatnonzero(df["split"] == "val")
    encoder, ckpt = load_encoder(args.init_from) if args.init_from else (load_pretrained_ecgfm(args.ecgfm), {})
    encoder.freeze_lower_layers(5)
    head = SingleECGModel()
    if "single_ecg_model" in ckpt:
        head.load_state_dict(ckpt["single_ecg_model"])
    encoder.to(device)
    head.to(device)

    y = df["label"].to_numpy()[train_rows]
    pos_weight = float(min((y == 0).sum() / max((y == 1).sum(), 1), args.pos_weight_cap))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW([{"params": [p for p in encoder.parameters() if p.requires_grad],
                                    "lr": args.encoder_lr},
                                   {"params": head.parameters(), "lr": args.lr}], weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    train_ds = WaveformDataset(df, train_rows, train=True, k_sample=args.k_sample, seed=args.seed, fs=args.fs)
    val_loader = DataLoader(WaveformDataset(df, val_rows, fs=args.fs), batch_size=args.batch_size,
                            num_workers=args.num_workers)
    y_val, pid_val = df["label"].to_numpy()[val_rows], df["patient_id"].to_numpy()[val_rows]

    best, bad = -1.0, 0
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            drop_last=True)  # batch norm needs more than one sample per batch
        encoder.train()
        head.train()
        optimizer.zero_grad()
        for step, (x, label) in enumerate(loader):
            with torch.autocast("cuda", enabled=cuda):
                loss = criterion(head(encoder(x.to(device))), label.to(device))
            scaler.scale(loss / args.accumulation_steps).backward()
            if (step + 1) % args.accumulation_steps == 0 or step + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        encoder.eval()
        head.eval()
        probs = []
        with torch.no_grad(), torch.autocast("cuda", enabled=cuda):
            for x, _ in val_loader:  # both windows -> mean embedding -> MLP
                probs.append(torch.sigmoid(head(encoder.embed_ecg(x.to(device)).mean(dim=1))).float().cpu().numpy())
        p = np.concatenate(probs)
        auc, max_pool = roc_auc_score(y_val, p), maxpool_auroc(y_val, p, pid_val)
        scheduler.step()
        print(f"epoch {epoch}: val per-ECG AUROC {auc:.4f}, max-pool AUROC {max_pool:.4f}", flush=True)
        if max_pool > best:
            best, bad = max_pool, 0
            save_encoder(encoder, os.path.join(args.out_dir, "encoder.pt"), single_ecg_model=head.state_dict(),
                         epoch=epoch, val_maxpool_auroc=max_pool)
        else:
            bad += 1
            if bad >= args.patience:
                break
    print(f"saved {os.path.join(args.out_dir, 'encoder.pt')} (best val max-pool AUROC {best:.4f})")


if __name__ == "__main__":
    main()
