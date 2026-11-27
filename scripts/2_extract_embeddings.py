"""Stage 2: embed every ECG with the frozen fine-tuned encoder."""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from afib_long.data import WaveformDataset, read_table  # noqa: E402
from afib_long.encoder import load_encoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True)
    ap.add_argument("--encoder", default="weights/encoder_ecgfm_finetuned.pt")
    ap.add_argument("--out", default="runs/embeddings.npz")
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = read_table(args.table)
    encoder = load_encoder(args.encoder, device)[0].eval()
    loader = DataLoader(WaveformDataset(df, np.arange(len(df)), fs=args.fs), batch_size=args.batch_size,
                        num_workers=args.num_workers)
    out = []
    with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
        for i, (x, _) in enumerate(loader):
            out.append(encoder.embed_ecg(x.to(device)).float().cpu().numpy())
            if (i + 1) % 100 == 0:
                print(f"{(i + 1) * args.batch_size} / {len(df)} ECGs", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, ids=df["ecg_id"].to_numpy(dtype=str), embeddings=np.concatenate(out).astype(np.float32))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
