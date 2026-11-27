"""Input table, ECG histories and datasets.

Input table, one CSV row per ECG: patient_id, ecg_id, acquisition_datetime, waveform_path, age, sex, and,
for training and evaluation only, split ("train", "val" or "test") and label (1-year AF, 0 or 1).
waveform_path is a .npy array of shape (12, samples), a 10-second 12-lead ECG at 250 Hz unless --fs is given.
age is in years at that ECG and sex is "male" or "female". Every row is scored as an index ECG, with the
same patient's earlier rows as its history (up to the 20 most recent ECGs, including the index ECG).
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_ecg

MAX_HISTORY = 20


def read_table(path):
    df = pd.read_csv(path, dtype={"ecg_id": str})
    df["acquisition_datetime"] = pd.to_datetime(df["acquisition_datetime"])
    df = df.sort_values(["patient_id", "acquisition_datetime"], kind="mergesort").reset_index(drop=True)
    df["age_scaled"] = df["age"].astype(float) / 100.0
    df["sex_male"] = (df["sex"].astype(str).str.lower() == "male").astype(np.float32)
    if "split" in df:
        assert (df.groupby("patient_id")["split"].nunique() == 1).all(), "split must be patient-level"
    return df


def build_histories(df, max_len=MAX_HISTORY):
    """For each row: (row positions of its history, days before the index ECG), oldest first."""
    times = df["acquisition_datetime"].to_numpy()
    rows, days = [None] * len(df), [None] * len(df)
    for pos in df.groupby("patient_id", sort=False).indices.values():
        for i in range(len(pos)):
            seq = pos[max(0, i + 1 - max_len):i + 1]
            sec = (times[pos[i]] - times[seq]).astype("timedelta64[s]").astype(np.float64)
            rows[pos[i]] = seq
            days[pos[i]] = np.array([float(f"{d:.4f}") for d in np.clip(sec / 86400.0, 0.0, None)])
    return rows, days


def load_embeddings(path, ecg_ids):
    """Stage-2 file (ids, (N, 2, 768) window embeddings) -> (len(ecg_ids), 768), mean of the two windows."""
    z = np.load(path, allow_pickle=False)
    index = {str(k): i for i, k in enumerate(z["ids"])}
    return z["embeddings"][[index[str(k)] for k in ecg_ids]].mean(axis=1)


class PatientSampledDataset(Dataset):
    """Index ECGs `index_rows` of table `df`, at most `k_sample` per patient per epoch in training."""

    def __init__(self, df, index_rows, train, k_sample, seed):
        self.df, self.index_rows = df, np.asarray(index_rows)
        self.train, self.k_sample, self.seed = train, k_sample, seed
        pid = df["patient_id"].to_numpy()[self.index_rows]
        self.by_patient = [g.to_numpy() for _, g in pd.Series(np.arange(len(pid))).groupby(pid, sort=False)]
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch, self.active = epoch, np.arange(len(self.index_rows))
        if self.train:
            rng = np.random.default_rng(self.seed + epoch)
            self.active = np.concatenate([idx if len(idx) <= self.k_sample else
                                          rng.choice(idx, size=self.k_sample, replace=False)
                                          for idx in self.by_patient])

    def __len__(self):
        return len(self.active)

    def label(self, r):
        return float(self.df["label"].iat[r]) if "label" in self.df else 0.0


class LongitudinalDataset(PatientSampledDataset):
    """ECG histories as model input, randomly truncated in training only (subsequence augmentation)."""

    def __init__(self, df, histories, emb, index_rows, train=False, k_sample=5, seed=42):
        self.rows, self.days, self.emb = histories[0], histories[1], emb
        self.age_sex = df[["age_scaled", "sex_male"]].to_numpy(np.float32)
        super().__init__(df, index_rows, train, k_sample, seed)

    def __getitem__(self, i):
        j = int(self.active[i])
        r = self.index_rows[j]
        seq, days = self.rows[r], self.days[r]
        if self.train and len(seq) > 1:
            rng = np.random.default_rng((self.seed, self.epoch, j, 0))
            n_prior = len(seq) - 1
            n_keep = int(rng.integers(0, n_prior + 1))
            if n_keep < n_prior:
                keep = np.append(np.sort(rng.choice(n_prior, size=n_keep, replace=False)), n_prior)
                seq, days = seq[keep], days[keep]
        emb = np.zeros((MAX_HISTORY, self.emb.shape[1]), np.float32)
        t = np.zeros(MAX_HISTORY, np.float32)
        mask = np.zeros(MAX_HISTORY, bool)
        emb[:len(seq)], t[:len(seq)], mask[:len(seq)] = self.emb[seq], days, True
        return (torch.from_numpy(emb), torch.from_numpy(t), torch.from_numpy(mask),
                torch.from_numpy(self.age_sex[r]), torch.tensor(self.label(r), dtype=torch.float32))


class WaveformDataset(PatientSampledDataset):
    """Preprocessed waveforms, one random 5-s window in training, both windows (2, 12, 2500) otherwise."""

    def __init__(self, df, index_rows, train=False, k_sample=5, seed=42, fs=250):
        self.fs = fs
        super().__init__(df, index_rows, train, k_sample, seed)

    def __getitem__(self, i):
        j = int(self.active[i])
        r = self.index_rows[j]
        x = preprocess_ecg(np.load(self.df["waveform_path"].iat[r]), fs=self.fs)
        if self.train:
            x = x[int(np.random.default_rng((self.seed, self.epoch, j)).integers(0, 2))]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(self.label(r), dtype=torch.float32)
