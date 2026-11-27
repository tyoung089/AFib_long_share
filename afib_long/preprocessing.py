"""ECG preprocessing: high-pass and notch filters, per-lead z-score, resampling to 500 Hz, two 5-s windows."""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample

TARGET_FS = 500
WINDOW = 2500  # 5 s at 500 Hz, the input length of ECG-FM


def preprocess_ecg(ecg, fs=250):
    """Raw ECG of shape (12, n) or (n, 12) at `fs` Hz -> float32 array (2, 12, 2500) at 500 Hz."""
    ecg = np.asarray(ecg)
    if ecg.ndim == 3 and ecg.shape[-1] == 1:
        ecg = ecg[..., 0]
    if ecg.shape[0] > ecg.shape[1]:
        ecg = ecg.T
    ecg = ecg.astype(np.float64)
    b, a = butter(2, 0.5 / (fs / 2), btype="high")
    ecg = filtfilt(b, a, ecg, axis=-1)
    for f0 in (50.0, 60.0):
        if f0 < fs / 2:
            b, a = iirnotch(f0, 30, fs)
            ecg = filtfilt(b, a, ecg, axis=-1)
    ecg = (ecg - ecg.mean(axis=-1, keepdims=True)) / (ecg.std(axis=-1, keepdims=True) + 1e-8)
    ecg = ecg.astype(np.float32)
    ecg = resample(ecg, int(ecg.shape[-1] * TARGET_FS / fs), axis=-1).astype(np.float32)
    if ecg.shape[-1] < 2 * WINDOW:
        ecg = np.concatenate([ecg, np.zeros((ecg.shape[0], 2 * WINDOW - ecg.shape[-1]), np.float32)], axis=-1)
    return np.stack([ecg[:, :WINDOW], ecg[:, WINDOW:2 * WINDOW]])
