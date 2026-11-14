"""SSVEP classification with CCA and filter-bank CCA (FBCCA). No per-user training needed."""
import numpy as np
from scipy.signal import butter, cheby1, filtfilt, iirnotch


def make_filters(fs, band=(5.0, 40.0), notch_hz=60.0):
    nyq = fs / 2
    b, a = butter(4, [band[0] / nyq, band[1] / nyq], btype="band")
    bn, an = iirnotch(notch_hz, 30, fs)
    return (b, a), (bn, an)


def preprocess(x, filters):
    """x: (n_samples, n_channels). Remove mean, notch mains, bandpass."""
    (b, a), (bn, an) = filters
    x = x - x.mean(axis=0)
    x = filtfilt(bn, an, x, axis=0)
    return filtfilt(b, a, x, axis=0)


def reference(f, n, fs, harmonics=3):
    t = np.arange(n) / fs
    cols = []
    for h in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * h * f * t), np.cos(2 * np.pi * h * f * t)]
    return np.column_stack(cols)


def max_canon_corr(X, Y):
    """Largest canonical correlation between X (n, p) and Y (n, q). Fast QR/SVD version."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    qx, _ = np.linalg.qr(X)
    qy, _ = np.linalg.qr(Y)
    return float(np.linalg.svd(qx.T @ qy, compute_uv=False)[0])


class CCAClassifier:
    def __init__(self, fs, freqs, n_samples, harmonics=3):
        self.refs = [reference(f, n_samples, fs, harmonics) for f in freqs]

    def scores(self, window):
        return np.array([max_canon_corr(window, R) for R in self.refs])


class FBCCAClassifier:
    """Filter-bank CCA (Chen et al. 2015): run CCA on several sub-bands, combine weighted."""

    def __init__(self, fs, freqs, n_samples, harmonics=3, n_bands=4, upper=40.0):
        self.refs = [reference(f, n_samples, fs, harmonics) for f in freqs]
        nyq = fs / 2
        lows = [5.0 + 8.0 * k for k in range(n_bands)]           # 5, 13, 21, 29 Hz
        self.bank = [cheby1(4, 0.5, [lo / nyq, upper / nyq], btype="band") for lo in lows]
        self.w = np.array([(k + 1) ** -1.25 + 0.25 for k in range(n_bands)])

    def scores(self, window):
        total = np.zeros(len(self.refs))
        for wk, (b, a) in zip(self.w, self.bank):
            xb = filtfilt(b, a, window, axis=0)
            total += wk * np.array([max_canon_corr(xb, R) for R in self.refs]) ** 2
        return np.sqrt(total / self.w.sum())
