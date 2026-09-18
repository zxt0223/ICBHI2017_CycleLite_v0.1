from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .frontend import feature_channels, window_starts, window_geometry
from .utils import load_json, digest


def load_manifest(cache):
    m = load_json(Path(cache) / "manifest.json")
    expected = m["data_signature"]
    if digest({k: v for k, v in m.items() if k != "data_signature"}) != expected:
        raise ValueError("Manifest changed since preprocessing; create a new audited cache")
    return m


def fit_statistics(cache, rows, channels):
    total, squares, count = None, None, 0
    for row in rows:
        x = np.load(Path(cache) / row["feature_path"], allow_pickle=False)[channels].astype(np.float64)
        s = x.sum((1, 2), keepdims=True)
        q = np.square(x).sum((1, 2), keepdims=True)
        total = s if total is None else total + s
        squares = q if squares is None else squares + q
        count += x.shape[1] * x.shape[2]
    if not count:
        raise ValueError("No training features")
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 1e-4))
    return {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist(),
            "fit_cycle_ids": sorted(r["cycle_id"] for r in rows)}


class CycleDataset(Dataset):
    def __init__(self, cache, rows, cfg, stats):
        self.cache, self.rows = Path(cache), rows
        self.channels = feature_channels(cfg["model"]["feature_mode"])
        self.width, self.stride = window_geometry(cfg)
        self.mean, self.std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        x = np.load(self.cache / r["feature_path"], allow_pickle=False)[self.channels].astype(np.float32)
        x = (x - self.mean) / self.std
        starts = window_starts(x.shape[-1], self.width, self.stride)
        windows = np.zeros((len(starts), x.shape[0], x.shape[1], self.width), np.float32)
        lengths = np.zeros(len(starts), np.int64)
        for i, start in enumerate(starts):
            part = x[..., start:start + self.width]
            lengths[i] = part.shape[-1]
            windows[i, ..., :part.shape[-1]] = part
        return torch.from_numpy(windows), torch.from_numpy(lengths), r["label"], r["cycle_id"]


def collate_cycles(items):
    b, w = len(items), max(x[0].shape[0] for x in items)
    x = torch.zeros((b, w, *items[0][0].shape[1:]), dtype=torch.float32)
    lengths = torch.zeros((b, w), dtype=torch.long)
    for i, (windows, valid, _, _) in enumerate(items):
        x[i, :len(windows)] = windows
        lengths[i, :len(windows)] = valid
    return x, lengths, torch.tensor([r[2] for r in items], dtype=torch.long), [r[3] for r in items]


def augment_features(x, lengths, cfg):
    """Mild, same-time masks in all channels; no label-changing mixup by default."""
    x = x.clone()
    for b in range(x.shape[0]):
        for w in range(x.shape[1]):
            n = int(lengths[b, w])
            if not n:
                continue
            if torch.rand(()) < cfg["probability"]:
                limit = min(cfg["frequency_mask"], x.shape[-2] // 8)
                size = int(torch.randint(limit + 1, ()))
                start = int(torch.randint(x.shape[-2] - size + 1, ()))
                x[b, w, :, start:start+size, :n] = 0
                limit = min(cfg["time_mask"], max(0, n // 20))
                size = int(torch.randint(limit + 1, ()))
                start = int(torch.randint(n - size + 1, ()))
                x[b, w, ..., start:start+size] = 0
            if cfg["feature_noise_std"] > 0:
                x[b, w, ..., :n] += torch.randn_like(x[b, w, ..., :n]) * cfg["feature_noise_std"]
    return x
