"""Deterministic SciPy frontend; no librosa/torchaudio version coupling."""
from functools import lru_cache
from math import gcd

import numpy as np
from scipy import signal
from scipy.io import wavfile


def read_audio(path, target_sr):
    sr, raw = wavfile.read(path)
    if np.issubdtype(raw.dtype, np.integer):
        if raw.dtype == np.uint8:
            x = (raw.astype(np.float64) - 128) / 128
        else:
            x = raw.astype(np.float64) / (float(np.iinfo(raw.dtype).max) + 1)
    else:
        x = raw.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(1)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"Invalid waveform: {path}")
    if sr != target_sr:
        d = gcd(int(sr), int(target_sr))
        x = signal.resample_poly(x, target_sr // d, sr // d)
    return x.astype(np.float32), int(sr)


def condition_audio(x, cfg):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if cfg["highpass_hz"] > 0:
        sos = signal.butter(2, cfg["highpass_hz"], btype="highpass", fs=cfg["sample_rate"], output="sos")
        if len(x) > 32:
            x = signal.sosfiltfilt(sos, x)
        else:
            x = signal.sosfilt(sos, x)
    # No silence deletion, spectral subtraction, per-cycle peak normalization or clipping.
    return x.astype(np.float32)


@lru_cache(maxsize=16)
def mel_bank(sr, n_fft, n_mels, fmin, fmax):
    if not 0 <= fmin < fmax <= sr / 2:
        raise ValueError("Require 0 <= fmin < fmax <= Nyquist")
    hz_to_mel = lambda h: 2595 * np.log10(1 + np.asarray(h) / 700)
    mel_to_hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    left = (freqs[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    right = (edges[2:, None] - freqs[None]) / (edges[2:] - edges[1:-1])[:, None]
    bank = np.maximum(0, np.minimum(left, right))
    bank *= (2 / (edges[2:] - edges[:-2]))[:, None]
    if (bank.sum(1) == 0).any():
        raise ValueError("Empty mel filter; increase FFT size or reduce mel count")
    return bank.astype(np.float32)


def mel_power(x, cfg, n_fft):
    hop = cfg["hop_length"]
    # Both resolutions have frame centers at 0, hop, ..., floor(N/hop)*hop.
    x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect" if len(x) > 1 else "edge")
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    window = signal.windows.hann(n_fft, sym=False).astype(np.float32)
    stft = np.fft.rfft(frames * window, axis=-1)
    power = (np.abs(stft) ** 2 / np.square(window).sum()).astype(np.float32)
    bank = mel_bank(cfg["sample_rate"], n_fft, cfg["n_mels"], cfg["fmin"], cfg["fmax"])
    return bank @ power.T


def pcen(energy, cfg):
    p = cfg["pcen"]
    dt = cfg["hop_length"] / cfg["sample_rate"]
    s = 1 - np.exp(-dt / p["time_constant"])
    # Initialize from the first frame; no artificial zero-state transient.
    initial = ((1 - s) * energy[:, 0])[:, None]
    smooth, _ = signal.lfilter([s], [1, -(1 - s)], energy, axis=-1, zi=initial)
    result = (energy / (p["eps"] + smooth) ** p["alpha"] + p["delta"]) ** p["r"] - p["delta"] ** p["r"]
    return result.astype(np.float32)


def extract_features(x, cfg):
    """Compute on the COMPLETE cycle; windowing happens after feature extraction."""
    powers = [mel_power(x, cfg, n) for n in cfg["n_ffts"]]
    channels = [np.log(np.maximum(e, 1e-10)) for e in powers]
    channels += [pcen(e, cfg) for e in powers]
    out = np.stack(channels).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite frontend output")
    return out


def feature_channels(mode):
    # Cache contains log-short, log-long, PCEN-short, PCEN-long.
    choices = {"log_short": [0], "log_dual": [0, 1], "dual_pcen": [0, 1, 2, 3]}
    if mode not in choices:
        raise ValueError(f"Unknown feature mode {mode}")
    return choices[mode]


def window_geometry(cfg):
    # Centered STFT has floor(N/hop)+1 frames. Include both time endpoints:
    # an exact 4 s cycle is 401 frames, and must NOT create two 400-frame windows.
    span = round(cfg["model"]["window_seconds"] * cfg["frontend"]["sample_rate"] / cfg["frontend"]["hop_length"])
    ratio = cfg["model"]["window_stride_ratio"]
    if span < 1 or not 0 < ratio <= 1:
        raise ValueError("Positive window duration and stride ratio in (0,1] required")
    return span + 1, max(1, round(span * ratio))


def window_starts(n_frames, width, stride):
    if width < 1 or not 1 <= stride <= width or n_frames < 1:
        raise ValueError("Invalid segmentation settings")
    if n_frames <= width:
        return [0]
    starts = list(range(0, n_frames - width + 1, stride))
    if starts[-1] != n_frames - width:
        starts.append(n_frames - width)
    return starts
