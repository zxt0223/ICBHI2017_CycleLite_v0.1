"""Parameter count, Conv/Linear MACs, batch-one latency and frontend timing."""
import argparse
import io
import os
import platform
import time

import numpy as np
import torch
from torch import nn

from .engine import load_checkpoint
from .frontend import feature_channels, condition_audio, extract_features, window_geometry
from .model import build_model
from .utils import load_config, write_json


def profile(cfg, device="cpu", repeats=30, threads=1, checkpoint=None):
    torch.set_num_threads(threads)
    model = build_model(cfg).to(device).eval()
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    m, f = cfg["model"], cfg["frontend"]
    frames, _ = window_geometry(cfg)
    x = torch.randn(1, 1, len(feature_channels(m["feature_mode"])), f["n_mels"], frames, device=device)
    lengths = torch.full((1, 1), frames, dtype=torch.long, device=device)
    macs, handles = [0], []
    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            macs[0] += output.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size)
        elif isinstance(module, nn.Conv1d):
            macs[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
        elif isinstance(module, nn.Linear):
            macs[0] += output.numel() * module.in_features
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
    sync = lambda: torch.cuda.synchronize(device) if str(device).startswith("cuda") else None
    with torch.inference_mode():
        model(x, lengths)
        for h in handles:
            h.remove()
        for _ in range(5):
            model(x, lengths)
        times = []
        for _ in range(repeats):
            sync()
            t0 = time.perf_counter()
            model(x, lengths)
            sync()
            times.append(1000 * (time.perf_counter() - t0))
    waveform = np.random.default_rng(7).normal(0, 0.02, round(m["window_seconds"] * f["sample_rate"])).astype(np.float32)
    frontend_times = []
    for _ in range(max(3, repeats // 5)):
        t0 = time.perf_counter()
        extract_features(condition_audio(waveform, f), f)
        frontend_times.append(1000 * (time.perf_counter() - t0))
    state = io.BytesIO()
    torch.save(model.state_dict(), state)
    params = sum(p.numel() for p in model.parameters())
    result = {"parameters": params, "parameters_million": params / 1e6,
              "fp32_parameter_MiB": params * 4 / 1024**2,
              "state_dict_MiB": len(state.getbuffer()) / 1024**2,
              "conv_linear_MACs": int(macs[0]), "conv_linear_MMACs": float(macs[0] / 1e6),
              "model_latency_ms_median": float(np.median(times)), "model_latency_ms_p95": float(np.percentile(times, 95)),
              "frontend_latency_ms_median": float(np.median(frontend_times)),
              "input_shape": list(x.shape), "batch_size": 1, "windows": 1,
              "window_seconds": m["window_seconds"], "device": str(device), "cpu_threads": threads,
              "platform": platform.platform(), "torch": str(torch.__version__), "repeats": repeats,
              "thread_environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
              "mac_scope": "Conv1d/Conv2d/Linear only; excludes STFT/PCEN, activations, norm, pooling and attention matmul",
              "latency_scope": "model on one fixed feature window; excludes disk I/O, resampling and window assembly; frontend reported separately",
              "data": "synthetic random inputs; not an accuracy measurement",
              "cycle_cost_note": "Backbone cost grows with the number of overlapping windows in the cycle"}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/proposed.yaml")
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", default="profile.json")
    a = p.parse_args()
    cp = load_checkpoint(a.checkpoint) if a.checkpoint else None
    cfg = cp["config"] if cp else load_config(a.config)
    result = profile(cfg, a.device, a.repeats, a.threads, cp)
    write_json(a.output, result)
    print(result)


if __name__ == "__main__":
    main()
