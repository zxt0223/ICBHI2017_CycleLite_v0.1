"""Classify one already-segmented respiratory cycle, or cycles in a WAV/TXT pair."""
import argparse
from pathlib import Path

import numpy as np
import torch

from . import CLASS_NAMES
from .engine import load_checkpoint
from .frontend import read_audio, condition_audio, extract_features, feature_channels, window_starts, window_geometry
from .model import build_model
from .utils import write_json


def tensorize(features, cfg, stats):
    channels = feature_channels(cfg["model"]["feature_mode"])
    # Match the float16 feature-cache roundtrip exactly.
    f = features.astype(np.float16).astype(np.float32)[channels]
    f = (f - np.asarray(stats["mean"], np.float32)) / np.asarray(stats["std"], np.float32)
    width, stride = window_geometry(cfg)
    starts = window_starts(f.shape[-1], width, stride)
    x = np.zeros((1, len(starts), f.shape[0], f.shape[1], width), np.float32)
    lengths = np.zeros((1, len(starts)), np.int64)
    for i, s in enumerate(starts):
        part = f[..., s:s+width]
        x[0, i, ..., :part.shape[-1]] = part
        lengths[0, i] = part.shape[-1]
    return torch.from_numpy(x), torch.from_numpy(lengths)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--wav", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--annotation", help="Cycle boundaries; first two whitespace-separated columns are start and end seconds")
    group.add_argument("--single-cycle", action="store_true", help="Assert the WAV is already one complete respiratory cycle")
    p.add_argument("--output", default="prediction.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cpu-threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.cpu_threads)
    cp = load_checkpoint(a.checkpoint)
    cfg = cp["config"]
    model = build_model(cfg).to(a.device).eval()
    model.load_state_dict(cp["model"])
    wave, _ = read_audio(a.wav, cfg["frontend"]["sample_rate"])
    wave = condition_audio(wave, cfg["frontend"])
    sr = cfg["frontend"]["sample_rate"]
    spans = [(0.0, len(wave)/sr)] if a.single_cycle else [tuple(map(float, l.split()[:2])) for l in Path(a.annotation).read_text().splitlines() if l.strip()]
    outputs = []
    for i, (start, end) in enumerate(spans):
        if not 0 <= start < end <= len(wave)/sr + 0.02:
            raise ValueError(f"Invalid boundary {start}, {end}")
        segment = wave[round(start*sr):min(len(wave), round(end*sr))]
        feat = extract_features(segment, cfg["frontend"])
        x, lengths = tensorize(feat, cfg, cp["statistics"])
        pred = model(x.to(a.device), lengths.to(a.device))
        probability = pred["logits"].softmax(1)[0].cpu().tolist()
        outputs.append({"cycle_index": i, "start": start, "end": end,
                        "predicted_class": CLASS_NAMES[int(np.argmax(probability))],
                        "probabilities": dict(zip(CLASS_NAMES, probability)),
                        "event_head_probabilities": dict(zip(["crackle", "wheeze"], pred["event_logits"].sigmoid()[0].cpu().tolist())),
                        "windows": x.shape[1]})
    write_json(a.output, {"task": "four-class respiratory sound pattern classification", "cycles": outputs,
                          "note": "Probabilities are uncalibrated research-model scores; this does not classify a disease."})
    print(outputs)


if __name__ == "__main__":
    main()
