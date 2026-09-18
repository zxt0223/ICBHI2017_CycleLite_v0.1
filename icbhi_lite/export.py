"""Export the feature-to-logit network; retain the exact Python frontend."""
import argparse
from pathlib import Path

import torch

from .engine import load_checkpoint
from .frontend import feature_channels, window_geometry
from .model import build_model, InferenceModel
from .utils import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    cp = load_checkpoint(a.checkpoint)
    model = build_model(cp["config"]).eval()
    model.load_state_dict(cp["model"])
    wrapper = InferenceModel(model).eval()
    scripted = torch.jit.freeze(torch.jit.script(wrapper))
    c = len(feature_channels(cp["config"]["model"]["feature_mode"]))
    f = cp["config"]["frontend"]["n_mels"]
    t, _ = window_geometry(cp["config"])
    with torch.inference_mode():
        for batch, windows in [(1, 1), (2, 3)]:
            x = torch.randn(batch, windows, c, f, t)
            lengths = torch.full((batch, windows), t, dtype=torch.long)
            lengths[0, 0] = max(1, t // 3)
            if windows > 1:
                lengths[0, -1] = 0
            torch.testing.assert_close(scripted(x, lengths), wrapper(x, lengths), rtol=1e-4, atol=1e-5)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(a.output)
    write_json(Path(a.output).with_suffix(".json"), {"config": cp["config"], "statistics": cp["statistics"],
               "class_names": cp["class_names"], "input": "normalized cached-style features B,W,C,F,T and int64 valid lengths B,W",
               "output": "four logits in class_names order", "frontend_included": False,
               "verified": "eager vs scripted on variable batch/window counts including masked windows",
               "note": "TorchScript legacy deployment format; no INT8 speedup or whole-waveform export is claimed"})
    print(f"Exported and numerically checked: {a.output}")


if __name__ == "__main__":
    main()
