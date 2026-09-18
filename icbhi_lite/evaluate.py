"""One prediction per original respiratory cycle; no test-time fitting."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from .data import load_manifest
from .engine import make_loader, predict, write_predictions, load_checkpoint
from .metrics import evaluate_labels, patient_bootstrap
from .model import build_model
from .utils import file_hash, write_json, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="cache/icbhi")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=2000)
    a = p.parse_args()
    torch.set_num_threads(a.cpu_threads)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    m, cp = load_manifest(a.cache), load_checkpoint(a.checkpoint)
    if cp["data_signature"] != m["data_signature"]:
        raise ValueError("Checkpoint and cache audit signatures differ")
    if cp["refit"] and a.split == "val":
        raise ValueError("Refit already trained on development-validation patients; cannot report held-out validation")
    rows = [r for r in m["rows"] if r["split"] == a.split]
    if set(cp["train_patients"]) & {r["patient"] for r in rows}:
        raise ValueError("Evaluation patients overlap model training patients")
    cfg = copy.deepcopy(cp["config"])
    method_signature = digest(cfg)
    cfg["train"].update(workers=a.workers, batch_size=a.batch_size)
    model = build_model(cfg).to(device)
    model.load_state_dict(cp["model"])
    loader = make_loader(a.cache, rows, cfg, cp["statistics"])
    metrics, values = predict(model, loader, device)
    lookup = {r["cycle_id"]: r for r in rows}
    patients = [lookup[ci]["patient"] for ci in values["cycle_ids"]]
    metrics.update(patient_bootstrap(values["labels"], values["probabilities"].argmax(1), patients, a.bootstrap))
    per_device = {}
    devices = np.asarray([lookup[ci]["device"] for ci in values["cycle_ids"]])
    for name in sorted(set(devices)):
        mask = devices == name
        per_device[name] = evaluate_labels(values["labels"][mask], values["probabilities"][mask].argmax(1))
    metrics.update(split=a.split, checkpoint_sha256=file_hash(a.checkpoint), epoch=cp["epoch"], seed=cp["seed"],
                   parameters=cp["parameters"], selection=cp["selection"], refit=cp["refit"],
                   official_counts_match=cp["official_counts_match"], data_signature=m["data_signature"],
                   official_split_sha256=m["official_split_sha256"], per_device=per_device,
                   method_signature=method_signature, model_config=cp["config"]["model"],
                   predictions="one per original annotated cycle; argmax; no threshold tuning")
    write_json(Path(a.output) / "metrics.json", metrics)
    write_predictions(Path(a.output) / "predictions.csv", values, rows)
    print({k: metrics[k] for k in ["score", "specificity", "sensitivity", "score_ci95", "split"]})


if __name__ == "__main__":
    main()
