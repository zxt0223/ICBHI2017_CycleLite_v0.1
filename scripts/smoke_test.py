"""Synthetic end-to-end regression only. Never an ICBHI accuracy benchmark."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from icbhi_lite.engine import load_checkpoint
from icbhi_lite.utils import write_json, load_config, load_json


def create_synthetic(root):
    root.mkdir(parents=True)
    rng = np.random.default_rng(19)
    split = []
    for i in range(16):
        sr = [8000, 16000, 22050, 44100][i % 4]
        name = f"{101+i}_1b1_Al_sc_ToyDevice{i%2}"
        pieces, annotation, start = [], [], 0.0
        for label in range(4):
            seconds = 0.65 + label * 0.18 if i % 3 else 1.65 + label * 0.23
            n = round(sr * seconds)
            t = np.arange(n) / sr
            x = rng.normal(0, 0.015, n)
            x += 0.02 * np.sin(2*np.pi*180*t)
            if label & 1:
                for center in [0.2, 0.45]:
                    x += 0.22 * np.exp(-((t-center)/0.003)**2) * np.cos(2*np.pi*1300*t)
            if label & 2:
                x += 0.09 * np.sin(2*np.pi*650*t) * np.sin(np.pi*np.arange(n)/n)**2
            pieces.append(x)
            end = start + n/sr
            annotation.append(f"{start:.9f}\t{end:.9f}\t{label%2}\t{label//2}")
            start = end
        x = (np.concatenate(pieces) * 32767).astype(np.int16)
        if i % 2:
            x = np.stack([x, x], 1)
        wavfile.write(root / (name + ".wav"), sr, x)
        (root / (name + ".txt")).write_text("\n".join(annotation) + "\n")
        split.append(name + (" train" if i < 12 else " test"))
    path = root.parent / "synthetic_split.txt"
    path.write_text("\n".join(split) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="verification/smoke_test.json")
    a = parser.parse_args()
    checks, logs = [], []
    with tempfile.TemporaryDirectory(prefix="cyclelite_smoke_") as tmp:
        d = Path(tmp)
        split = create_synthetic(d / "audio")
        cfg = load_config(ROOT / "configs/proposed.yaml")
        cfg["model"].update(width=0.25, window_seconds=1.0, window_stride_ratio=1.0)
        cfg["train"].update(epochs=2, workers=0, batch_size=8, warmup_epochs=0)
        config = d / "smoke.yaml"
        config.write_text(yaml.safe_dump(cfg))
        def run(label, module, *arguments, expect_success=True):
            print(f"[smoke] {label}", flush=True)
            result = subprocess.run([sys.executable, "-m", "icbhi_lite."+module, *map(str, arguments)],
                                    cwd=ROOT, env=os.environ.copy(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            logs.append(f"=== {label} ===\n{result.stdout}")
            if (result.returncode == 0) != expect_success:
                raise RuntimeError(result.stdout)
            checks.append({"check": label, "passed": True})
        cache = d / "cache"
        run("resampling_stereo_annotation_cache_and_patient_split", "prepare", "--data-root", d/"audio", "--official-split", split,
            "--output", cache, "--config", config, "--allow-nonstandard-counts")
        common = ["--cache", cache, "--config", config, "--device", "cpu", "--cpu-threads", "1", "--seed", "7"]
        run("uninterrupted_training_and_validation", "train", *common, "--output", d/"full")
        run("pause_training", "train", *common, "--output", d/"resumed", "--stop-after", "1")
        run("resume_training", "train", *common, "--output", d/"resumed", "--resume", d/"resumed/last.pt")
        run("completed_run_resume_is_idempotent", "train", *common, "--output", d/"resumed", "--resume", d/"resumed/last.pt")
        full, resumed = load_checkpoint(d/"full/last.pt"), load_checkpoint(d/"resumed/last.pt")
        for k in full["model"]:
            torch.testing.assert_close(full["model"][k], resumed["model"][k], rtol=0, atol=0)
        checks.append({"check": "resume_matches_uninterrupted_weights_bitwise_on_cpu", "passed": True})
        run("held_out_synthetic_evaluation_and_cluster_bootstrap", "evaluate", "--cache", cache, "--checkpoint", d/"full/best.pt",
            "--split", "test", "--output", d/"test", "--workers", "0", "--cpu-threads", "1", "--bootstrap", "100")
        for script, arguments in [
            ("summarize_results.py", ["--results", d/"test/metrics.json", "--output", d/"summary.json"]),
            ("compare_predictions.py", ["--baseline", d/"test/predictions.csv", "--proposed", d/"test/predictions.csv",
                                        "--repeats", "100", "--output", d/"comparison.json"])]:
            subprocess.run([sys.executable, str(ROOT/"scripts"/script), *map(str, arguments)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        comparison = load_json(d/"comparison.json")
        if comparison["paired_difference_ci95"] != [0.0, 0.0]:
            raise AssertionError("Identical predictions must have zero paired difference")
        checks.append({"check": "multi_seed_summary_and_paired_patient_comparison", "passed": True})
        run("variable_window_script_export", "export", "--checkpoint", d/"full/best.pt", "--output", d/"export/model.ts")
        wav = sorted((d/"audio").glob("*.wav"))[0]
        run("wav_annotation_cycle_inference", "infer", "--checkpoint", d/"full/best.pt", "--wav", wav,
            "--annotation", wav.with_suffix(".txt"), "--cpu-threads", "1", "--output", d/"inference.json")
        # Inference probabilities must match the cached evaluation pipeline.
        from icbhi_lite.data import load_manifest
        from icbhi_lite.engine import make_loader, predict
        from icbhi_lite.model import build_model
        cp = load_checkpoint(d/"full/best.pt")
        model = build_model(cp["config"]).eval()
        model.load_state_dict(cp["model"])
        rows = [r for r in load_manifest(cache)["rows"] if r["recording"] == wav.stem]
        loader = make_loader(cache, rows, cp["config"], cp["statistics"])
        torch.set_num_threads(1)
        _, values = predict(model, loader, torch.device("cpu"))
        inference = load_json(d/"inference.json")
        for p1, record in zip(values["probabilities"], inference["cycles"]):
            np.testing.assert_allclose(p1, list(record["probabilities"].values()), rtol=1e-5, atol=1e-6)
        checks.append({"check": "raw_wav_inference_matches_cached_cycle_predictions", "passed": True})
        kd_cfg = copy.deepcopy(cfg)
        kd_cfg["loss"]["kd_weight"] = 0.3
        kd_path = d / "kd.yaml"
        kd_path.write_text(yaml.safe_dump(kd_cfg))
        run("knowledge_distillation_training", "train", "--cache", cache, "--config", kd_path, "--device", "cpu",
            "--cpu-threads", "1", "--output", d/"kd", "--teacher", d/"full/best.pt", "--epochs", "1")
        run("fixed_epoch_full_train_refit", "train", *common, "--output", d/"refit", "--refit", "--epochs", "1")
        run("refit_validation_leakage_is_rejected", "evaluate", "--cache", cache, "--checkpoint", d/"refit/final.pt",
            "--split", "val", "--output", d/"invalid", expect_success=False)
        run("teacher_patient_leakage_is_rejected", "train", "--cache", cache, "--config", kd_path, "--device", "cpu",
            "--cpu-threads", "1", "--output", d/"bad_kd", "--teacher", d/"refit/final.pt", "--epochs", "1", expect_success=False)
    output = Path(a.output)
    write_json(output, {"status": "passed", "data": "generated synthetic audio ONLY, not ICBHI2017",
                        "torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(),
                        "checks": checks, "real_icbhi_official_score": None})
    output.with_suffix(".log").write_text("\n".join(logs), encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(checks), "accuracy_claim": "none"}))


if __name__ == "__main__":
    main()
