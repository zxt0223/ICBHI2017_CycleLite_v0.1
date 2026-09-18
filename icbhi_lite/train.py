"""Validation-selected development training, or fixed-epoch full-train refit."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from . import CLASS_NAMES
from .data import load_manifest, fit_statistics, augment_features
from .engine import EMA, make_loader, predict, write_predictions, save_checkpoint, load_checkpoint
from .frontend import feature_channels
from .losses import class_weights, supervised_loss
from .model import build_model
from .utils import load_config, seed_all, write_json, digest, file_hash


def rng_state(loader):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": loader.generator.get_state()}


def restore_rng(state, loader):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    loader.generator.set_state(state["loader"].cpu())


def learning_rate(epoch, total, warmup, peak, minimum):
    warmup = min(warmup, max(0, total - 1))
    if epoch < warmup:
        return peak * (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(1, total - warmup - 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def train(args):
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.workers is not None:
        cfg["train"]["workers"] = args.workers
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if cfg["train"]["epochs"] < 1:
        raise ValueError("epochs must be positive")
    seed_all(args.seed, cfg["train"]["deterministic"])
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    manifest = load_manifest(args.cache)
    if cfg["frontend"] != manifest["frontend"]:
        raise ValueError("Frontend config differs from cache; prepare a new cache")
    rows = manifest["rows"]
    train_rows = [r for r in rows if (r["official_split"] == "train" if args.refit else r["split"] == "train")]
    val_rows = [] if args.refit else [r for r in rows if r["split"] == "val"]
    train_ids = sorted(r["cycle_id"] for r in train_rows)
    output = Path(args.output)
    resume = load_checkpoint(args.resume) if args.resume else None
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory not empty; use --resume or a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        if Path(args.resume).name != "last.pt" or Path(args.resume).resolve().parent != output.resolve():
            raise ValueError("Resume from this run's last.pt into the same output directory")
        if resume["config"] != cfg or resume["seed"] != args.seed or resume["refit"] != args.refit:
            raise ValueError("Resume config, seed and refit setting must match the original run")
        if resume["data_signature"] != manifest["data_signature"] or resume["train_cycle_ids"] != train_ids:
            raise ValueError("Resume data/split mismatch")
        stats = resume["statistics"]
    else:
        stats = fit_statistics(args.cache, train_rows, feature_channels(cfg["model"]["feature_mode"]))
    write_json(output / "config.json", cfg)
    write_json(output / "normalization.json", stats)
    train_loader = make_loader(args.cache, train_rows, cfg, stats, train=True, seed=args.seed)
    val_loader = make_loader(args.cache, val_rows, cfg, stats) if val_rows else None
    model = build_model(cfg).to(device)
    ema = EMA(model, cfg["train"]["ema_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    amp = bool(cfg["train"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    weights = class_weights([r["label"] for r in train_rows], cfg["loss"]["class_weight"]).to(device)
    teacher, teacher_hash = None, None
    if args.teacher:
        tc = load_checkpoint(args.teacher)
        if tc["data_signature"] != manifest["data_signature"] or tc["train_cycle_ids"] != train_ids:
            raise ValueError("Teacher must use exactly the same training patients/cycles; leakage or split mismatch")
        if digest(tc["statistics"]) != digest(stats):
            raise ValueError("Teacher/student normalization differs")
        for name in ["feature_mode", "window_seconds", "window_stride_ratio"]:
            if tc["config"]["model"][name] != cfg["model"][name]:
                raise ValueError(f"Teacher/student input differs: {name}")
        teacher = build_model(tc["config"]).to(device).eval()
        teacher.load_state_dict(tc["model"])
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher_hash = file_hash(args.teacher)
    if cfg["loss"]["kd_weight"] > 0 and teacher is None:
        raise ValueError("kd_weight > 0 requires --teacher")
    if teacher is not None and cfg["loss"]["kd_weight"] <= 0:
        raise ValueError("Teacher provided but kd_weight is zero; use configs/distill.yaml")
    best_score, best_ce, best_epoch, start_epoch = -1.0, float("inf"), 0, 0
    if resume:
        if resume["teacher_sha256"] != teacher_hash:
            raise ValueError("Teacher changed on resume")
        model.load_state_dict(resume["raw_model"])
        ema.model.load_state_dict(resume["model"])
        ema.steps = resume["ema_steps"]
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch, best_score, best_ce, best_epoch = resume["epoch"], resume["best_score"], resume["best_ce"], resume["best_epoch"]
        restore_rng(resume["rng"], train_loader)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Device={device}; training parameters={parameters:,}; train cycles={len(train_rows)}; val cycles={len(val_rows)}", flush=True)
    print("Protocol: " + ("fixed-epoch refit on all official training patients" if args.refit else "patient-disjoint development validation; official test is not evaluated"), flush=True)
    epoch = start_epoch - 1
    if resume and start_epoch >= cfg["train"]["epochs"] and args.refit and not (output / "final.pt").exists():
        save_checkpoint(output / "final.pt", resume)
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t0 = time.perf_counter()
        model.train()
        lr = learning_rate(epoch, cfg["train"]["epochs"], cfg["train"]["warmup_epochs"],
                           cfg["train"]["learning_rate"], cfg["train"]["min_learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        total_loss, seen = 0.0, 0
        component_sums = {k: 0.0 for k in ["ce", "event", "consistency", "kd"]}
        for x, lengths, labels, _ in train_loader:
            x = augment_features(x, lengths, cfg["augment"])
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs = model(x, lengths)
                with torch.no_grad():
                    tl = teacher(x, lengths)["logits"] if teacher is not None else None
                loss, parts = supervised_loss(outputs, labels, weights, cfg["loss"], tl)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; inspect audio/cache/learning rate")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_grad_norm"])
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= before:
                ema.update(model)
            seen += len(labels)
            total_loss += loss.item() * len(labels)
            for k, value in parts.items():
                component_sums[k] += value * len(labels)
        val, prediction = predict(ema.model, val_loader, device) if val_loader is not None else (None, None)
        improved = val is not None and (val["score"] > best_score + 1e-9 or
                   (abs(val["score"] - best_score) <= 1e-9 and val["cross_entropy"] < best_ce))
        if improved:
            best_score, best_ce, best_epoch = val["score"], val["cross_entropy"], epoch + 1
        record = {"epoch": epoch + 1, "lr": lr, "train_loss": total_loss / seen,
                  "loss_components": {k: v / seen for k, v in component_sums.items()},
                  "validation": val, "seconds": time.perf_counter() - t0}
        checkpoint = {"schema_version": 1, "model": ema.model.state_dict(), "raw_model": model.state_dict(),
                      "ema_steps": ema.steps, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                      "config": cfg, "statistics": stats, "data_signature": manifest["data_signature"],
                      "official_split_sha256": manifest["official_split_sha256"], "official_counts_match": manifest["official_counts_match"],
                      "class_names": CLASS_NAMES, "train_cycle_ids": train_ids,
                      "train_patients": sorted({r["patient"] for r in train_rows}),
                      "validation_patients": sorted({r["patient"] for r in val_rows}),
                      "epoch": epoch + 1, "best_epoch": best_epoch, "best_score": best_score,
                      "best_ce": best_ce, "validation": val, "seed": args.seed, "refit": args.refit,
                      "selection": "fixed_epoch" if args.refit else "validation_score_then_cross_entropy",
                      "teacher_sha256": teacher_hash, "parameters": parameters, "torch_version": str(torch.__version__),
                      "rng": rng_state(train_loader)}
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
            write_json(output / "best_validation.json", {"epoch": epoch + 1, **val})
            write_predictions(output / "validation_predictions.csv", prediction, val_rows)
        save_checkpoint(output / "last.pt", checkpoint)
        if args.refit and epoch + 1 == cfg["train"]["epochs"]:
            save_checkpoint(output / "final.pt", checkpoint)
        with open(output / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if args.stop_after and epoch + 1 >= args.stop_after:
            break
    result = {"status": "completed" if epoch + 1 == cfg["train"]["epochs"] else "paused",
              "parameters": parameters, "last_epoch": epoch + 1,
              "best_validation_score": None if args.refit else best_score,
              "best_epoch": None if args.refit else best_epoch,
              "official_test_evaluated": False, "refit": args.refit}
    write_json(output / "run_summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="cache/icbhi")
    p.add_argument("--config", default="configs/proposed.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--teacher")
    p.add_argument("--resume")
    p.add_argument("--refit", action="store_true", help="Use all official training cycles; epochs must be chosen on development validation first")
    p.add_argument("--stop-after", type=int, help="Pause after this epoch; keep schedule intact for resuming")
    args = p.parse_args()
    if args.refit and args.epochs is None:
        p.error("--refit requires an explicit validation-selected --epochs")
    train(args)


if __name__ == "__main__":
    main()
