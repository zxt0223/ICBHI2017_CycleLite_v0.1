import copy
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CycleDataset, collate_cycles
from .metrics import evaluate_labels
from .utils import seed_worker


class EMA:
    def __init__(self, model, decay=0.995):
        self.model = copy.deepcopy(model).eval()
        self.decay, self.steps = decay, 0
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        source = model.state_dict()
        for name, target in self.model.state_dict().items():
            if target.is_floating_point():
                target.mul_(decay).add_(source[name].detach(), alpha=1 - decay)
            else:
                target.copy_(source[name])


def make_loader(cache, rows, cfg, stats, train=False, seed=1):
    ds = CycleDataset(cache, rows, cfg, stats)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=train,
                      num_workers=cfg["train"]["workers"], collate_fn=collate_cycles,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=cfg["train"]["workers"] > 0)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    ys, probs, logits, ids = [], [], [], []
    loss, samples = 0.0, 0
    for x, lengths, y, cycle_ids in loader:
        out = model(x.to(device), lengths.to(device))["logits"].float()
        loss += torch.nn.functional.cross_entropy(out, y.to(device), reduction="sum").item()
        samples += len(y)
        logits.append(out.cpu().numpy())
        probs.append(out.softmax(-1).cpu().numpy())
        ys.extend(y.tolist())
        ids.extend(cycle_ids)
    if not samples:
        raise ValueError("Empty prediction set")
    probs, logits, ys = np.concatenate(probs), np.concatenate(logits), np.asarray(ys)
    result = evaluate_labels(ys, probs.argmax(1))
    result["cross_entropy"] = loss / samples
    return result, {"labels": ys, "probabilities": probs, "logits": logits, "cycle_ids": ids}


def write_predictions(path, values, rows):
    lookup = {r["cycle_id"]: r for r in rows}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cycle_id", "patient", "device", "true", "pred", "p_normal", "p_crackle", "p_wheeze", "p_both"])
        for ci, y, probs in zip(values["cycle_ids"], values["labels"], values["probabilities"]):
            r = lookup[ci]
            w.writerow([ci, r["patient"], r["device"], int(y), int(probs.argmax()), *map(float, probs)])


def save_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def load_checkpoint(path, device="cpu"):
    # Repository checkpoints contain tensors + builtin containers only.
    return torch.load(path, map_location=device, weights_only=True)
