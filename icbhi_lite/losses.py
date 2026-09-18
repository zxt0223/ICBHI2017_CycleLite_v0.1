import torch
import torch.nn.functional as F


def class_weights(labels, mode="sqrt"):
    counts = torch.bincount(torch.as_tensor(labels), minlength=4).float()
    if (counts == 0).any():
        raise ValueError("All four classes must be present in training")
    if mode == "sqrt":
        weight = counts.rsqrt()
    elif mode == "score":
        # A CE surrogate for the metric's normal-vs-total-abnormal weighting.
        weight = torch.stack([1/counts[0], 1/counts[1:].sum(), 1/counts[1:].sum(), 1/counts[1:].sum()])
    elif mode == "none":
        weight = torch.ones(4)
    else:
        raise ValueError(f"Unsupported class weight scheme: {mode}")
    return weight / ((weight * counts).sum() / counts.sum())


def supervised_loss(outputs, labels, weights, cfg, teacher_logits=None):
    logits, auxiliary = outputs["logits"].float(), outputs["event_logits"].float()
    ce = F.cross_entropy(logits, labels, weight=weights, label_smoothing=cfg["label_smoothing"])
    event_targets = torch.stack([(labels % 2).float(), (labels // 2).float()], 1)
    event = F.binary_cross_entropy_with_logits(auxiliary, event_targets)
    p = logits.softmax(1)
    marginal = torch.stack([p[:, 1] + p[:, 3], p[:, 2] + p[:, 3]], 1)
    consistency = F.mse_loss(auxiliary.sigmoid(), marginal)
    kd = logits.new_zeros(())
    if teacher_logits is not None:
        temperature = cfg["kd_temperature"]
        kd = F.kl_div(F.log_softmax(logits / temperature, dim=1),
                      F.softmax(teacher_logits.float() / temperature, dim=1), reduction="batchmean") * temperature**2
    total = ce + cfg["event_weight"] * event + cfg["consistency_weight"] * consistency + cfg["kd_weight"] * kd
    return total, {"ce": float(ce.detach()), "event": float(event.detach()),
                   "consistency": float(consistency.detach()), "kd": float(kd.detach())}
