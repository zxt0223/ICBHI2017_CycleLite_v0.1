"""ICBHI four-class metrics. Rows=true labels; columns=predictions."""
import numpy as np


def from_confusion(cm):
    cm = np.asarray(cm, dtype=np.int64)
    if cm.shape != (4, 4) or (cm < 0).any():
        raise ValueError("Expected a nonnegative 4x4 confusion matrix")
    support = cm.sum(1)
    correct = cm.diagonal()
    sp = float(correct[0] / support[0]) if support[0] else None
    # Misclassifying crackle as wheeze is an ERROR, even though both are abnormal.
    se = float(correct[1:].sum() / support[1:].sum()) if support[1:].sum() else None
    score = 50.0 * (sp + se) if sp is not None and se is not None else None
    recall = np.divide(correct, support, out=np.zeros(4), where=support > 0)
    precision = np.divide(correct, cm.sum(0), out=np.zeros(4), where=cm.sum(0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(4), where=(precision + recall) > 0)
    return {"score": score, "specificity": None if sp is None else 100 * sp,
            "sensitivity": None if se is None else 100 * se,
            "accuracy": float(100 * correct.sum() / cm.sum()) if cm.sum() else None,
            "macro_f1": float(100 * f1.mean()), "recall_per_class": (100 * recall).tolist(),
            "support": support.tolist(), "confusion_matrix": cm.tolist(), "unit": "percent"}


def evaluate_labels(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or not len(y_true):
        raise ValueError("Expected nonempty matching label vectors")
    if ((y_true < 0) | (y_true > 3) | (y_pred < 0) | (y_pred > 3)).any():
        raise ValueError("Class IDs must be 0,1,2,3")
    return from_confusion(np.bincount(4 * y_true + y_pred, minlength=16).reshape(4, 4))


def patient_bootstrap(y_true, y_pred, patients, repeats=2000, seed=2026):
    """Cluster bootstrap; respiratory cycles from one patient stay together."""
    patients = np.asarray(patients)
    unique = np.unique(patients)
    if len(unique) < 2:
        return {"score_ci95": None, "valid_resamples": 0}
    cms = []
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    for p in unique:
        keep = patients == p
        cms.append(np.bincount(4 * y_true[keep] + y_pred[keep], minlength=16).reshape(4, 4))
    cms = np.stack(cms)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        score = from_confusion(cms[rng.integers(len(unique), size=len(unique))].sum(0))["score"]
        if score is not None:
            scores.append(score)
    return {"score_ci95": np.percentile(scores, [2.5, 97.5]).tolist() if scores else None,
            "valid_resamples": len(scores), "requested_resamples": repeats,
            "bootstrap_unit": "patient", "bootstrap_seed": seed}
