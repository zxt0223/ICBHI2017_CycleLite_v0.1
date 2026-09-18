"""Paired patient-cluster bootstrap of a prespecified model comparison."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icbhi_lite.metrics import from_confusion
from icbhi_lite.utils import write_json


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len({r["cycle_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate cycle IDs")
    return {r["cycle_id"]: r for r in rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--proposed", required=True)
    p.add_argument("--repeats", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", default="paired_comparison.json")
    a = p.parse_args()
    if a.repeats < 1:
        raise ValueError("repeats must be positive")
    base, prop = read(a.baseline), read(a.proposed)
    if set(base) != set(prop):
        raise ValueError("Compare exactly the same cycles")
    patients = sorted({r["patient"] for r in base.values()})
    if len(patients) < 2:
        raise ValueError("At least two patients required")
    index = {p: i for i, p in enumerate(patients)}
    cms = np.zeros((2, len(patients), 4, 4), np.int64)
    for ci, b in base.items():
        r = prop[ci]
        if b["patient"] != r["patient"] or b["true"] != r["true"]:
            raise ValueError("Patient/ground-truth mismatch")
        for j, rec in enumerate([b, r]):
            y, prediction = int(rec["true"]), int(rec["pred"])
            if y not in range(4) or prediction not in range(4):
                raise ValueError("Invalid class ID")
            cms[j, index[rec["patient"]], y, prediction] += 1
    rng = np.random.default_rng(a.seed)
    differences = []
    for _ in range(a.repeats):
        chosen = rng.integers(len(patients), size=len(patients))
        scores = [from_confusion(cm[chosen].sum(0))["score"] for cm in cms]
        if all(s is not None for s in scores):
            differences.append(scores[1] - scores[0])
    actual = [from_confusion(cm.sum(0))["score"] for cm in cms]
    if any(s is None for s in actual) or not differences:
        raise ValueError("Both normal and abnormal samples are required for an ICBHI score comparison")
    write_json(a.output, {"baseline_score": actual[0], "proposed_score": actual[1],
                          "difference_percentage_points": actual[1] - actual[0],
                          "paired_difference_ci95": np.percentile(differences, [2.5, 97.5]).tolist(),
                          "valid_resamples": len(differences), "bootstrap_unit": "patient", "seed": a.seed,
                          "scope": "Conditional on these two fixed trained models; does not quantify training-seed uncertainty"})


if __name__ == "__main__":
    main()
