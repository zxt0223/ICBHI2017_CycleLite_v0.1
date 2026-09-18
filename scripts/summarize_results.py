"""Report all supplied seeds; do not select the best seed using test results."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icbhi_lite.utils import load_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", nargs="+", required=True, help="metrics.json files from icbhi_lite.evaluate")
    p.add_argument("--output", default="results_summary.json")
    a = p.parse_args()
    groups = defaultdict(list)
    for path in a.results:
        r = load_json(path)
        groups[(r["data_signature"], r["method_signature"], r["split"], r["refit"])].append((path, r))
    summary = []
    for key, values in groups.items():
        seeds = [r["seed"] for _, r in values]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seeds in a method/protocol group")
        result = {"data_signature": key[0], "method_signature": key[1], "split": key[2], "refit": key[3],
                  "seeds": seeds, "runs": len(values), "files": [p for p, _ in values],
                  "model_config": values[0][1]["model_config"], "parameters": values[0][1]["parameters"]}
        for metric in ["score", "specificity", "sensitivity", "macro_f1", "accuracy"]:
            x = [r[metric] for _, r in values]
            result[metric] = {"mean": float(np.mean(x)), "sample_std": float(np.std(x, ddof=1)) if len(x) > 1 else None,
                              "individual": x}
        summary.append(result)
    write_json(a.output, {"unit": "percent", "groups": summary, "note": "Mean/std across all supplied independent training seeds; no ensemble"})
    print(summary)


if __name__ == "__main__":
    main()
