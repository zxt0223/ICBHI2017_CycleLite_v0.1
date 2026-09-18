"""Sequential, validation-only ablations; official test is never called here."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="cache/icbhi")
    p.add_argument("--configs", nargs="+", default=["baseline", "ablation_frontend", "ablation_axis", "ablation_pooling", "proposed"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--output-root", default="runs/ablations")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    for name in a.configs:
        path = ROOT / "configs" / (name + ".yaml")
        if not path.exists() or name == "distill":
            raise ValueError("Use a provided non-distillation config; distillation requires an explicit teacher")
        for seed in a.seeds:
            cmd = [sys.executable, "-m", "icbhi_lite.train", "--config", str(path), "--cache", a.cache,
                   "--output", str(Path(a.output_root) / f"{name}_seed{seed}"), "--seed", str(seed),
                   "--device", a.device, "--workers", str(a.workers)]
            if a.epochs:
                cmd += ["--epochs", str(a.epochs)]
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
