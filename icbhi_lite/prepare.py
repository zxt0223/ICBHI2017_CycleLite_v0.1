"""Build an audited, immutable cycle cache using the supplied official split."""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import CLASS_NAMES
from .frontend import read_audio, condition_audio, extract_features
from .utils import load_config, digest, file_hash, write_json


def read_official_split(path):
    result = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\s,;]+", line)
        if len(parts) != 2:
            raise ValueError(f"Split line {number}: expected 'recording_or_patient train|test'")
        key, split = parts
        if key.lower() in {"filename", "recording", "patient", "patient_id"} and split.lower() in {"split", "set"}:
            continue
        key = Path(key).stem
        split = split.lower()
        if split not in {"train", "test"}:
            raise ValueError(f"Split line {number}: unsupported split {split!r}")
        if key in result:
            raise ValueError(f"Duplicate split key: {key}")
        result[key] = split
    if not result or set(result.values()) != {"train", "test"}:
        raise ValueError("Split file must contain both train and test")
    return result


def validate_patient_separation(rows):
    ownership = defaultdict(set)
    for row in rows:
        ownership[row["patient"]].add(row["official_split"])
    leaking = {p: sorted(v) for p, v in ownership.items() if len(v) != 1}
    if leaking:
        raise ValueError(f"Patient leakage in supplied split: {leaking}")


def select_validation_patients(rows, fraction, seed, attempts=2000):
    """Deterministic group holdout; uses ONLY official training labels."""
    if not 0 < fraction < 0.5:
        raise ValueError("Validation fraction must lie in (0, 0.5)")
    train = [r for r in rows if r["official_split"] == "train"]
    patients = sorted({r["patient"] for r in train})
    if len(patients) < 4:
        raise ValueError("Need at least four training patients for development holdout")
    counts = np.array([[sum(r["patient"] == p and r["label"] == c for r in train)
                        for c in range(4)] for p in patients])
    total = counts.sum(0)
    if (total == 0).any():
        raise ValueError("Official training set is missing a class")
    n_val = max(1, min(len(patients) - 1, round(len(patients) * fraction)))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(attempts):
        idx = rng.choice(len(patients), n_val, replace=False)
        vc = counts[idx].sum(0)
        if (vc == 0).any() or ((total - vc) == 0).any():
            continue
        cost = float(np.abs(vc / total - fraction).mean())
        if best is None or cost < best[0]:
            best = cost, idx.copy()
    if best is None:
        raise ValueError("Cannot create patient-disjoint train/val with all four classes; inspect data")
    return sorted(patients[i] for i in best[1])


def build(data_root, split_file, output, cfg, val_fraction=0.2, split_seed=42, allow_nonstandard=False):
    output, data_root = Path(output), Path(data_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Cache directory is not empty: {output}; use a new directory")
    mapping = read_official_split(split_file)
    wavs = sorted(data_root.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files beneath {data_root}")
    if len({w.stem for w in wavs}) != len(wavs):
        raise ValueError("Duplicate recording basenames found; point at one dataset copy")
    output.mkdir(parents=True, exist_ok=True)
    (output / "features").mkdir(exist_ok=True)
    rows, sources, seen_keys, boundary_fixes = [], [], set(), []
    for index, path in enumerate(wavs):
        patient = path.stem.split("_")[0]
        if not patient.isdigit() or len(path.stem.split("_")) < 5:
            raise ValueError(f"Unexpected ICBHI recording name: {path.name}")
        applicable = [k for k in (path.stem, patient) if k in mapping]
        if not applicable or len({mapping[k] for k in applicable}) != 1:
            raise ValueError(f"Missing or contradictory split for {path.name}")
        seen_keys.update(applicable)
        official = mapping[applicable[0]]
        ann = path.with_suffix(".txt")
        if not ann.is_file():
            raise FileNotFoundError(ann)
        x, original_sr = read_audio(path, cfg["sample_rate"])
        x = condition_audio(x, cfg)
        duration = len(x) / cfg["sample_rate"]
        sources.append({"recording": path.stem, "wav_sha256": file_hash(path),
                        "annotation_sha256": file_hash(ann), "original_sr": original_sr})
        lines = [s for s in ann.read_text().splitlines() if s.strip()]
        if not lines:
            raise ValueError(f"No respiratory cycles: {ann}")
        for ci, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Malformed cycle annotation: {ann}:{ci+1}")
            start, end, crackle, wheeze = map(float, parts)
            if not np.isfinite([start, end, crackle, wheeze]).all():
                raise ValueError(f"Nonfinite annotation: {ann}:{ci+1}")
            if crackle not in (0, 1) or wheeze not in (0, 1):
                raise ValueError(f"Nonbinary event annotation: {ann}:{ci+1}")
            if start < -0.02 or end > duration + 0.02 or end <= start:
                raise ValueError(f"Cycle boundary outside recording: {ann}:{ci+1}")
            a, b = max(0, round(start * cfg["sample_rate"])), min(len(x), round(end * cfg["sample_rate"]))
            if b <= a:
                raise ValueError("Empty respiratory cycle")
            if start < 0 or end > duration:
                boundary_fixes.append(f"{path.stem}:{ci}")
            cycle_id = f"{path.stem}__{ci:03d}"
            feat = extract_features(x[a:b], cfg)
            relative = f"features/{cycle_id}.npy"
            np.save(output / relative, feat.astype(np.float16), allow_pickle=False)
            rows.append({"cycle_id": cycle_id, "recording": path.stem, "patient": patient,
                         "device": path.stem.split("_")[-1], "label": int(crackle) + 2 * int(wheeze),
                         "start": start, "end": end, "samples": b - a, "frames": feat.shape[-1],
                         "official_split": official, "feature_path": relative})
        if (index + 1) % 50 == 0 or index == len(wavs) - 1:
            print(f"Preprocessed recordings {index+1}/{len(wavs)}; cycles {len(rows)}", flush=True)
    if set(mapping) - seen_keys:
        raise ValueError(f"Split entries have no audio: {sorted(set(mapping) - seen_keys)[:10]}")
    validate_patient_separation(rows)
    expected = {"train": 4142, "test": 2756}
    actual = dict(Counter(r["official_split"] for r in rows))
    full = actual == expected and len(wavs) == 920 and len({r["patient"] for r in rows}) == 126
    if not full and not allow_nonstandard:
        raise ValueError(f"Expected 920 recordings / 126 patients / {expected}, got {len(wavs)} / "
                         f"{len({r['patient'] for r in rows})} / {actual}. Inspect dataset; "
                         "--allow-nonstandard-counts is for synthetic/subset development only.")
    val_patients = select_validation_patients(rows, val_fraction, split_seed)
    for r in rows:
        r["split"] = "test" if r["official_split"] == "test" else ("val" if r["patient"] in val_patients else "train")
    report = {s: {"cycles": sum(r["split"] == s for r in rows),
                  "patients": len({r["patient"] for r in rows if r["split"] == s}),
                  "classes": [sum(r["split"] == s and r["label"] == c for r in rows) for c in range(4)]}
              for s in ("train", "val", "test")}
    manifest = {"schema_version": 1, "class_names": CLASS_NAMES, "frontend": cfg,
                "official_split_sha256": file_hash(split_file), "official_counts_match": full,
                "official_split_origin": str(Path(split_file).resolve()),
                "protocol_note": "User-supplied split; counts and patient disjointness checked; verify source against official release.",
                "split_seed": split_seed, "val_fraction": val_fraction, "val_patients": val_patients,
                "report": report, "boundary_clamps_le_20ms": boundary_fixes, "sources": sources, "rows": rows}
    manifest["data_signature"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    write_json(output / "audit.json", {k: manifest[k] for k in ["class_names", "report", "official_counts_match",
               "official_split_sha256", "data_signature", "val_patients", "boundary_clamps_le_20ms", "protocol_note"]})
    print(f"Audit saved: {output / 'audit.json'}", flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--official-split", required=True)
    p.add_argument("--output", default="cache/icbhi")
    p.add_argument("--config", default="configs/proposed.yaml")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--allow-nonstandard-counts", action="store_true")
    a = p.parse_args()
    build(a.data_root, a.official_split, a.output, load_config(a.config)["frontend"],
          a.val_fraction, a.split_seed, a.allow_nonstandard_counts)


if __name__ == "__main__":
    main()
