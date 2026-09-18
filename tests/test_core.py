import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from icbhi_lite.data import fit_statistics
from icbhi_lite.frontend import extract_features, window_starts, condition_audio, window_geometry
from icbhi_lite.losses import class_weights, supervised_loss
from icbhi_lite.metrics import evaluate_labels, from_confusion, patient_bootstrap
from icbhi_lite.model import build_model, InferenceModel
from icbhi_lite.prepare import read_official_split, validate_patient_separation, select_validation_patients
from icbhi_lite.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class ProtocolTests(unittest.TestCase):
    def test_abnormal_confusion_is_not_a_true_positive(self):
        result = evaluate_labels([0, 1, 2, 3], [0, 2, 3, 1])
        self.assertEqual(result["specificity"], 100)
        self.assertEqual(result["sensitivity"], 0)
        self.assertEqual(result["score"], 50)

    def test_metric_uses_pooled_abnormal_not_macro_recall(self):
        cm = np.diag([80, 20, 10, 2])
        cm[0, 1], cm[1, 0], cm[2, 0], cm[3, 0] = 20, 80, 10, 8
        result = from_confusion(cm)
        self.assertAlmostEqual(result["score"], 50 * (0.8 + 32 / 130))

    def test_undefined_subgroup_score_is_not_fabricated(self):
        self.assertIsNone(evaluate_labels([1, 2], [1, 2])["score"])

    def test_split_duplicates_and_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "split.txt"
            path.write_text("101 train\n102 test\n")
            self.assertEqual(read_official_split(path)["101"], "train")
            path.write_text("101 train\n101 test\n")
            with self.assertRaises(ValueError):
                read_official_split(path)
        with self.assertRaises(ValueError):
            validate_patient_separation([{"patient": "101", "official_split": s} for s in ["train", "test"]])

    def test_validation_patients_do_not_depend_on_test_labels(self):
        rows = [{"patient": str(p), "official_split": "train" if p < 10 else "test", "label": c}
                for p in range(12) for c in range(4)]
        first = select_validation_patients(rows, 0.2, 42)
        for r in rows:
            if r["official_split"] == "test":
                r["label"] = 0
        self.assertEqual(first, select_validation_patients(rows, 0.2, 42))
        self.assertFalse(set(first) & {"10", "11"})

    def test_statistics_use_only_selected_training_cycles(self):
        with tempfile.TemporaryDirectory() as d:
            for name, v in [("train_a", 1), ("train_b", 3), ("test", 10000)]:
                np.save(Path(d) / (name + ".npy"), np.full((4, 3, 4), v, dtype=np.float32))
            rows = [{"cycle_id": n, "feature_path": n + ".npy"} for n in ["train_a", "train_b"]]
            stats = fit_statistics(d, rows, [0, 1])
            np.testing.assert_allclose(stats["mean"], 2)
            np.testing.assert_allclose(stats["std"], 1)
            self.assertNotIn("test", stats["fit_cycle_ids"])

    def test_patient_bootstrap_reproducible(self):
        a = patient_bootstrap([0, 1, 0, 2], [0, 1, 0, 0], ["a", "a", "b", "b"], 100, 7)
        b = patient_bootstrap([0, 1, 0, 2], [0, 1, 0, 0], ["a", "a", "b", "b"], 100, 7)
        self.assertEqual(a, b)
        self.assertEqual(a["bootstrap_unit"], "patient")


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/proposed.yaml")["frontend"]

    def test_silence_and_tiny_cycles_finite(self):
        for n in [1, 20, 159, 1600]:
            x = condition_audio(np.zeros(n, np.float32), self.cfg)
            features = extract_features(x, self.cfg)
            self.assertEqual(features.shape, (4, 64, n // 160 + 1))
            self.assertTrue(np.isfinite(features).all())

    def test_all_frames_covered_without_stretching(self):
        for n in [1, 200, 400, 401, 1001]:
            covered = np.zeros(n, bool)
            for s in window_starts(n, 400, 200):
                covered[s:s+400] = True
            self.assertTrue(covered.all())

    def test_exact_four_second_cycle_is_one_window(self):
        cfg = load_config(ROOT / "configs/proposed.yaml")
        width, stride = window_geometry(cfg)
        n_frames = round(cfg["model"]["window_seconds"] * self.cfg["sample_rate"]) // self.cfg["hop_length"] + 1
        self.assertEqual(window_starts(n_frames, width, stride), [0])


class NetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.cfg = load_config(ROOT / "configs/proposed.yaml")

    def test_real_network_has_finite_gradients(self):
        model = build_model(self.cfg).train()
        x = torch.randn(2, 2, 4, 64, 101)
        lengths = torch.tensor([[101, 52], [80, 0]])
        out = model(x, lengths)
        loss, _ = supervised_loss(out, torch.tensor([1, 3]), class_weights([0, 1, 2, 3]), self.cfg["loss"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(model.event_attention.weight.grad.abs().sum().item(), 0)

    def test_padded_values_and_absent_windows_cannot_change_predictions(self):
        model = build_model(self.cfg).eval()
        x = torch.randn(1, 1, 4, 64, 101)
        lengths = torch.tensor([[55]])
        corrupt = x.clone()
        corrupt[..., 55:] = 1e5
        extra = torch.cat([corrupt, torch.full_like(corrupt, -1e5)], dim=1)
        with torch.inference_mode():
            expected = model(x, lengths)["logits"]
            actual = model(extra, torch.tensor([[55, 0]]))["logits"]
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)

    def test_window_permutation_preserves_cycle_prediction(self):
        model = build_model(self.cfg).eval()
        x = torch.randn(1, 3, 4, 64, 80)
        lengths = torch.tensor([[80, 35, 70]])
        order = [2, 0, 1]
        with torch.inference_mode():
            a = model(x, lengths)["logits"]
            b = model(x[:, order], lengths[:, order])["logits"]
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_scripted_network_matches_python(self):
        model = InferenceModel(build_model(self.cfg).eval()).eval()
        script = torch.jit.freeze(torch.jit.script(model))
        with torch.inference_mode():
            for b, w in [(1, 1), (2, 3)]:
                x = torch.randn(b, w, 4, 64, 101)
                lengths = torch.full((b, w), 101)
                lengths[0, 0] = 35
                if w > 1:
                    lengths[0, -1] = 0
                torch.testing.assert_close(script(x, lengths), model(x, lengths), rtol=1e-4, atol=1e-5)

    def test_knowledge_distillation_backpropagates(self):
        cfg = dict(self.cfg["loss"], kd_weight=0.3)
        logits = torch.randn(4, 4, requires_grad=True)
        outputs = {"logits": logits, "event_logits": torch.randn(4, 2, requires_grad=True)}
        loss, parts = supervised_loss(outputs, torch.arange(4), class_weights([0, 1, 2, 3]), cfg, torch.randn(4, 4))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(parts["kd"], 0)


if __name__ == "__main__":
    unittest.main()
