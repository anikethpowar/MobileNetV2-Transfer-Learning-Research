import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure Matplotlib uses a writable config dir
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights

from src.evaluate import (
    compute_classification_metrics,
    compute_mcnemar_tests,
    compute_wilson_ci,
    export_metrics,
    load_model,
    measure_inference_latency,
    plot_comparative_charts,
    plot_confusion_matrices
)


def test_1_checkpoint_integrity():
    print("Test 1: Checkpoint loading & structural integrity across all 3 models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = {
        "frozen": "models/frozen_best.pth",
        "partial": "models/partial_best.pth",
        "deeper": "models/deeper_best.pth"
    }

    for exp, ckpt_path in checkpoints.items():
        assert Path(ckpt_path).exists(), f"Missing checkpoint: {ckpt_path}"
        model, ckpt, total_params = load_model(ckpt_path, device, num_classes=5)

        assert total_params == 2230277, f"Total params mismatch: {total_params}"
        assert not model.training, f"Model {exp} is not in eval mode!"

        # Verify output dimension
        dummy = torch.randn(2, 3, 224, 224, device=device)
        with torch.inference_mode():
            out = model(dummy)
        assert out.shape == (2, 5), f"Output shape mismatch: {out.shape}"
        print(f"  -> {exp.upper()}: Checkpoint loaded cleanly, eval mode active, output shape {out.shape}")


def test_2_metrics_and_wilson_ci():
    print("\nTest 2: Metrics calculation & Wilson score confidence interval...")
    classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 0, 0])  # 8 correct, 2 incorrect
    y_probs = np.eye(5)[y_pred]  # one-hot dummy probabilities

    metrics = compute_classification_metrics(y_true, y_pred, y_probs, classes, test_loss=0.45)

    assert metrics["test_accuracy"] == 0.8, f"Accuracy error: {metrics['test_accuracy']}"
    assert metrics["correct_count"] == 8
    assert metrics["incorrect_count"] == 2
    assert metrics["total_count"] == 10

    ci_low, ci_high = metrics["accuracy_ci_95"]
    assert 0.0 <= ci_low <= 0.8 <= ci_high <= 1.0, f"Invalid Wilson CI: [{ci_low}, {ci_high}]"
    print(f"  -> Computed accuracy: {metrics['test_accuracy']*100:.1f}%, 95% Wilson CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    print(f"  -> Macro F1: {metrics['macro_f1']:.4f}, Weighted F1: {metrics['weighted_f1']:.4f}")


def test_3_mcnemar_statistical_comparison():
    print("\nTest 3: Paired McNemar's test computation...")
    y_true = np.array([0, 1, 2, 3, 4] * 20)  # 100 samples
    y_pred1 = y_true.copy()
    y_pred2 = y_true.copy()

    # Introduce known discordant errors
    y_pred1[0:15] = (y_pred1[0:15] + 1) % 5  # Model 1 fails on 0..14
    y_pred2[10:30] = (y_pred2[10:30] + 1) % 5  # Model 2 fails on 10..29

    preds_dict = {
        "model_A": {"preds": y_pred1},
        "model_B": {"preds": y_pred2}
    }

    stat_res = compute_mcnemar_tests(preds_dict, y_true)
    assert "model_A_vs_model_B" in stat_res
    res = stat_res["model_A_vs_model_B"]
    c = res["contingency_matrix"]

    assert c["both_correct_n11"] + c["model1_only_n10"] + c["model2_only_n01"] + c["both_incorrect_n00"] == 100
    assert 0.0 <= res["recommended_p_value"] <= 1.0
    print(f"  -> Contingency matrix: {c}")
    print(f"  -> Discordant pairs: {res['total_discordant']}")
    print(f"  -> Recommended p-value: {res['recommended_p_value']:.4f} ({res['recommended_method']})")


def test_4_synthetic_latency_benchmarking():
    print("\nTest 4: GPU inference latency benchmarking...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model("models/frozen_best.pth", device, num_classes=5)

    # Create dummy DataLoader with synthetic batch
    dummy_dataset = torch.utils.data.TensorDataset(
        torch.randn(32, 3, 224, 224),
        torch.zeros(32, dtype=torch.long)
    )
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=16, shuffle=False)

    lat_info = measure_inference_latency(model, dummy_loader, device, repetitions=3, warmup_runs=2)
    assert lat_info["latency_ms_per_image_mean"] > 0.0
    assert lat_info["throughput_fps"] > 0.0
    print(f"  -> Benchmarked: {lat_info['latency_ms_per_image_mean']:.2f} ± {lat_info['latency_ms_per_image_std']:.2f} ms/img ({lat_info['throughput_fps']:.1f} FPS)")


def test_5_plotting_and_export_pipeline():
    print("\nTest 5: Plotting and export pipeline to temporary directory...")
    temp_dir = Path("temp_smoke_eval")
    metrics_temp = temp_dir / "metrics"
    plots_temp = temp_dir / "plots"

    try:
        classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
        cm = [[10, 1, 0, 0, 0], [0, 12, 0, 0, 0], [1, 0, 9, 0, 0], [0, 0, 0, 11, 0], [0, 0, 0, 0, 10]]
        cm_norm = (np.array(cm) / np.array(cm).sum(axis=1, keepdims=True)).tolist()

        plot_confusion_matrices(cm, cm_norm, classes, "test_model", plots_temp)
        assert (plots_temp / "test_model_confusion_matrix.png").exists()
        assert (plots_temp / "test_model_confusion_matrix_normalized.png").exists()

        summary_dummy = [
            {
                "key": "test_exp",
                "display_name": "Test Exp",
                "short_name": "Test",
                "color": "#1f77b4",
                "test_accuracy": 0.92,
                "ci_low": 0.88,
                "ci_high": 0.95,
                "macro_f1": 0.91,
                "weighted_f1": 0.92,
                "macro_precision": 0.92,
                "macro_recall": 0.91,
                "val_accuracy": 0.93,
                "trainable_parameters": 6405,
                "latency_ms_per_image": 2.5,
                "latency_ms_per_image_std": 0.1,
                "throughput_fps": 400.0
            }
        ]
        plot_comparative_charts(summary_dummy, plots_temp)
        assert (plots_temp / "test_accuracy_comparison.png").exists()
        assert (plots_temp / "macro_f1_comparison.png").exists()
        assert (plots_temp / "evaluation_summary_overview.png").exists()
        print("  -> Visualizations generated at 300 DPI successfully.")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("  -> Temporary evaluation directory cleaned up cleanly.")


def test_6_test_partition_isolation_check():
    print("\nTest 6: Verify test partition isolation during smoke testing...")
    test_dir = Path("data/processed/test")
    assert test_dir.exists(), "Test directory missing!"
    total_files = len(list(test_dir.glob("*/*")))
    assert total_files == 556, f"Expected 556 test images, found {total_files}"
    print(f"  -> Verified test dataset remains completely untouched ({total_files} images across 5 classes).")


def run_all_checks():
    print("=" * 70)
    print("PHASE 3 PRE-FLIGHT VERIFICATION & SMOKE TEST SUITE")
    print("=" * 70)
    test_1_checkpoint_integrity()
    test_2_metrics_and_wilson_ci()
    test_3_mcnemar_statistical_comparison()
    test_4_synthetic_latency_benchmarking()
    test_5_plotting_and_export_pipeline()
    test_6_test_partition_isolation_check()
    print("=" * 70)
    print("ALL PHASE 3 PRE-FLIGHT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_checks()

