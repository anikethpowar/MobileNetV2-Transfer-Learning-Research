import argparse
import csv
import json
import os
import time
from pathlib import Path

# Ensure Matplotlib uses a writable config dir in all environments
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


EXPERIMENT_METADATA = {
    "frozen": {
        "display_name": "Experiment A (Frozen Backbone)",
        "short_name": "Exp A: Frozen",
        "checkpoint": "models/frozen_best.pth",
        "unfrozen_blocks": 0,
        "color": "#1f77b4"
    },
    "partial": {
        "display_name": "Experiment B (Partial Fine-Tuning)",
        "short_name": "Exp B: Partial (2 blk)",
        "checkpoint": "models/partial_best.pth",
        "unfrozen_blocks": 2,
        "color": "#ff7f0e"
    },
    "deeper": {
        "display_name": "Experiment C (Deeper Fine-Tuning)",
        "short_name": "Exp C: Deeper (5 blk)",
        "checkpoint": "models/deeper_best.pth",
        "unfrozen_blocks": 5,
        "color": "#2ca02c"
    }
}


def compute_wilson_ci(k, n, confidence=0.95):
    """
    Compute two-sided Wilson score confidence interval for a proportion k/n.
    Particularly well-suited for classification accuracy near upper boundaries (90-95%).
    """
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p_hat = k / n
    denominator = 1.0 + (z ** 2) / n
    centre_adj = p_hat + (z ** 2) / (2.0 * n)
    se = z * np.sqrt((p_hat * (1.0 - p_hat) + (z ** 2) / (4.0 * n)) / n)
    lower = max(0.0, (centre_adj - se) / denominator)
    upper = min(1.0, (centre_adj + se) / denominator)
    return float(lower), float(upper)


def load_model(checkpoint_path, device, num_classes=5):
    """
    Reconstruct MobileNetV2 architecture with identical 5-class linear classifier
    and restore the saved state dictionary strictly.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file '{checkpoint_path}' does not exist! "
            "Verify all experiments have completed and checkpoints are saved."
        )

    # 1. Reconstruct MobileNetV2 architecture
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    # 2. Load checkpoint dictionary
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # 3. Verify classifier output dimension
    classifier_weight = state_dict.get("classifier.1.weight")
    if classifier_weight is None:
        raise KeyError(f"Expected key 'classifier.1.weight' not found in '{checkpoint_path}'.")
    if classifier_weight.shape[0] != num_classes:
        raise ValueError(
            f"Classifier dimension mismatch in '{checkpoint_path}': "
            f"expected {num_classes}, found {classifier_weight.shape[0]}."
        )

    # 4. Strict state dict restoration
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return model, checkpoint, total_params


def get_eval_loader(data_dir="data/processed/test", batch_size=16, num_workers=2):
    """
    Construct a deterministic evaluation DataLoader matching the ImageNet evaluation transform.
    Shuffle is strictly False to guarantee identical sample sequence across models.
    """
    eval_path = Path(data_dir)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation directory '{data_dir}' not found.")

    weights = MobileNet_V2_Weights.DEFAULT
    eval_transform = weights.transforms()

    dataset = datasets.ImageFolder(root=str(eval_path), transform=eval_transform)

    pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Extract relative paths as stable image identifiers
    image_rel_paths = [
        os.path.relpath(path, str(eval_path)) for path, _ in dataset.samples
    ]

    return loader, dataset.classes, image_rel_paths


def run_inference(model, loader, device, classes):
    """
    Execute deterministic forward pass across the entire evaluation dataset under torch.inference_mode().
    Accumulates predictions, ground-truth labels, full softmax probabilities, and cross-entropy loss.
    """
    criterion = nn.CrossEntropyLoss(reduction="sum")

    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return (
        np.array(all_targets, dtype=int),
        np.array(all_preds, dtype=int),
        np.array(all_probs, dtype=float),
        avg_loss
    )


def measure_inference_latency(model, loader, device, repetitions=5, warmup_runs=5):
    """
    Benchmark inference latency on the evaluation dataset.
    Follows rigorous GPU benchmarking protocols:
    - GPU warm-up prior to timing.
    - Explicit CUDA synchronization before and after timed intervals.
    - Excludes model loading, initialization, and data transfer setup.
    - Multiple repetitions reporting mean and standard deviation.
    """
    model.eval()

    # 1. Warm-up
    with torch.inference_mode():
        warmup_batches = 0
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            _ = model(images)
            warmup_batches += 1
            if warmup_batches >= warmup_runs:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()

    # 2. Repeated measurement runs
    durations = []
    total_samples = len(loader.dataset)

    with torch.inference_mode():
        for _ in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                _ = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            durations.append(end_time - start_time)

    durations = np.array(durations)
    mean_duration = float(np.mean(durations))
    std_duration = float(np.std(durations))

    # Latency per image in milliseconds
    latency_ms_per_image = (durations / total_samples) * 1000.0
    mean_latency_ms = float(np.mean(latency_ms_per_image))
    std_latency_ms = float(np.std(latency_ms_per_image))

    # Throughput (Images per second)
    throughput_fps = total_samples / mean_duration if mean_duration > 0 else 0.0

    return {
        "repetitions": repetitions,
        "total_time_mean_s": mean_duration,
        "total_time_std_s": std_duration,
        "latency_ms_per_image_mean": mean_latency_ms,
        "latency_ms_per_image_std": std_latency_ms,
        "throughput_fps": float(throughput_fps)
    }


def compute_classification_metrics(y_true, y_pred, y_probs, classes, test_loss):
    """
    Calculate overall and per-class classification metrics via scikit-learn.
    zero_division=0 is used to handle edge cases without generating warnings or errors.
    """
    num_classes = len(classes)
    total_count = len(y_true)
    correct_count = int(np.sum(y_true == y_pred))
    incorrect_count = total_count - correct_count

    acc = float(accuracy_score(y_true, y_pred))
    ci_low, ci_high = compute_wilson_ci(correct_count, total_count, confidence=0.95)

    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    class_supports = [int(np.sum(y_true == i)) for i in range(num_classes)]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)), normalize="true")

    per_class_report = []
    for i, cls_name in enumerate(classes):
        per_class_report.append({
            "class": cls_name,
            "precision": float(per_class_prec[i]),
            "recall": float(per_class_rec[i]),
            "f1_score": float(per_class_f1[i]),
            "support": class_supports[i]
        })

    return {
        "test_accuracy": acc,
        "accuracy_ci_95": [ci_low, ci_high],
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "test_loss": float(test_loss),
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "total_count": total_count,
        "per_class_report": per_class_report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist()
    }


def compute_mcnemar_tests(predictions_dict, y_true):
    """
    Perform paired classification comparisons using McNemar's test for all model pairs.
    Reports 2x2 contingency matrix, exact binomial p-value, and Edwards continuity-corrected chi-square.
    """
    exp_keys = list(predictions_dict.keys())
    paired_results = {}

    for i in range(len(exp_keys)):
        for j in range(i + 1, len(exp_keys)):
            exp1 = exp_keys[i]
            exp2 = exp_keys[j]
            pair_key = f"{exp1}_vs_{exp2}"

            preds1 = np.array(predictions_dict[exp1]["preds"])
            preds2 = np.array(predictions_dict[exp2]["preds"])

            corr1 = (preds1 == y_true)
            corr2 = (preds2 == y_true)

            n11 = int(np.sum(corr1 & corr2))    # Both correct
            n10 = int(np.sum(corr1 & ~corr2))   # Exp1 correct, Exp2 incorrect
            n01 = int(np.sum(~corr1 & corr2))   # Exp1 incorrect, Exp2 correct
            n00 = int(np.sum(~corr1 & ~corr2))  # Both incorrect

            disc = n10 + n01  # Discordant pairs

            if disc == 0:
                p_exact = 1.0
                chi2_stat = 0.0
                p_asymp = 1.0
                rec_method = "identical_predictions"
                rec_p = 1.0
            else:
                # Exact two-sided binomial test on discordant pairs
                res_binom = stats.binomtest(min(n10, n01), n=disc, p=0.5, alternative="two-sided")
                p_exact = float(res_binom.pvalue)

                # Edwards continuity-corrected Chi-Square statistic
                chi2_stat = float(((abs(n10 - n01) - 1.0) ** 2) / disc)
                p_asymp = float(stats.chi2.sf(chi2_stat, df=1))

                rec_method = "exact_binomial" if disc < 25 else "mcnemar_chi2_cc"
                rec_p = p_exact if disc < 25 else p_asymp

            stat_sig_05 = bool(rec_p < 0.05)
            stat_sig_01 = bool(rec_p < 0.01)

            paired_results[pair_key] = {
                "model_1": exp1,
                "model_2": exp2,
                "contingency_matrix": {
                    "both_correct_n11": n11,
                    "model1_only_n10": n10,
                    "model2_only_n01": n01,
                    "both_incorrect_n00": n00
                },
                "total_discordant": disc,
                "mcnemar_chi2_stat": chi2_stat,
                "p_value_asymptotic_cc": p_asymp,
                "p_value_exact_binomial": p_exact,
                "recommended_method": rec_method,
                "recommended_p_value": rec_p,
                "statistically_significant_alpha_0_05": stat_sig_05,
                "statistically_significant_alpha_0_01": stat_sig_01,
                "interpretation": (
                    f"Difference is statistically significant at alpha=0.05 (p={rec_p:.4f})"
                    if stat_sig_05 else
                    f"Difference is not statistically significant at alpha=0.05 (p={rec_p:.4f})"
                )
            }

    return paired_results


def plot_confusion_matrices(cm, cm_norm, classes, exp_name, output_dir):
    """
    Generate and save raw counts and row-normalized confusion matrices as 300 DPI PNGs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = np.array(cm)
    cm_norm = np.array(cm_norm)

    # 1. Raw Count Confusion Matrix
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        linewidths=0.5,
        linecolor="gray"
    )
    plt.title(f"Confusion Matrix (Counts) — {exp_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="medium")
    plt.ylabel("True Class", fontsize=11, fontweight="medium")
    plt.tight_layout()
    raw_path = output_dir / f"{exp_name}_confusion_matrix.png"
    plt.savefig(raw_path, dpi=300)
    plt.close()
    print(f"Saved: {raw_path}")

    # 2. Normalized Confusion Matrix (Proportions)
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="gray"
    )
    plt.title(f"Normalized Confusion Matrix (Proportions) — {exp_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="medium")
    plt.ylabel("True Class", fontsize=11, fontweight="medium")
    plt.tight_layout()
    norm_path = output_dir / f"{exp_name}_confusion_matrix_normalized.png"
    plt.savefig(norm_path, dpi=300)
    plt.close()
    print(f"Saved: {norm_path}")


def plot_combined_confusion_matrices(cms, cms_norm, classes, exp_keys, output_dir):
    """
    Generate a 1x3 side-by-side comparative confusion matrix figure across all 3 models.
    """
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)

    for i, exp in enumerate(exp_keys):
        ax = axes[i]
        sns.heatmap(
            np.array(cms_norm[exp]),
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar=(i == 2),
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            linewidths=0.5,
            linecolor="gray"
        )
        meta = EXPERIMENT_METADATA[exp]
        ax.set_title(meta["display_name"], fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Class", fontsize=10)
        if i == 0:
            ax.set_ylabel("True Class", fontsize=10)
        else:
            ax.set_ylabel("")

    fig.suptitle("Comparative Normalized Confusion Matrices Across Transfer-Learning Strategies", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    combined_path = output_dir / "confusion_matrices_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {combined_path}")


def plot_comparative_charts(summary_records, output_dir):
    """
    Generate the 6 required comparative plots and a comprehensive summary dashboard.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [r["short_name"] for r in summary_records]
    colors = [r["color"] for r in summary_records]

    # 1. Test Accuracy with 95% Wilson Confidence Intervals
    plt.figure(figsize=(7, 5), dpi=300)
    accs = [r["test_accuracy"] for r in summary_records]
    yerr_low = [r["test_accuracy"] - r["ci_low"] for r in summary_records]
    yerr_high = [r["ci_high"] - r["test_accuracy"] for r in summary_records]

    bars = plt.bar(
        labels,
        accs,
        color=colors,
        width=0.5,
        edgecolor="black",
        linewidth=0.8,
        yerr=[yerr_low, yerr_high],
        capsize=5,
        error_kw={"elinewidth": 1.2, "ecolor": "black"}
    )
    for bar in bars:
        h = bar.get_height()
        plt.annotate(
            f"{h*100:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10
        )
    ymin = max(0.0, min(r["ci_low"] for r in summary_records) - 0.08)
    plt.ylim(ymin, 1.02)
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    plt.ylabel("Test Accuracy", fontsize=11, fontweight="medium")
    plt.title("Test Accuracy Comparison with 95% Wilson Confidence Intervals", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p1 = output_dir / "test_accuracy_comparison.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # 2. Macro-F1 Comparison
    plt.figure(figsize=(7, 5), dpi=300)
    f1s = [r["macro_f1"] for r in summary_records]
    bars = plt.bar(labels, f1s, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        plt.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10
        )
    ymin = max(0.0, min(f1s) - 0.08)
    plt.ylim(ymin, 1.02)
    plt.ylabel("Macro-Averaged F1-Score", fontsize=11, fontweight="medium")
    plt.title("Macro-Averaged F1-Score Comparison", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p2 = output_dir / "macro_f1_comparison.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # 3. Macro Precision vs Macro Recall Grouped Bar Chart
    plt.figure(figsize=(8, 5), dpi=300)
    x = np.arange(len(labels))
    width = 0.35
    precs = [r["macro_precision"] for r in summary_records]
    recs = [r["macro_recall"] for r in summary_records]

    rects1 = plt.bar(x - width/2, precs, width, label="Macro Precision", color="#4a7bb0", edgecolor="black", linewidth=0.8)
    rects2 = plt.bar(x + width/2, recs, width, label="Macro Recall", color="#80b1d3", edgecolor="black", linewidth=0.8)

    for bar in rects1:
        h = bar.get_height()
        plt.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in rects2:
        h = bar.get_height()
        plt.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ymin = max(0.0, min(min(precs), min(recs)) - 0.08)
    plt.ylim(ymin, 1.02)
    plt.xticks(x, labels)
    plt.ylabel("Score", fontsize=11, fontweight="medium")
    plt.title("Macro Precision and Macro Recall Comparison Across Models", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend(framealpha=0.95, loc="lower right")
    plt.tight_layout()
    p3 = output_dir / "precision_recall_comparison.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")

    # 4. Validation Accuracy vs Test Accuracy (Generalization Gap)
    plt.figure(figsize=(8, 5), dpi=300)
    val_accs = [r["val_accuracy"] for r in summary_records]
    test_accs = [r["test_accuracy"] for r in summary_records]

    rects1 = plt.bar(x - width/2, val_accs, width, label="Best Validation Accuracy", color="#5c9eb8", edgecolor="black", linewidth=0.8)
    rects2 = plt.bar(x + width/2, test_accs, width, label="Final Test Accuracy", color="#e87d5d", edgecolor="black", linewidth=0.8)

    for bar in rects1:
        h = bar.get_height()
        plt.annotate(f"{h*100:.2f}%", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in rects2:
        h = bar.get_height()
        plt.annotate(f"{h*100:.2f}%", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ymin = max(0.0, min(min(val_accs), min(test_accs)) - 0.08)
    plt.ylim(ymin, 1.02)
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    plt.xticks(x, labels)
    plt.ylabel("Accuracy", fontsize=11, fontweight="medium")
    plt.title("Generalization Assessment: Validation Accuracy vs. Test Accuracy", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend(framealpha=0.95, loc="lower right")
    plt.tight_layout()
    p4 = output_dir / "val_vs_test_accuracy.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved: {p4}")

    # 5. Trainable Parameters vs Test Accuracy (Parameter Efficiency)
    plt.figure(figsize=(8, 5), dpi=300)
    params = [r["trainable_parameters"] for r in summary_records]

    for r in summary_records:
        plt.scatter(
            r["trainable_parameters"],
            r["test_accuracy"],
            color=r["color"],
            s=120,
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
            label=f"{r['short_name']} ({r['trainable_parameters']:,} params)"
        )
        plt.annotate(
            f"{r['short_name']}\n{r['test_accuracy']*100:.2f}%",
            xy=(r["trainable_parameters"], r["test_accuracy"]),
            xytext=(10, -5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold"
        )

    plt.xscale("log")
    plt.xlabel("Trainable Parameters (log scale)", fontsize=11, fontweight="medium")
    plt.ylabel("Test Accuracy", fontsize=11, fontweight="medium")
    plt.title("Parameter Efficiency: Trainable Parameters vs. Test Accuracy", fontsize=12, fontweight="bold", pad=12)
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ymin = max(0.0, min(test_accs) - 0.05)
    plt.ylim(ymin, max(test_accs) + 0.03)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(framealpha=0.95, loc="lower right")
    plt.tight_layout()
    p5 = output_dir / "parameters_vs_test_accuracy.png"
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"Saved: {p5}")

    # 6. Inference Latency & Throughput Comparison
    plt.figure(figsize=(8, 5), dpi=300)
    lats = [r["latency_ms_per_image"] for r in summary_records]
    lat_errs = [r["latency_ms_per_image_std"] for r in summary_records]

    bars = plt.bar(labels, lats, yerr=lat_errs, color=colors, width=0.5, edgecolor="black", linewidth=0.8, capsize=5)
    for bar, r in zip(bars, summary_records):
        h = bar.get_height()
        plt.annotate(
            f"{h:.2f} ms\n({r['throughput_fps']:.1f} FPS)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9
        )
    plt.ylim(0, max(lats) * 1.35)
    plt.ylabel("Inference Latency per Image (ms)", fontsize=11, fontweight="medium")
    plt.title("Computational Inference Performance Comparison on RTX 3050 GPU", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p6 = output_dir / "inference_latency_comparison.png"
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"Saved: {p6}")

    # 7. Comprehensive Overview Dashboard (2x3 Grid)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    # (0,0) Test Accuracy
    ax = axes[0, 0]
    b = ax.bar(labels, accs, color=colors, width=0.5, edgecolor="black", linewidth=0.8, yerr=[yerr_low, yerr_high], capsize=4)
    for bar in b:
        h = bar.get_height()
        ax.annotate(f"{h*100:.2f}%", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylim(ymin, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title("Test Accuracy (95% Wilson CI)", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # (0,1) Macro F1 & Weighted F1
    ax = axes[0, 1]
    wf1s = [r["weighted_f1"] for r in summary_records]
    ax.bar(x - width/2, f1s, width, label="Macro F1", color="#6baed6", edgecolor="black", linewidth=0.8)
    ax.bar(x + width/2, wf1s, width, label="Weighted F1", color="#3182bd", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(min(f1s) - 0.08, 1.02)
    ax.set_title("Macro & Weighted F1-Scores", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")

    # (0,2) Generalization Gap (Val vs Test)
    ax = axes[0, 2]
    ax.bar(x - width/2, val_accs, width, label="Validation Acc", color="#5c9eb8", edgecolor="black", linewidth=0.8)
    ax.bar(x + width/2, test_accs, width, label="Test Acc", color="#e87d5d", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(min(min(val_accs), min(test_accs)) - 0.08, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title("Val Accuracy vs. Test Accuracy", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")

    # (1,0) Precision & Recall
    ax = axes[1, 0]
    ax.bar(x - width/2, precs, width, label="Macro Precision", color="#74c476", edgecolor="black", linewidth=0.8)
    ax.bar(x + width/2, recs, width, label="Macro Recall", color="#31a354", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(min(min(precs), min(recs)) - 0.08, 1.02)
    ax.set_title("Macro Precision vs. Macro Recall", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")

    # (1,1) Parameters vs Accuracy
    ax = axes[1, 1]
    for r in summary_records:
        ax.scatter(r["trainable_parameters"], r["test_accuracy"], color=r["color"], s=100, edgecolor="black", zorder=3)
        ax.annotate(r["short_name"], (r["trainable_parameters"], r["test_accuracy"]), xytext=(8, -3), textcoords="offset points", fontsize=8, fontweight="bold")
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title("Parameters vs. Test Accuracy", fontweight="bold")
    ax.set_xlabel("Trainable Parameters (log scale)")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)

    # (1,2) Inference Latency
    ax = axes[1, 2]
    b = ax.bar(labels, lats, yerr=lat_errs, color=colors, width=0.5, edgecolor="black", linewidth=0.8, capsize=4)
    for bar in b:
        h = bar.get_height()
        ax.annotate(f"{h:.2f} ms", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylim(0, max(lats) * 1.35)
    ax.set_title("Inference Latency per Image (ms)", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    fig.suptitle("MobileNetV2 Phase 3 Final Test Evaluation Dashboard", fontsize=15, fontweight="bold", y=0.98)
    p7 = output_dir / "evaluation_summary_overview.png"
    plt.savefig(p7, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p7}")


def export_metrics(results, statistical_res, predictions_dict, output_metrics_dir, classes):
    """
    Save all machine-readable metric tables, classification reports, and prediction records.
    """
    metrics_dir = Path(output_metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary CSV Table
    summary_rows = []
    for exp_key, data in results.items():
        metrics = data["metrics"]
        lat = data["latency"]
        summary_rows.append({
            "experiment": exp_key,
            "display_name": data["display_name"],
            "test_accuracy": metrics["test_accuracy"],
            "accuracy_ci_95_low": metrics["accuracy_ci_95"][0],
            "accuracy_ci_95_high": metrics["accuracy_ci_95"][1],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "test_loss": metrics["test_loss"],
            "correct_count": metrics["correct_count"],
            "incorrect_count": metrics["incorrect_count"],
            "total_count": metrics["total_count"],
            "trainable_parameters": data["trainable_parameters"],
            "total_parameters": data["total_parameters"],
            "unfrozen_blocks": data["unfrozen_blocks"],
            "inference_latency_ms": lat["latency_ms_per_image_mean"],
            "inference_latency_std_ms": lat["latency_ms_per_image_std"],
            "throughput_fps": lat["throughput_fps"],
            "val_accuracy": data["val_accuracy"]
        })

    summary_csv_path = metrics_dir / "final_test_results.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_csv_path}")

    # 2. Complete Summary JSON
    summary_json_path = metrics_dir / "final_test_results.json"
    full_json_payload = {
        "evaluation_dataset": {
            "name": "TensorFlow Flowers Test Partition",
            "total_images": results[list(results.keys())[0]]["metrics"]["total_count"],
            "classes": classes,
            "deterministic_transform": "ImageNet default validation transforms (256 bilinear resize, 224 center crop, normalization)"
        },
        "experiments": results,
        "statistical_comparison": statistical_res
    }
    with open(summary_json_path, "w") as f:
        json.dump(full_json_payload, f, indent=2)
    print(f"Saved: {summary_json_path}")

    # 3. Statistical Comparison JSON
    stat_json_path = metrics_dir / "statistical_comparison.json"
    with open(stat_json_path, "w") as f:
        json.dump(statistical_res, f, indent=2)
    print(f"Saved: {stat_json_path}")

    # 4. Individual Per-Class Classification Reports
    for exp_key, data in results.items():
        report_path = metrics_dir / f"{exp_key}_classification_report.csv"
        rep_rows = data["metrics"]["per_class_report"]
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1_score", "support"])
            writer.writeheader()
            writer.writerows(rep_rows)
            # Add summary rows
            writer.writerow({
                "class": "macro_avg",
                "precision": data["metrics"]["macro_precision"],
                "recall": data["metrics"]["macro_recall"],
                "f1_score": data["metrics"]["macro_f1"],
                "support": data["metrics"]["total_count"]
            })
            writer.writerow({
                "class": "weighted_avg",
                "precision": data["metrics"]["macro_precision"],
                "recall": data["metrics"]["macro_recall"],
                "f1_score": data["metrics"]["weighted_f1"],
                "support": data["metrics"]["total_count"]
            })
            writer.writerow({
                "class": "overall_accuracy",
                "precision": data["metrics"]["test_accuracy"],
                "recall": data["metrics"]["test_accuracy"],
                "f1_score": data["metrics"]["test_accuracy"],
                "support": data["metrics"]["total_count"]
            })
        print(f"Saved: {report_path}")

    # 5. Prediction-Level Records with Softmax Probabilities
    for exp_key, pred_data in predictions_dict.items():
        pred_path = metrics_dir / f"{exp_key}_predictions.csv"
        image_paths = pred_data["image_paths"]
        y_true = pred_data["targets"]
        y_pred = pred_data["preds"]
        y_probs = pred_data["probs"]

        prob_headers = [f"prob_{c}" for c in classes]
        fieldnames = ["image_path", "true_class", "predicted_class", "confidence", "is_correct"] + prob_headers

        with open(pred_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(len(image_paths)):
                t_cls = classes[y_true[idx]]
                p_cls = classes[y_pred[idx]]
                probs = y_probs[idx]
                conf = float(np.max(probs))
                is_corr = bool(t_cls == p_cls)

                row = {
                    "image_path": image_paths[idx],
                    "true_class": t_cls,
                    "predicted_class": p_cls,
                    "confidence": conf,
                    "is_correct": is_corr
                }
                for c_idx, c_name in enumerate(classes):
                    row[f"prob_{c_name}"] = float(probs[c_idx])
                writer.writerow(row)
        print(f"Saved: {pred_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Final Evaluation of MobileNetV2 Transfer Learning Models"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/processed/test",
        help="Path to untouched test dataset partition (default: data/processed/test)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker count (default: 2)"
    )
    parser.add_argument(
        "--latency-repetitions",
        type=int,
        default=5,
        help="Number of full evaluation passes for latency benchmarking (default: 5)"
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="results/metrics",
        help="Destination directory for evaluation metrics (default: results/metrics)"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="results/plots",
        help="Destination directory for evaluation plots (default: results/plots)"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute on dummy/validation data into temporary directory without modifying official test outputs"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("MobileNetV2 Transfer Learning: Phase 3 Final Evaluation Pipeline")
    print("=" * 70)
    print(f"Device:                 {device}")
    if torch.cuda.is_available():
        print(f"GPU Name:               {torch.cuda.get_device_name(0)}")
        print(f"GPU Total VRAM:         {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Evaluation Partition:   {args.test_dir}")
    print(f"Batch Size:             {args.batch_size}")
    print(f"Smoke Test Mode:        {args.smoke_test}")
    print("=" * 70)

    # Route paths for smoke-test mode to prevent overwriting official outputs
    if args.smoke_test:
        metrics_dir = Path("results/metrics/smoke_eval")
        plots_dir = Path("results/plots/smoke_eval")
    else:
        metrics_dir = Path(args.metrics_dir)
        plots_dir = Path(args.plots_dir)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Evaluation DataLoader
    eval_loader, classes, image_paths = get_eval_loader(
        data_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    num_classes = len(classes)
    total_images = len(image_paths)

    print(f"Dataset Loaded:         {total_images} images across {num_classes} classes: {classes}")

    results = {}
    predictions_dict = {}
    cms = {}
    cms_norm = {}
    summary_records = []

    # 2. Evaluate Each Model
    for exp_key, meta in EXPERIMENT_METADATA.items():
        print("\n" + "-" * 70)
        print(f"Evaluating {meta['display_name']} [{exp_key}]...")
        print(f"Loading checkpoint: {meta['checkpoint']}")

        model, ckpt, total_params = load_model(
            checkpoint_path=meta["checkpoint"],
            device=device,
            num_classes=num_classes
        )

        trainable_params = ckpt.get("trainable_parameters", 0)
        val_acc = float(ckpt.get("best_val_accuracy", 0.0))

        print(f"Architecture loaded:    {total_params:,} total parameters")
        print(f"Trainable parameters:   {trainable_params:,} ({meta['unfrozen_blocks']} feature blocks unfrozen)")
        print(f"Validation Baseline:    {val_acc * 100:.2f}%")

        # Forward pass on evaluation partition
        y_true, y_pred, y_probs, test_loss = run_inference(
            model=model,
            loader=eval_loader,
            device=device,
            classes=classes
        )

        # Compute metrics
        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            classes=classes,
            test_loss=test_loss
        )

        # Measure inference latency
        print("Benchmarking inference latency...")
        latency_info = measure_inference_latency(
            model=model,
            loader=eval_loader,
            device=device,
            repetitions=args.latency_repetitions
        )

        print(f"Test Accuracy:          {metrics['test_accuracy'] * 100:.2f}% (95% CI: [{metrics['accuracy_ci_95'][0]*100:.2f}%, {metrics['accuracy_ci_95'][1]*100:.2f}%])")
        print(f"Correct Predictions:    {metrics['correct_count']} / {metrics['total_count']}")
        print(f"Macro F1-Score:         {metrics['macro_f1']:.4f}")
        print(f"Weighted F1-Score:      {metrics['weighted_f1']:.4f}")
        print(f"Test Loss:              {metrics['test_loss']:.4f}")
        print(f"Inference Latency:      {latency_info['latency_ms_per_image_mean']:.2f} ± {latency_info['latency_ms_per_image_std']:.2f} ms/image ({latency_info['throughput_fps']:.1f} FPS)")

        results[exp_key] = {
            "display_name": meta["display_name"],
            "short_name": meta["short_name"],
            "checkpoint": meta["checkpoint"],
            "unfrozen_blocks": meta["unfrozen_blocks"],
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "val_accuracy": val_acc,
            "metrics": metrics,
            "latency": latency_info
        }

        predictions_dict[exp_key] = {
            "targets": y_true,
            "preds": y_pred,
            "probs": y_probs,
            "image_paths": image_paths
        }

        cms[exp_key] = metrics["confusion_matrix"]
        cms_norm[exp_key] = metrics["confusion_matrix_normalized"]

        summary_records.append({
            "key": exp_key,
            "display_name": meta["display_name"],
            "short_name": meta["short_name"],
            "color": meta["color"],
            "test_accuracy": metrics["test_accuracy"],
            "ci_low": metrics["accuracy_ci_95"][0],
            "ci_high": metrics["accuracy_ci_95"][1],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "val_accuracy": val_acc,
            "trainable_parameters": trainable_params,
            "latency_ms_per_image": latency_info["latency_ms_per_image_mean"],
            "latency_ms_per_image_std": latency_info["latency_ms_per_image_std"],
            "throughput_fps": latency_info["throughput_fps"]
        })

        # Save individual confusion matrix plots
        plot_confusion_matrices(
            cm=metrics["confusion_matrix"],
            cm_norm=metrics["confusion_matrix_normalized"],
            classes=classes,
            exp_name=exp_key,
            output_dir=plots_dir
        )

    # 3. Statistical Comparison (McNemar's Test)
    print("\n" + "=" * 70)
    print("Computing Statistical Comparisons (Paired McNemar's Tests)...")
    print("=" * 70)
    first_exp = list(predictions_dict.keys())[0]
    ground_truth = predictions_dict[first_exp]["targets"]
    statistical_results = compute_mcnemar_tests(predictions_dict, ground_truth)

    for pair_name, stat in statistical_results.items():
        print(f"\nComparison: {pair_name}")
        c = stat["contingency_matrix"]
        print(f"  Contingency Table:")
        print(f"    Both correct (n11): {c['both_correct_n11']}")
        print(f"    {stat['model_1']} only (n10): {c['model1_only_n10']}")
        print(f"    {stat['model_2']} only (n01): {c['model2_only_n01']}")
        print(f"    Both incorrect (n00): {c['both_incorrect_n00']}")
        print(f"  Discordant count:    {stat['total_discordant']}")
        print(f"  McNemar Chi2 stat:   {stat['mcnemar_chi2_stat']:.4f}")
        print(f"  Exact Binomial p:    {stat['p_value_exact_binomial']:.4f}")
        print(f"  Asymptotic Chi2 p:   {stat['p_value_asymptotic_cc']:.4f}")
        print(f"  Recommended p-value: {stat['recommended_p_value']:.4f} ({stat['recommended_method']})")
        print(f"  Significant (α=0.05): {stat['statistically_significant_alpha_0_05']}")
        print(f"  Conclusion:          {stat['interpretation']}")

    # 4. Generate Comparative Visualizations
    print("\n" + "=" * 70)
    print("Generating Comparative Visualizations...")
    print("=" * 70)
    plot_combined_confusion_matrices(
        cms=cms,
        cms_norm=cms_norm,
        classes=classes,
        exp_keys=list(EXPERIMENT_METADATA.keys()),
        output_dir=plots_dir
    )
    plot_comparative_charts(summary_records, plots_dir)

    # 5. Export Metrics and Predictions
    print("\n" + "=" * 70)
    print("Exporting Final Metrics and Predictions...")
    print("=" * 70)
    export_metrics(
        results=results,
        statistical_res=statistical_results,
        predictions_dict=predictions_dict,
        output_metrics_dir=metrics_dir,
        classes=classes
    )

    print("\n" + "=" * 70)
    print("PHASE 3 FINAL EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

