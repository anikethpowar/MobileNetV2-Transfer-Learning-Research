import argparse
import csv
import os
from pathlib import Path

# Set Matplotlib cache dir before importing matplotlib to avoid permission warnings
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# Experiment metadata and styling configuration
EXPERIMENT_CONFIGS = {
    "frozen": {
        "label": "Exp A: Frozen Backbone (0 blocks, lr=1e-3)",
        "short_label": "Exp A: Frozen",
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
        "history_file": Path("results/metrics/frozen_history.csv"),
        "model_file": Path("models/frozen_best.pth")
    },
    "partial": {
        "label": "Exp B: Partial Fine-Tuning (2 blocks, lr=1e-4)",
        "short_label": "Exp B: Partial (2 blk)",
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "--",
        "history_file": Path("results/metrics/partial_history.csv"),
        "model_file": Path("models/partial_best.pth")
    },
    "deeper": {
        "label": "Exp C: Deeper Fine-Tuning (5 blocks, lr=1e-4)",
        "short_label": "Exp C: Deeper (5 blk)",
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-.",
        "history_file": Path("results/metrics/deeper_history.csv"),
        "model_file": Path("models/deeper_best.pth")
    }
}


def load_experiment_data():
    """
    Read available history CSVs and model checkpoints without fabricating missing data.
    """
    data = {}
    for exp_id, cfg in EXPERIMENT_CONFIGS.items():
        hist_path = cfg["history_file"]
        model_path = cfg["model_file"]

        history_rows = []
        if hist_path.exists():
            with open(hist_path, "r") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    history_rows.append({
                        "epoch": int(r["epoch"]),
                        "train_loss": float(r["train_loss"]),
                        "train_accuracy": float(r["train_accuracy"]),
                        "val_loss": float(r["val_loss"]),
                        "val_accuracy": float(r["val_accuracy"])
                    })

        best_val_acc = None
        duration = None
        if model_path.exists():
            ckpt = torch.load(model_path, map_location="cpu")
            best_val_acc = ckpt.get("best_val_accuracy")
            duration = ckpt.get("training_seconds")

        if history_rows or best_val_acc is not None:
            data[exp_id] = {
                "config": cfg,
                "history": history_rows,
                "best_val_acc": best_val_acc,
                "duration": duration
            }

    return data


def plot_metric_vs_epoch(data, metric_key, ylabel, title, output_path):
    """Plot a single metric (loss or accuracy) across epochs for all available experiments."""
    plt.figure(figsize=(8, 5), dpi=300)
    has_data = False

    for exp_id, exp_data in data.items():
        history = exp_data["history"]
        if not history:
            continue
        has_data = True
        cfg = exp_data["config"]
        epochs = [r["epoch"] for r in history]
        values = [r[metric_key] for r in history]

        plt.plot(
            epochs,
            values,
            label=cfg["label"],
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=cfg["linestyle"],
            linewidth=2.0,
            markersize=6
        )

    if not has_data:
        plt.close()
        return

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, fontweight="medium")
    plt.ylabel(ylabel, fontsize=11, fontweight="medium")
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, loc="best", framealpha=0.95)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar_chart(categories, values, colors, ylabel, title, output_path, is_percentage=False):
    """Generate a clean, publication-ready bar chart."""
    if not categories or not values:
        return

    plt.figure(figsize=(7, 5), dpi=300)
    bars = plt.bar(categories, values, color=colors, width=0.55, edgecolor="black", linewidth=0.8, alpha=0.9)

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.ylabel(ylabel, fontsize=11, fontweight="medium")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        text = f"{h * 100:.2f}%" if is_percentage else f"{h:.1f}s"
        plt.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    if is_percentage:
        ymin = max(0.0, min(values) - 0.15) if values else 0.0
        plt.ylim(ymin, 1.05)
        # Format y ticks as percentages
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def generate_overview_figure(data, output_path):
    """Generate a 2x3 combined overview figure comparing all curves and metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # 1. Training Accuracy
    ax = axes[0, 0]
    for exp_id, exp_data in data.items():
        if exp_data["history"]:
            cfg = exp_data["config"]
            ax.plot([r["epoch"] for r in exp_data["history"]],
                    [r["train_accuracy"] for r in exp_data["history"]],
                    label=cfg["short_label"], color=cfg["color"], marker=cfg["marker"], linestyle=cfg["linestyle"])
    ax.set_title("Training Accuracy vs Epoch", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(range(1, 11))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8)

    # 2. Validation Accuracy
    ax = axes[0, 1]
    for exp_id, exp_data in data.items():
        if exp_data["history"]:
            cfg = exp_data["config"]
            ax.plot([r["epoch"] for r in exp_data["history"]],
                    [r["val_accuracy"] for r in exp_data["history"]],
                    label=cfg["short_label"], color=cfg["color"], marker=cfg["marker"], linestyle=cfg["linestyle"])
    ax.set_title("Validation Accuracy vs Epoch", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(range(1, 11))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8)

    # 3. Best Validation Accuracy Bar Chart
    ax = axes[0, 2]
    cats, vals, colors = [], [], []
    for exp_id in ["frozen", "partial", "deeper"]:
        if exp_id in data and data[exp_id]["best_val_acc"] is not None:
            cfg = data[exp_id]["config"]
            cats.append(cfg["short_label"])
            vals.append(data[exp_id]["best_val_acc"])
            colors.append(cfg["color"])
    if cats:
        bars = ax.bar(cats, vals, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h*100:.2f}%", (bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ymin = max(0.0, min(vals) - 0.15) if vals else 0.0
        ax.set_ylim(ymin, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title("Best Validation Accuracy", fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # 4. Training Loss
    ax = axes[1, 0]
    for exp_id, exp_data in data.items():
        if exp_data["history"]:
            cfg = exp_data["config"]
            ax.plot([r["epoch"] for r in exp_data["history"]],
                    [r["train_loss"] for r in exp_data["history"]],
                    label=cfg["short_label"], color=cfg["color"], marker=cfg["marker"], linestyle=cfg["linestyle"])
    ax.set_title("Training Loss vs Epoch", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_xticks(range(1, 11))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8)

    # 5. Validation Loss
    ax = axes[1, 1]
    for exp_id, exp_data in data.items():
        if exp_data["history"]:
            cfg = exp_data["config"]
            ax.plot([r["epoch"] for r in exp_data["history"]],
                    [r["val_loss"] for r in exp_data["history"]],
                    label=cfg["short_label"], color=cfg["color"], marker=cfg["marker"], linestyle=cfg["linestyle"])
    ax.set_title("Validation Loss vs Epoch", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_xticks(range(1, 11))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8)

    # 6. Training Duration Bar Chart
    ax = axes[1, 2]
    cats_dur, vals_dur, colors_dur = [], [], []
    for exp_id in ["frozen", "partial", "deeper"]:
        if exp_id in data and data[exp_id]["duration"] is not None:
            cfg = data[exp_id]["config"]
            cats_dur.append(cfg["short_label"])
            vals_dur.append(data[exp_id]["duration"])
            colors_dur.append(cfg["color"])
    if cats_dur:
        bars = ax.bar(cats_dur, vals_dur, color=colors_dur, width=0.5, edgecolor="black", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}s", (bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_title("Total Training Duration", fontweight="bold")
    ax.set_ylabel("Duration (seconds)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    fig.suptitle(
        "MobileNetV2 Transfer Learning: Feature Extraction vs Controlled Fine-Tuning",
        fontsize=15, fontweight="bold", y=0.98
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for MobileNetV2 transfer learning experiments.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Directory to save figures (default: results/plots)"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_experiment_data()

    if not data:
        print("No experimental data found in results/metrics/ or models/.")
        return

    print(f"Generating publication plots for experiments: {list(data.keys())}...")

    # 1. Training accuracy vs epoch
    plot_metric_vs_epoch(
        data,
        metric_key="train_accuracy",
        ylabel="Training Accuracy",
        title="Training Accuracy vs. Epoch Across Transfer-Learning Strategies",
        output_path=out_dir / "training_accuracy.png"
    )

    # 2. Validation accuracy vs epoch
    plot_metric_vs_epoch(
        data,
        metric_key="val_accuracy",
        ylabel="Validation Accuracy",
        title="Validation Accuracy vs. Epoch Across Transfer-Learning Strategies",
        output_path=out_dir / "validation_accuracy.png"
    )

    # 3. Training loss vs epoch
    plot_metric_vs_epoch(
        data,
        metric_key="train_loss",
        ylabel="Training Loss (Cross Entropy)",
        title="Training Loss vs. Epoch Across Transfer-Learning Strategies",
        output_path=out_dir / "training_loss.png"
    )

    # 4. Validation loss vs epoch
    plot_metric_vs_epoch(
        data,
        metric_key="val_loss",
        ylabel="Validation Loss (Cross Entropy)",
        title="Validation Loss vs. Epoch Across Transfer-Learning Strategies",
        output_path=out_dir / "validation_loss.png"
    )

    # 5. Bar chart: best validation accuracy
    cats, vals, colors = [], [], []
    for exp_id in ["frozen", "partial", "deeper"]:
        if exp_id in data and data[exp_id]["best_val_acc"] is not None:
            cfg = data[exp_id]["config"]
            cats.append(cfg["short_label"])
            vals.append(data[exp_id]["best_val_acc"])
            colors.append(cfg["color"])

    plot_bar_chart(
        categories=cats,
        values=vals,
        colors=colors,
        ylabel="Validation Accuracy",
        title="Best Validation Accuracy Comparison",
        output_path=out_dir / "best_validation_accuracy_comparison.png",
        is_percentage=True
    )

    # 6. Bar chart: training duration
    cats_dur, vals_dur, colors_dur = [], [], []
    for exp_id in ["frozen", "partial", "deeper"]:
        if exp_id in data and data[exp_id]["duration"] is not None:
            cfg = data[exp_id]["config"]
            cats_dur.append(cfg["short_label"])
            vals_dur.append(data[exp_id]["duration"])
            colors_dur.append(cfg["color"])

    plot_bar_chart(
        categories=cats_dur,
        values=vals_dur,
        colors=colors_dur,
        ylabel="Training Duration (seconds)",
        title="Computational Cost: Total Training Duration Comparison",
        output_path=out_dir / "training_duration_comparison.png",
        is_percentage=False
    )

    # 7. Comprehensive overview
    generate_overview_figure(data, out_dir / "experiment_comparison_overview.png")

    print("All plots generated successfully.")


if __name__ == "__main__":
    main()

