import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm


def set_seed(seed=42):
    """Seed Python, NumPy, PyTorch CPU and CUDA RNGs for experimental reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure reproducible cuDNN behavior where supported
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loaders(batch_size, num_workers=2):
    """
    Construct training and validation DataLoaders with identical transforms and batch size.
    Notice: The test split is never accessed during training, model development, or validation.
    """
    weights = MobileNet_V2_Weights.DEFAULT

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = weights.transforms()

    train_dataset = datasets.ImageFolder(
        "data/processed/train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        "data/processed/val",
        transform=val_transform
    )

    assert train_dataset.classes == val_dataset.classes, (
        f"Class mismatch between train ({train_dataset.classes}) "
        f"and val ({val_dataset.classes})"
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, train_dataset.classes


def create_model(num_classes, experiment, checkpoint_path=None):
    """
    Construct MobileNetV2 for the specified transfer-learning experiment.

    Experiment A ('frozen'):
      - Feature extractor completely frozen.
      - Classifier replaced with a new linear head (5 classes).
      - Initialized from ImageNet pretrained weights.

    Experiment B ('partial'):
      - Reconstruct MobileNetV2 with 5-class linear head.
      - Load complete trained model weights from Experiment A ('models/frozen_best.pth').
      - Freeze the entire feature extractor initially.
      - Unfreeze only the final 2 complete MobileNetV2 feature blocks (features[17], features[18]).
      - Keep classifier trainable.

    Experiment C ('deeper'):
      - Same as Experiment B, but starting independently from Experiment A checkpoint
        and unfreezing the final 5 complete MobileNetV2 feature blocks (features[14]..features[18]).
      - Keep classifier trainable.
    """
    if experiment == "frozen":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Freeze the pretrained feature extractor
        for param in model.features.parameters():
            param.requires_grad = False

        # Replace ImageNet 1000-class classifier with 5-class head
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        for param in model.classifier.parameters():
            param.requires_grad = True

        unfrozen_blocks = 0
        source_desc = "torchvision.models.MobileNet_V2_Weights.DEFAULT"
        return model, unfrozen_blocks, source_desc

    elif experiment in ["partial", "deeper"]:
        if checkpoint_path is None:
            checkpoint_path = "models/frozen_best.pth"

        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(
                f"Source checkpoint not found: '{checkpoint_path}'. "
                f"Experiment '{experiment}' requires the trained Experiment A checkpoint."
            )

        # 1. Reconstruct MobileNetV2 architecture with 5-class classifier
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

        # 2. Load complete trained model weights from Experiment A checkpoint
        checkpoint = torch.load(ckpt_file, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        # Verify classifier output dimension matches num_classes (5)
        classifier_out = state_dict["classifier.1.weight"].shape[0]
        if classifier_out != num_classes:
            raise ValueError(
                f"Classifier output dimension mismatch in '{checkpoint_path}': "
                f"expected {num_classes}, found {classifier_out}."
            )

        # Load weights into the reconstructed model
        model.load_state_dict(state_dict, strict=True)

        # 3. Freeze the entire feature extractor initially
        for param in model.features.parameters():
            param.requires_grad = False

        # 4. Unfreeze the specified final feature blocks
        blocks = 2 if experiment == "partial" else 5
        feature_children = list(model.features.children())
        for block in feature_children[-blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # 5. Keep the classifier trainable (never re-initialize it)
        for param in model.classifier.parameters():
            param.requires_grad = True

        return model, blocks, str(ckpt_file)
    else:
        raise ValueError(
            f"Unsupported experiment: '{experiment}'. Expected 'frozen', 'partial', or 'deeper'."
        )


def run_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    """
    Execute one full epoch (training or validation).

    Pretrained BatchNorm running statistics (running_mean and running_var) are kept
    fixed in eval mode during fine-tuning to prevent noisy updates on small batches.
    Loss is computed as a sample-weighted epoch average.
    Accuracy is computed across all evaluated examples.
    """
    training = optimizer is not None

    if training:
        model.train()
        # Keep BatchNorm running statistics fixed in pretrained layers
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch_idx, (images, labels) in enumerate(tqdm(loader, leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_count

    if total == 0:
        return 0.0, 0.0

    return total_loss / total, correct / total


def update_experiment_summary(summary_path="results/metrics/experiment_summary.json"):
    """
    Create or update the machine-readable experiment summary JSON file.
    Preserves Experiment A baseline from its existing checkpoint and history without alteration.
    Includes Experiment B and C when their respective artifacts exist.
    """
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "project": "A Comparative Study of Feature Extraction and Fine-Tuning Strategies in Transfer Learning Using MobileNetV2 for Image Classification",
        "dataset": {
            "name": "TensorFlow Flowers",
            "num_classes": 5,
            "classes": ["daisy", "dandelion", "roses", "sunflowers", "tulips"],
            "splits": {"train": 2567, "val": 547, "test": 556}
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "seed": 42
        },
        "experiments": {}
    }

    # Load baseline Experiment A (frozen) from existing artifacts without modifying them
    frozen_ckpt = Path("models/frozen_best.pth")
    frozen_hist = Path("results/metrics/frozen_history.csv")
    if frozen_ckpt.exists() and frozen_hist.exists():
        ckpt_a = torch.load(frozen_ckpt, map_location="cpu")
        with open(frozen_hist, "r") as f:
            rows_a = list(csv.DictReader(f))
        best_row_a = max(rows_a, key=lambda r: float(r["val_accuracy"]))
        final_row_a = rows_a[-1]

        summary["experiments"]["frozen"] = {
            "experiment_name": "Experiment A (Feature Extraction / Frozen Backbone)",
            "status": "completed",
            "unfrozen_blocks": 0,
            "trainable_parameters": 6405,
            "total_parameters": 2230277,
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": len(rows_a),
            "best_epoch": int(best_row_a["epoch"]),
            "best_val_accuracy": float(ckpt_a.get("best_val_accuracy", best_row_a["val_accuracy"])),
            "final_train_accuracy": float(final_row_a["train_accuracy"]),
            "final_train_loss": float(final_row_a["train_loss"]),
            "final_val_loss": float(final_row_a["val_loss"]),
            "training_seconds": float(ckpt_a.get("training_seconds", 0.0)),
            "checkpoint_file": str(frozen_ckpt),
            "history_file": str(frozen_hist),
            "source_checkpoint": "torchvision.models.MobileNet_V2_Weights.DEFAULT"
        }

    # Inspect Experiment B (partial)
    partial_ckpt = Path("models/partial_best.pth")
    partial_hist = Path("results/metrics/partial_history.csv")
    if partial_ckpt.exists() and partial_hist.exists():
        ckpt_b = torch.load(partial_ckpt, map_location="cpu")
        with open(partial_hist, "r") as f:
            rows_b = list(csv.DictReader(f))
        best_row_b = max(rows_b, key=lambda r: float(r["val_accuracy"]))
        final_row_b = rows_b[-1]
        summary["experiments"]["partial"] = {
            "experiment_name": "Experiment B (Partial Fine-Tuning / 2 Blocks Unfrozen)",
            "status": "completed",
            "unfrozen_blocks": ckpt_b.get("unfrozen_blocks", 2),
            "trainable_parameters": ckpt_b.get("trainable_parameters", 892485),
            "total_parameters": 2230277,
            "learning_rate": ckpt_b.get("learning_rate", 0.0001),
            "batch_size": ckpt_b.get("batch_size", 16),
            "epochs": len(rows_b),
            "best_epoch": int(ckpt_b.get("best_epoch", best_row_b["epoch"])),
            "best_val_accuracy": float(ckpt_b.get("best_val_accuracy", best_row_b["val_accuracy"])),
            "final_train_accuracy": float(final_row_b["train_accuracy"]),
            "final_train_loss": float(final_row_b["train_loss"]),
            "final_val_loss": float(final_row_b["val_loss"]),
            "training_seconds": float(ckpt_b.get("training_seconds", 0.0)),
            "checkpoint_file": str(partial_ckpt),
            "history_file": str(partial_hist),
            "source_checkpoint": ckpt_b.get("source_checkpoint", "models/frozen_best.pth")
        }
    else:
        summary["experiments"]["partial"] = {
            "experiment_name": "Experiment B (Partial Fine-Tuning / 2 Blocks Unfrozen)",
            "status": "pending",
            "unfrozen_blocks": 2,
            "trainable_parameters": 892485,
            "learning_rate": 0.0001,
            "batch_size": 16,
            "epochs": 10,
            "source_checkpoint": "models/frozen_best.pth"
        }

    # Inspect Experiment C (deeper)
    deeper_ckpt = Path("models/deeper_best.pth")
    deeper_hist = Path("results/metrics/deeper_history.csv")
    if deeper_ckpt.exists() and deeper_hist.exists():
        ckpt_c = torch.load(deeper_ckpt, map_location="cpu")
        with open(deeper_hist, "r") as f:
            rows_c = list(csv.DictReader(f))
        best_row_c = max(rows_c, key=lambda r: float(r["val_accuracy"]))
        final_row_c = rows_c[-1]
        summary["experiments"]["deeper"] = {
            "experiment_name": "Experiment C (Deeper Fine-Tuning / 5 Blocks Unfrozen)",
            "status": "completed",
            "unfrozen_blocks": ckpt_c.get("unfrozen_blocks", 5),
            "trainable_parameters": ckpt_c.get("trainable_parameters", 1687749),
            "total_parameters": 2230277,
            "learning_rate": ckpt_c.get("learning_rate", 0.0001),
            "batch_size": ckpt_c.get("batch_size", 16),
            "epochs": len(rows_c),
            "best_epoch": int(ckpt_c.get("best_epoch", best_row_c["epoch"])),
            "best_val_accuracy": float(ckpt_c.get("best_val_accuracy", best_row_c["val_accuracy"])),
            "final_train_accuracy": float(final_row_c["train_accuracy"]),
            "final_train_loss": float(final_row_c["train_loss"]),
            "final_val_loss": float(final_row_c["val_loss"]),
            "training_seconds": float(ckpt_c.get("training_seconds", 0.0)),
            "checkpoint_file": str(deeper_ckpt),
            "history_file": str(deeper_hist),
            "source_checkpoint": ckpt_c.get("source_checkpoint", "models/frozen_best.pth")
        }
    else:
        summary["experiments"]["deeper"] = {
            "experiment_name": "Experiment C (Deeper Fine-Tuning / 5 Blocks Unfrozen)",
            "status": "pending",
            "unfrozen_blocks": 5,
            "trainable_parameters": 1687749,
            "learning_rate": 0.0001,
            "batch_size": 16,
            "epochs": 10,
            "source_checkpoint": "models/frozen_best.pth"
        }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary updated: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Transfer Learning Experiments for MobileNetV2 on TensorFlow Flowers"
    )
    parser.add_argument(
        "--experiment",
        choices=["frozen", "partial", "deeper"],
        default="frozen",
        help="Transfer learning strategy: frozen (Exp A), partial (Exp B), or deeper (Exp C)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16; use 8 if VRAM is constrained)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 0.001 for frozen, 0.0001 for partial/deeper)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes (default: 2)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to initial baseline checkpoint for fine-tuning (default: models/frozen_best.pth)"
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=None,
        help="Custom output checkpoint path (default: models/{experiment}_best.pth)"
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="Custom output metrics CSV path (default: results/metrics/{experiment}_history.csv)"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a brief 1-epoch dry run on 2 batches without modifying official outputs"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("MobileNetV2 Transfer Learning Experiment Runner")
    print("=" * 60)
    print(f"Device:           {device}")
    if torch.cuda.is_available():
        print(f"GPU Name:         {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory:       {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Experiment:       {args.experiment}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Random Seed:      {args.seed}")
    print(f"Smoke Test Mode:  {args.smoke_test}")

    train_loader, val_loader, classes = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    model, unfrozen_blocks, source_ckpt_desc = create_model(
        num_classes=len(classes),
        experiment=args.experiment,
        checkpoint_path=args.checkpoint
    )
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Classes:          {classes}")
    print(f"Unfrozen Blocks:  {unfrozen_blocks}")
    print(f"Trainable Params: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
    print(f"Total Params:     {total_params:,}")
    print(f"Source Weights:   {source_ckpt_desc}")

    # Set learning rate according to experimental protocol
    if args.lr is not None:
        lr = args.lr
    else:
        lr = 0.001 if args.experiment == "frozen" else 0.0001
    print(f"Learning Rate:    {lr}")
    print("=" * 60)

    # Optimizer receives only parameters that require gradients
    trainable_param_groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_param_groups, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Determine destination paths
    if args.smoke_test:
        model_path = args.output_model or f"models/smoke_test_{args.experiment}.pth"
        metrics_path = args.output_metrics or f"results/metrics/smoke_test_{args.experiment}.csv"
        epochs = 1
        max_batches = 2
    else:
        model_path = args.output_model or f"models/{args.experiment}_best.pth"
        metrics_path = args.output_metrics or f"results/metrics/{args.experiment}_history.csv"
        epochs = args.epochs
        max_batches = None

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    best_epoch = -1
    best_weights = None
    history = []
    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_batches
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            max_batches=max_batches
        )

        epoch_duration = time.time() - epoch_start

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "epoch_duration_seconds": epoch_duration
        })

        print(
            f"Epoch {epoch + 1:2d}/{epochs:2d} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_duration:.2f}s"
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())

    total_duration = time.time() - total_start

    # Save comprehensive checkpoint dictionary
    checkpoint_payload = {
        "model_state_dict": best_weights,
        "experiment": args.experiment,
        "classes": classes,
        "best_val_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "training_seconds": total_duration,
        "learning_rate": lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "unfrozen_blocks": unfrozen_blocks,
        "trainable_parameters": trainable_params,
        "source_checkpoint": source_ckpt_desc
    }
    torch.save(checkpoint_payload, model_path)

    # Save training history CSV
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print("\n" + "=" * 60)
    print("Training run finished successfully!")
    print(f"Best Validation Accuracy: {best_accuracy:.4f} (at Epoch {best_epoch})")
    print(f"Total Duration:          {total_duration:.2f} seconds")
    print(f"Checkpoint Saved:        {model_path}")
    print(f"History Saved:           {metrics_path}")
    print("=" * 60)

    # Update summary JSON only when running official experiments
    if not args.smoke_test:
        update_experiment_summary()


if __name__ == "__main__":
    main()
