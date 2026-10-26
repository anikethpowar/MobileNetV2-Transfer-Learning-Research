import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.train import create_model, get_loaders, run_epoch, set_seed


def test_1_checkpoint_loading():
    print("Testing Step 1: Experiment A checkpoint loading...")
    ckpt_path = Path("models/frozen_best.pth")
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in ckpt, "Checkpoint missing 'model_state_dict'"
    assert "classes" in ckpt, "Checkpoint missing 'classes'"
    print(f"  -> Checkpoint loaded successfully with keys: {list(ckpt.keys())}")


def test_2_output_classes():
    print("Testing Step 2: Number of output classes...")
    ckpt = torch.load("models/frozen_best.pth", map_location="cpu")
    expected_classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    assert ckpt["classes"] == expected_classes, (
        f"Classes mismatch: expected {expected_classes}, found {ckpt['classes']}"
    )
    weight_shape = ckpt["model_state_dict"]["classifier.1.weight"].shape
    assert weight_shape == (5, 1280), f"Classifier weight shape mismatch: {weight_shape}"
    bias_shape = ckpt["model_state_dict"]["classifier.1.bias"].shape
    assert bias_shape == (5,), f"Classifier bias shape mismatch: {bias_shape}"
    print(f"  -> Verified 5 classes: {expected_classes}")
    print(f"  -> Verified classifier dimension: {weight_shape[0]}")


def test_3_trainable_parameters():
    print("Testing Step 3: Trainable parameter counts across experiments...")
    expected_counts = {
        "frozen": 6405,
        "partial": 892485,
        "deeper": 1687749
    }
    for exp, expected in expected_counts.items():
        model, unfrozen_blocks, _ = create_model(5, exp)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable == expected, (
            f"Parameter count mismatch for {exp}: expected {expected}, got {trainable}"
        )
        print(f"  -> {exp.upper():7s}: {trainable:,} trainable / {total:,} total params ({trainable/total*100:.2f}%)")


def test_4_layer_unfreezing_specificity():
    print("Testing Step 4: Verification that only intended blocks are trainable...")

    # Partial: only features[17], features[18], and classifier are trainable
    model_b, blocks_b, _ = create_model(5, "partial")
    feature_children_b = list(model_b.features.children())
    for idx in range(17):
        for name, param in feature_children_b[idx].named_parameters():
            assert not param.requires_grad, f"Exp B: features[{idx}].{name} should be frozen!"
    for idx in [17, 18]:
        for name, param in feature_children_b[idx].named_parameters():
            assert param.requires_grad, f"Exp B: features[{idx}].{name} should be trainable!"
    for name, param in model_b.classifier.named_parameters():
        assert param.requires_grad, f"Exp B: classifier.{name} should be trainable!"
    print("  -> Exp B verified: features[0..16] frozen, features[17, 18] unfrozen, classifier trainable.")

    # Deeper: only features[14..18] and classifier are trainable
    model_c, blocks_c, _ = create_model(5, "deeper")
    feature_children_c = list(model_c.features.children())
    for idx in range(14):
        for name, param in feature_children_c[idx].named_parameters():
            assert not param.requires_grad, f"Exp C: features[{idx}].{name} should be frozen!"
    for idx in range(14, 19):
        for name, param in feature_children_c[idx].named_parameters():
            assert param.requires_grad, f"Exp C: features[{idx}].{name} should be trainable!"
    for name, param in model_c.classifier.named_parameters():
        assert param.requires_grad, f"Exp C: classifier.{name} should be trainable!"
    print("  -> Exp C verified: features[0..13] frozen, features[14..18] unfrozen, classifier trainable.")


def test_5_and_6_forward_and_finite_loss():
    print("Testing Steps 5 & 6: Forward pass and finite loss on small batch...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    for exp in ["partial", "deeper"]:
        model, _, _ = create_model(5, exp)
        model = model.to(device)
        model.eval()

        dummy_images = torch.randn(4, 3, 224, 224, device=device)
        dummy_labels = torch.tensor([0, 2, 4, 1], device=device)

        with torch.no_grad():
            outputs = model(dummy_images)
            assert outputs.shape == (4, 5), f"Output shape mismatch: {outputs.shape}"
            loss = criterion(outputs, dummy_labels)
            assert torch.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
            print(f"  -> {exp.upper()} forward pass successful: output {outputs.shape}, loss={loss.item():.4f} (finite).")


def test_7_backward_propagation():
    print("Testing Step 7: Backward propagation and gradient flow...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    for exp in ["partial", "deeper"]:
        model, _, _ = create_model(5, exp)
        model = model.to(device)
        model.train()

        dummy_images = torch.randn(4, 3, 224, 224, device=device)
        dummy_labels = torch.tensor([1, 3, 2, 0], device=device)

        outputs = model(dummy_images)
        loss = criterion(outputs, dummy_labels)
        loss.backward()

        # Check that unfrozen parameters have non-zero gradients and frozen have None
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Trainable param {name} has no grad!"
                assert torch.isfinite(p.grad).all(), f"Gradient for {name} has non-finite values!"
            else:
                assert p.grad is None, f"Frozen param {name} unexpectedly has grad!"
        print(f"  -> {exp.upper()} backward pass verified: gradients correctly populated only on trainable parameters.")


def test_8_file_writing_safety():
    print("Testing Step 8: Safe checkpoint and metric saving to temporary files...")
    temp_ckpt_path = Path("models/temp_smoke_test.pth")
    temp_csv_path = Path("results/metrics/temp_smoke_test.csv")

    try:
        dummy_payload = {
            "model_state_dict": {"test": torch.zeros(1)},
            "experiment": "smoke_test",
            "classes": ["a", "b"],
            "best_val_accuracy": 0.99
        }
        torch.save(dummy_payload, temp_ckpt_path)
        assert temp_ckpt_path.exists(), "Failed to write temp checkpoint"

        with open(temp_csv_path, "w") as f:
            f.write("epoch,loss\n1,0.5\n")
        assert temp_csv_path.exists(), "Failed to write temp CSV"

        print("  -> Verified temporary file write permissions.")
    finally:
        if temp_ckpt_path.exists():
            temp_ckpt_path.unlink()
        if temp_csv_path.exists():
            temp_csv_path.unlink()
        print("  -> Temporary test files cleanly removed.")


def test_9_no_test_split_access():
    print("Testing Step 9: Confirm no code path loads the test split...")
    train_loader, val_loader, classes = get_loaders(batch_size=8)
    train_dir = Path(train_loader.dataset.root).resolve()
    val_dir = Path(val_loader.dataset.root).resolve()

    test_dir = Path("data/processed/test").resolve()
    assert train_dir != test_dir, "Train loader is pointing to test split!"
    assert val_dir != test_dir, "Val loader is pointing to test split!"
    assert "test" not in str(train_dir), f"Suspicious train directory: {train_dir}"
    assert "test" not in str(val_dir), f"Suspicious val directory: {val_dir}"

    print(f"  -> Train data path: {train_dir}")
    print(f"  -> Val data path:   {val_dir}")
    print("  -> Confirmed test split is strictly isolated and never accessed.")


def run_all_tests():
    print("=" * 70)
    print("PHASE 2 PRE-FLIGHT VERIFICATION & SMOKE TEST SUITE")
    print("=" * 70)
    set_seed(42)
    test_1_checkpoint_loading()
    test_2_output_classes()
    test_3_trainable_parameters()
    test_4_layer_unfreezing_specificity()
    test_5_and_6_forward_and_finite_loss()
    test_7_backward_propagation()
    test_8_file_writing_safety()
    test_9_no_test_split_access()
    print("=" * 70)
    print("ALL 9 PRE-FLIGHT VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

