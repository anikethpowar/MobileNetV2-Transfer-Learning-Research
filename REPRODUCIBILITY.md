
# Reproducibility Guide

## 1. System Requirements

Original experimental environment:

- Linux Mint
- Python 3.12.3
- NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM
- 16 GB RAM
- PyTorch 2.11.0+cu128
- TorchVision 0.26.0+cu128

A CUDA-capable GPU is recommended, but smaller-scale execution may also be possible on CPU.

## 2. Create a Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install Dependencies

Install compatible PyTorch and TorchVision builds using:

https://pytorch.org/get-started/locally/

For the original environment, the CUDA 12.8 PyTorch package index was used.

Then install the scientific Python dependencies:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn pillow tqdm
```

Verify GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 4. Download Dataset

```bash
mkdir -p data/raw

wget -c \
  https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz \
  -O data/raw/flower_photos.tgz

tar -xzf data/raw/flower_photos.tgz -C data/raw/
```

## 5. Prepare Dataset

```bash
python src/prepare_dataset.py
```

Expected split:

| Split | Images |
|---|---:|
| Train | 2567 |
| Validation | 547 |
| Test | 556 |

The script uses random seed 42.

## 6. Train Models

Run the experiments sequentially.

### Frozen Baseline

```bash
python src/train.py --experiment frozen --epochs 10 --batch-size 16
```

### Partial Fine-Tuning

```bash
python src/train.py --experiment partial --epochs 10 --batch-size 16
```

### Deeper Fine-Tuning

```bash
python src/train.py --experiment deeper --epochs 10 --batch-size 16
```

Both fine-tuning experiments independently load the best frozen baseline checkpoint.

## 7. Final Evaluation

```bash
python src/evaluate.py --batch-size 16
```

The evaluation produces:

- Overall and per-class classification metrics
- Test loss
- Confusion matrices
- Confidence intervals
- Paired statistical comparisons
- GPU inference benchmarks
- Prediction-level CSV files

## 8. Experimental Configuration

| Parameter | Value |
|---|---|
| Backbone | MobileNetV2 |
| Pretraining | ImageNet |
| Input size | 224 x 224 |
| Number of classes | 5 |
| Optimizer | Adam |
| Batch size | 16 |
| Epochs | 10 |
| Random seed | 42 |
| Baseline learning rate | 0.001 |
| Fine-tuning learning rate | 0.0001 |
| Checkpoint selection | Best validation accuracy |

## 9. Reproducibility Limitations

Exact numerical reproduction may vary because of CUDA kernels, library versions, hardware differences, and other sources of nondeterminism.

The original experiments used one random seed.

Training times and inference measurements are hardware-dependent.

The original model checkpoints and image dataset are not included in the repository.

## 10. Research Artifacts

The repository includes:

- Training histories
- Evaluation metrics
- Statistical comparison results
- Prediction-level records
- Confusion matrices
- Comparative figures

These artifacts support independent inspection of the reported findings.
