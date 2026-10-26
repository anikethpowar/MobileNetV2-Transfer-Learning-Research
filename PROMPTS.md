
# AI-Assisted Development and Prompt Log

## Overview

This project was developed with assistance from ChatGPT and Google Antigravity.

This document summarizes the development instructions, AI contributions, and verification procedures.

It is a reconstructed development log, not a verbatim transcript of every prompt.

## Phase 1: Environment Setup

### Objective

Configure a local deep learning environment capable of running pretrained MobileNetV2 on an NVIDIA RTX 3050 Laptop GPU.

### Requirements

- Python 3.12 virtual environment
- CUDA-enabled PyTorch
- TorchVision
- ImageNet-pretrained MobileNetV2
- GPU availability verification

### Outcome

Successfully loaded MobileNetV2 pretrained weights and verified CUDA inference.

## Phase 2: Dataset Preparation

### Objective

Prepare the TensorFlow Flowers dataset for image classification.

### Requirements

- Five flower categories
- Reproducible random seed
- Separate training, validation, and test partitions
- Preserve class labels
- Avoid test-data leakage

### Outcome

Created 2,567 training, 547 validation, and 556 test images.

## Phase 3: Frozen Feature Extraction

### Objective

Establish a baseline using a frozen MobileNetV2 feature extractor.

### Requirements

- Freeze pretrained feature layers
- Replace classifier with five-class output
- Adam optimizer
- 10 training epochs
- Save best validation checkpoint
- Record training metrics

### Outcome

Best validation accuracy: 91.59%.

## Phase 4: Controlled Fine-Tuning

### Objective

Investigate the effect of fine-tuning depth.

### Core Development Instructions

Implement two experiments initialized independently from the frozen baseline checkpoint.

Experiment B:
- Unfreeze final 2 feature blocks
- Keep classifier trainable
- Learning rate 0.0001

Experiment C:
- Unfreeze final 5 feature blocks
- Keep classifier trainable
- Learning rate 0.0001

Maintain identical dataset splits, batch size, optimizer, random seed, and validation procedure.

Preserve pretrained BatchNorm running statistics.

Do not overwrite existing experiment results.

### Outcome

Partial fine-tuning validation accuracy: 94.70%.

Deeper fine-tuning validation accuracy: 93.42%.

## Phase 5: Final Evaluation

### Objective

Evaluate all three checkpoints on the untouched test dataset.

### Core Development Instructions

Implement:

- Deterministic test preprocessing
- Checkpoint integrity checks
- Accuracy, precision, recall, and F1-score
- Per-class classification reports
- Confusion matrices
- Wilson confidence intervals
- Paired McNemar tests
- GPU inference benchmarking
- Publication-quality plots
- CSV and JSON result exports

Do not retrain models or tune parameters using the test set.

Verify the evaluation pipeline on synthetic or validation data before final testing.

### Outcome

Final test accuracies:

- Frozen: 91.55%
- Partial: 93.88%
- Deeper: 94.24%

## Phase 6: Research Documentation

### Objective

Prepare an IEEE-format undergraduate research paper and a reproducible GitHub repository.

### Requirements

- Accurate reporting of actual measurements
- Proper distinction between validation and test results
- Academic citations from verifiable sources
- Discussion of experimental limitations
- No fabricated findings or references

## Tools and Responsibilities

### ChatGPT

Assisted with:

- Research topic selection
- Transfer learning explanation
- Experimental methodology
- Implementation requirements
- Result interpretation
- GitHub documentation
- IEEE paper planning

### Google Antigravity

Assisted with:

- Fine-tuning implementation
- Training pipeline improvements
- Checkpoint management
- Evaluation pipeline implementation
- Smoke testing
- Statistical analysis implementation
- Plot generation

## Research Integrity

AI-generated code and recommendations were subject to local execution and verification.

Experimental results were produced using PyTorch on the documented hardware.

The project does not claim a novel neural network architecture.

The research contribution is an empirical comparison of fine-tuning strategies under a documented experimental setup.
