# Experimental Methodology: Controlled Transfer Learning with MobileNetV2

**Project Title:** A Comparative Study of Feature Extraction and Fine-Tuning Strategies in Transfer Learning Using MobileNetV2 for Image Classification  
**Target Architecture:** MobileNetV2 (torchvision implementation)  
**Dataset:** TensorFlow Flowers (5 Classes)  
**Hardware Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), 16 GB System RAM, Linux Mint  
**Software Stack:** Python 3.12.3, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128  

---

## 1. Overview and Research Objective

The goal of this research project is to evaluate and compare the empirical classification performance, convergence behavior, and computational overhead of distinct transfer-learning strategies using a convolutional neural network architecture (MobileNetV2). 

Specifically, this study contrasts:
1. **Feature Extraction (Frozen Backbone Baseline):** Only a newly initialized classification head is trained, while all feature-extraction layers remain fixed to ImageNet weights.
2. **Partial Fine-Tuning:** The classifier and the final two complete MobileNetV2 feature blocks are trained simultaneously, starting from the weights learned during the feature extraction phase.
3. **Deeper Fine-Tuning:** The classifier and the final five complete MobileNetV2 feature blocks are trained simultaneously, also starting from the weights learned during the feature extraction phase.

By controlling data splits, data augmentations, optimizer choices, and initialization origins, the independent variable across fine-tuning experiments is strictly isolated: the depth of feature representation adaptation (number of unfrozen residual blocks).

---

## 2. Dataset and Splits

The dataset employed is the **TensorFlow Flowers** dataset, comprising 3,670 color images categorized across 5 distinct botanical classes:
- **Daisy**
- **Dandelion**
- **Roses**
- **Sunflowers**
- **Tulips**

The dataset has been partitioned into fixed, reproducible splits using a fixed random seed (`seed = 42`):
- **Training set:** 2,567 images (~70.0%)
- **Validation set:** 547 images (~14.9%)
- **Test set:** 556 images (~15.1%)

### Experimental Partition Isolation
In adherence to standard empirical research principles, the **test partition is held out entirely**. It is never referenced or loaded during model training, hyperparameter configuration, or validation checkpoint selection. Model selection is performed exclusively using the validation partition.

---

## 3. Pretrained Architecture and Block Structure

The base model is torchvision's MobileNetV2 (`torchvision.models.mobilenet_v2`), pretrained on ImageNet-1K (`MobileNet_V2_Weights.DEFAULT`).

MobileNetV2 consists of:
- An initial standard convolutional block (`features[0]`: `Conv2dNormActivation`, 3 $\rightarrow$ 32 channels).
- 17 Inverted Residual blocks (`features[1]` through `features[17]`).
- A final pointwise $1 \times 1$ convolutional expansion block (`features[18]`: `Conv2dNormActivation`, 320 $\rightarrow$ 1280 channels).
- A classification head (`classifier`): `Sequential(Dropout(p=0.2), Linear(in_features=1280, out_features=5))`.

Total model parameters: **2,230,277**.

---

## 4. Experimental Configurations

### Experiment A: Baseline Feature Extraction (Completed)
- **Initialization:** ImageNet pretrained weights (`MobileNet_V2_Weights.DEFAULT`).
- **Feature Extractor:** Fully frozen (`requires_grad = False` across all 19 feature blocks).
- **Classification Head:** Linear layer ($1280 \rightarrow 5$), randomly initialized and trainable.
- **Trainable Parameters:** 6,405 (~0.29% of network parameters).
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$).
- **Learning Rate:** $1 \times 10^{-3}$ ($0.001$).
- **Batch Size:** 16.
- **Epochs:** 10.
- **Observed Result:** Best validation accuracy of 91.59% (epoch 9), final training accuracy of 95.40%, total training duration of 78.2 seconds.
- **Artifacts:** `models/frozen_best.pth`, `results/metrics/frozen_history.csv`.

### Experiment B: Partial Fine-Tuning
- **Initialization:** Full model weights (both features and classifier) loaded from `models/frozen_best.pth`.
- **Feature Extractor:**
  - Blocks `features[0]` through `features[16]` are frozen (`requires_grad = False`).
  - Final 2 blocks (`features[17]` Inverted Residual + `features[18]` Conv2dNormActivation) are unfrozen (`requires_grad = True`).
- **Classification Head:** Kept trainable (`requires_grad = True`), preserving weights learned in Experiment A.
- **Trainable Parameters:** 892,485 (~40.02% of network parameters).
- **Optimizer:** Adam.
- **Learning Rate:** $1 \times 10^{-4}$ ($0.0001$).
- **Batch Size:** 16 (default).
- **Epochs:** 10.
- **Artifacts:** `models/partial_best.pth`, `results/metrics/partial_history.csv`.

### Experiment C: Deeper Fine-Tuning
- **Initialization:** Full model weights loaded independently from `models/frozen_best.pth` (ensuring no cumulative drift from Experiment B).
- **Feature Extractor:**
  - Blocks `features[0]` through `features[13]` are frozen (`requires_grad = False`).
  - Final 5 blocks (`features[14]` through `features[18]`) are unfrozen (`requires_grad = True`).
- **Classification Head:** Kept trainable (`requires_grad = True`), preserving weights learned in Experiment A.
- **Trainable Parameters:** 1,687,749 (~75.67% of network parameters).
- **Optimizer:** Adam.
- **Learning Rate:** $1 \times 10^{-4}$ ($0.0001$).
- **Batch Size:** 16 (default).
- **Epochs:** 10.
- **Artifacts:** `models/deeper_best.pth`, `results/metrics/deeper_history.csv`.

---

## 5. Controlled Experimental Variables

To ensure valid comparative analysis between fine-tuning strategies:

| Experimental Parameter | Experiment A (Baseline) | Experiment B (Partial) | Experiment C (Deeper) |
| :--- | :--- | :--- | :--- |
| **Source Initialization** | ImageNet Pretrained | Experiment A Checkpoint | Experiment A Checkpoint |
| **Unfrozen Feature Blocks** | 0 | 2 (`features[17:19]`) | 5 (`features[14:19]`) |
| **Trainable Parameters** | 6,405 (0.29%) | 892,485 (40.02%) | 1,687,749 (75.67%) |
| **Optimizer** | Adam | Adam | Adam |
| **Learning Rate** | $1 \times 10^{-3}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **Batch Size** | 16 | 16 | 16 |
| **Epochs** | 10 | 10 | 10 |
| **Random Seed** | 42 | 42 | 42 |
| **Image Resolution** | $224 \times 224$ | $224 \times 224$ | $224 \times 224$ |
| **Pretrained BatchNorm Stats** | Frozen (`eval()`) | Frozen (`eval()`) | Frozen (`eval()`) |
| **Checkpoint Selection Rule** | Max Validation Accuracy | Max Validation Accuracy | Max Validation Accuracy |

### Note on Comparative Scope
Experiment A used a learning rate of $1 \times 10^{-3}$ and trained only the randomly initialized linear classifier directly on ImageNet feature embeddings. Experiments B and C utilize a reduced learning rate ($1 \times 10^{-4}$) to prevent catastrophic forgetting of pretrained weights while fine-tuning convolutional representations. Therefore, the overall study evaluates the progression from a frozen-backbone baseline to two controlled depths of fine-tuning, rather than identical parameter update budgets.

---

## 6. Preprocessing, Data Augmentation, and Batch Normalization

### Data Pipeline
- **Training Augmentations:**
  - Random Resized Crop: $224 \times 224$ pixels, scaling range $[0.8, 1.0]$.
  - Random Horizontal Flip ($p = 0.5$).
  - Conversion to floating-point tensor ($[0.0, 1.0]$).
  - Normalization using standard ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Validation Preprocessing:**
  - Deterministic evaluation transform matching torchvision's default MobileNetV2 evaluation: bilinear resize to 256 pixels, central crop to $224 \times 224$, tensor conversion, and ImageNet normalization.

### Batch Normalization Handling
Pretrained `BatchNorm2d` layers contain running statistics (`running_mean` and `running_var`) accumulated over ImageNet's 1.28 million images. On small datasets (such as 2,567 training images) with small mini-batch sizes (16), dynamically updating these statistics with batch estimates can introduce high variance and degrade feature representations.

Consequently, during training epochs in all experiments:
- All `nn.BatchNorm2d` modules in `model.features` are kept explicitly in evaluation mode (`module.eval()`).
- Normalization utilizes the stable ImageNet running statistics without updating them during the forward pass.
- In unfrozen blocks, learnable affine parameters ($\gamma$ and $\beta$) continue to receive gradient updates through standard backpropagation.

---

## 7. Hardware Safeguards and Memory Management

To guarantee reliable execution within a 4 GB GPU VRAM envelope (NVIDIA RTX 3050 Laptop GPU):
- Mini-batch size default is set to 16, with CLI override support for 8 if needed.
- `optimizer.zero_grad(set_to_none=True)` is used to eliminate gradient tensor allocation overhead.
- DataLoader uses CUDA pinned memory (`pin_memory=True`) with a constrained worker count (`num_workers=2`).
- Evaluation runs strictly within `torch.no_grad()` to eliminate computation graph memory accumulation.

---

## 8. Evaluation Metrics and Reporting

1. **Classification Accuracy:** Computed across all examples as:
   $$\text{Accuracy} = \frac{\sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)}{N}$$
2. **Sample-Weighted Epoch Loss:** Computed over all $N$ partition examples using multi-class Cross-Entropy:
   $$\text{Loss}_{\text{epoch}} = \frac{1}{N} \sum_{b=1}^{B} \sum_{k=1}^{n_b} \ell(\hat{y}_{b,k}, y_{b,k})$$
3. **Execution Duration:** Per-epoch duration (seconds) and cumulative training duration (seconds).
4. **Checkpoint Selection:** The model checkpoint achieving the highest validation accuracy is stored as `{experiment}_best.pth`.

---

## 9. Reproducibility Controls

- **RNG Initialization:** Master seed `42` is set across Python's `random`, NumPy's `random`, PyTorch CPU (`torch.manual_seed`), and PyTorch CUDA (`torch.cuda.manual_seed_all`).
- **cuDNN Flags:** Deterministic algorithms enabled (`torch.backends.cudnn.deterministic = True`, `benchmark = False`).
- **Metadata Archiving:** Every saved checkpoint embeds the random seed, learning rate, batch size, unfrozen block count, trainable parameter count, and source checkpoint path.
- **Reproducibility Scope:** While random number generators and deterministic algorithms are fixed, minor numerical non-determinism across different GPU architectures or CUDA library builds may exist; claims of bitwise reproducibility are not asserted across disparate systems.

---

## 10. Threats to Validity and Experimental Limitations

1. **Hardware Constraints:** Training is performed on an entry-level mobile GPU (4 GB VRAM), limiting batch sizes to $\le 16$.
2. **Single Baseline Origin:** Both fine-tuning experiments branch from the single best checkpoint of Experiment A, reflecting realistic research workflows while coupling downstream outcomes to Experiment A's optimization trajectory.
3. **Domain Similarity:** Flower classification shares low-level textural and edge features present in ImageNet-1K (which already contains several floral and plant classes), which may favor frozen feature extractors relative to domains with significant distribution shift (e.g., medical radiography).

