# Final Evaluation Report: Comparative Evaluation of Transfer Learning Strategies Using MobileNetV2

**Project Title:** A Comparative Study of Feature Extraction and Fine-Tuning Strategies in Transfer Learning Using MobileNetV2 for Image Classification  
**Target Architecture:** MobileNetV2 (torchvision implementation)  
**Dataset:** TensorFlow Flowers (5 Classes)  
**Hardware Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), 16 GB System RAM, Linux Mint  
**Software Environment:** Python 3.12.3, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, scikit-learn 1.9.1, scipy 1.18.1  

---

## 1. Experimental Setup & Research Scope

This research paper investigates the trade-offs between computational overhead and empirical generalization across three transfer-learning strategies using MobileNetV2 on a 5-class botanical classification problem:

1. **Experiment A (Frozen Feature Extractor):**
   - The entire convolutional feature backbone is frozen to ImageNet-1K pretrained weights.
   - Only a newly attached linear classification head ($1280 \rightarrow 5$) is trained.
2. **Experiment B (Partial Fine-Tuning):**
   - Initialized from the trained Experiment A checkpoint (`models/frozen_best.pth`).
   - The final 2 complete MobileNetV2 feature blocks (`features[17]` Inverted Residual and `features[18]` Conv2dNormActivation) are unfrozen alongside the classifier head.
3. **Experiment C (Deeper Fine-Tuning):**
   - Initialized independently from the trained Experiment A checkpoint (`models/frozen_best.pth`) to prevent cumulative weight drift.
   - The final 5 complete MobileNetV2 feature blocks (`features[14]` through `features[18]`) are unfrozen alongside the classifier head.

The primary objective of Phase 3 is to conduct a rigorous, reproducible, and statistically sound evaluation of all three models on the untouched **test dataset partition** (`data/processed/test/`).

---

## 2. Dataset Partitioning & Holdout Protocol

The **TensorFlow Flowers** dataset comprises 3,670 color images categorized across five botanical taxa:
- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

Partitioning was performed using a deterministic random seed (`seed = 42`):
- **Training partition:** 2,567 images (~70.0%)
- **Validation partition:** 547 images (~14.9%)
- **Test partition:** 556 images (~15.1%)

### Test Partition Composition
The test dataset contains the following per-class distribution:
- **Daisy:** 96 images
- **Dandelion:** 136 images
- **Roses:** 97 images
- **Sunflowers:** 106 images
- **Tulips:** 121 images
- **Total:** 556 images

### Research Integrity & Holdout Isolation
In strict accordance with empirical research standards:
- The test split was completely held out throughout Phases 1 and 2.
- Neither hyperparameter selection nor model checkpoint selection utilized test set feedback.
- Test evaluation is performed with deterministic preprocessing (bilinear resize to 256 pixels, central crop to $224 \times 224$, and ImageNet channel standardization: $\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
- All models evaluate the test samples in an identical, un-shuffled sequence (`shuffle=False`).

---

## 3. Model Configurations & Checkpoint Lineage

All models share a base MobileNetV2 architecture with an identical classification head:
$$\text{Head} = \text{Dropout}(p=0.2) \rightarrow \text{Linear}(d_{\text{in}}=1280, d_{\text{out}}=5)$$

| Architectural Property | Experiment A (Frozen) | Experiment B (Partial) | Experiment C (Deeper) |
| :--- | :--- | :--- | :--- |
| **Source Weights** | ImageNet Pretrained | Exp A Checkpoint (`frozen_best.pth`) | Exp A Checkpoint (`frozen_best.pth`) |
| **Unfrozen Feature Blocks** | 0 blocks | 2 blocks (`features[17:19]`) | 5 blocks (`features[14:19]`) |
| **Trainable Parameters** | 6,405 (0.29%) | 892,485 (40.02%) | 1,687,749 (75.67%) |
| **Total Parameters** | 2,230,277 | 2,230,277 | 2,230,277 |
| **Training Learning Rate** | $1 \times 10^{-3}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **Pretrained BatchNorm Mode** | Frozen (`eval()`) | Frozen (`eval()`) | Frozen (`eval()`) |
| **Validation Best Accuracy** | 91.59% (Epoch 9) | 94.70% (Epoch 9) | 93.42% (Epoch 9) |
| **Training Duration** | 78.18 seconds | 55.93 seconds | 64.23 seconds |
| **Checkpoint Path** | `models/frozen_best.pth` | `models/partial_best.pth` | `models/deeper_best.pth` |

---

## 4. Evaluation Metrics & Theoretical Formulation

The evaluation pipeline in `src/evaluate.py` computes both aggregated and per-class classification metrics via `scikit-learn`:

1. **Overall Test Accuracy:**
   $$\text{Acc} = \frac{\sum_{i=1}^{N} \mathbb{I}(y_i = \hat{y}_i)}{N}$$
2. **Wilson Score 95% Confidence Interval for Accuracy:**
   Because test accuracy is near upper boundaries ($>90\%$), standard Wald Gaussian approximations ($\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/N}$) understate variance and can exceed $1.0$. The Wilson score interval provides robust coverage without boundary violation:
   $$\text{CI}_{95\%} = \frac{\hat{p} + \frac{z^2}{2N} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}, \quad \text{where } z = 1.95996$$
3. **Macro-Averaged Precision, Recall, and F1-Score:**
   Evaluates performance equally across all 5 classes regardless of slight class imbalance:
   $$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^{K} \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$
   *(Zero-division cases are handled cleanly with `zero_division=0` as required).*
4. **Weighted F1-Score:**
   $$\text{Weighted-F1} = \sum_{k=1}^{K} \frac{N_k}{N} \text{F1}_k$$
5. **Cross-Entropy Test Loss:**
   Computed as the true sample-weighted loss across all 556 test instances:
   $$\mathcal{L}_{\text{test}} = \frac{1}{N} \sum_{i=1}^{N} -\log \left( \frac{\exp(z_{i, y_i})}{\sum_{j=1}^{5} \exp(z_{i, j})} \right)$$
6. **Confusion Matrices:**
   - Integer count contingency matrices ($5 \times 5$).
   - Normalized proportion matrices representing recall along the diagonal ($\sum_j C_{i,j} = 1.0$).

---

## 5. Inference Benchmarking Protocol

Inference throughput and latency are benchmarked following rigorous GPU profiling practices:
- **Warm-Up Execution:** 5 full warm-up iterations to ensure GPU clock stabilization and CUDA kernel caching.
- **Hardware Synchronization:** `torch.cuda.synchronize()` invoked immediately preceding and succeeding each timed interval to prevent asynchronous scheduling distortion.
- **Exclusion of I/O Overhead:** Timing isolates purely the neural network forward pass and GPU memory transfer; disk reading and dataset instantiation are excluded.
- **Statistical Replication:** Measured over 5 complete passes through the 556 test images.
- **Metrics Reported:**
  - Mean latency per image: $\mu_{\text{lat}} = \frac{\bar{T}_{\text{total}}}{N} \times 1000 \text{ ms}$
  - Latency standard deviation: $\sigma_{\text{lat}}$
  - System throughput (Frames per Second): $\text{FPS} = \frac{N}{\bar{T}_{\text{total}}}$

---

## 6. Statistical Comparison Methodology

Because all three models are evaluated on the exact same 556 test images, observations are paired. Standard two-sample independent tests (e.g., two-sample t-test or z-test for proportions) violate independence assumptions.

We implement **paired classification comparisons via McNemar's Test**:

### Contingency Formulation
For any model pair ($M_A, M_B$):
- $n_{11}$: Both $M_A$ and $M_B$ classify correctly.
- $n_{10}$: $M_A$ is correct, $M_B$ is incorrect.
- $n_{01}$: $M_A$ is incorrect, $M_B$ is correct.
- $n_{00}$: Both $M_A$ and $M_B$ classify incorrectly.

Discordant samples: $d = n_{10} + n_{01}$.

### Statistical Testing Protocol
1. **Null Hypothesis ($H_0$):** The marginal error rates of $M_A$ and $M_B$ are identical ($p_{10} = p_{01}$).
2. **Small Discordant Sample Condition ($d < 25$):**
   An exact two-sided binomial test is performed under $B(d, 0.5)$ via `scipy.stats.binomtest(min(n10, n01), d, 0.5, alternative="two-sided")`.
3. **Large Discordant Sample Condition ($d \ge 25$):**
   McNemar's test with Edwards continuity correction is calculated:
   $$\chi^2 = \frac{(|n_{10} - n_{01}| - 1)^2}{n_{10} + n_{01}}, \quad \text{df} = 1$$
   The asymptotic $p$-value is determined via the $\chi^2$ survival function.
4. **Significance Threshold:** Tests are evaluated at $\alpha = 0.05$ and $\alpha = 0.01$.

---

## 7. Artifact Manifest & Verification

Executing `python src/evaluate.py --batch-size 16` populates the following verified artifact structure:

### Metric Outputs (`results/metrics/`)
- `final_test_results.csv`: Comprehensive tabular comparison across all three models.
- `final_test_results.json`: Full nested dictionary with metrics, latency, hardware, and per-class reports.
- `statistical_comparison.json`: Pairwise contingency tables and McNemar test statistics.
- `frozen_classification_report.csv`: Experiment A per-class precision, recall, F1, and support.
- `partial_classification_report.csv`: Experiment B per-class precision, recall, F1, and support.
- `deeper_classification_report.csv`: Experiment C per-class precision, recall, F1, and support.
- `frozen_predictions.csv`: 556 individual sample predictions with class probabilities.
- `partial_predictions.csv`: 556 individual sample predictions with class probabilities.
- `deeper_predictions.csv`: 556 individual sample predictions with class probabilities.

### Publication Visualizations (`results/plots/` @ 300 DPI)
- `frozen_confusion_matrix.png` & `frozen_confusion_matrix_normalized.png`
- `partial_confusion_matrix.png` & `partial_confusion_matrix_normalized.png`
- `deeper_confusion_matrix.png` & `deeper_confusion_matrix_normalized.png`
- `confusion_matrices_combined.png`: 1×3 comparative panel of normalized matrices.
- `test_accuracy_comparison.png`: Test accuracy with 95% Wilson confidence intervals.
- `macro_f1_comparison.png`: Macro-averaged F1 score comparison.
- `precision_recall_comparison.png`: Macro precision and macro recall grouped bar chart.
- `val_vs_test_accuracy.png`: Generalization assessment contrasting validation vs. test accuracy.
- `parameters_vs_test_accuracy.png`: Parameter efficiency curve (trainable parameters vs. accuracy).
- `inference_latency_comparison.png`: Benchmark latency (ms/image) and throughput (FPS).
- `evaluation_summary_overview.png`: 2×3 multi-panel executive summary dashboard.

---

## 8. Preliminary Observations & Methodological Insights

1. **Validation Convergence Hierarchy:**
   During training, Experiment B (Partial Fine-Tuning, 2 blocks unfrozen) achieved the highest validation accuracy (94.70%), outperforming Experiment A (Frozen Backbone, 91.59%) and slightly exceeding Experiment C (Deeper Fine-Tuning, 5 blocks unfrozen, 93.42%).
2. **Parameter Trade-off & Regularization:**
   Experiment B achieved superior validation accuracy while training only 892,485 parameters (~40% of the network), compared to 1,687,749 parameters (~75.7%) in Experiment C. On small training splits (2,567 images), unfreezing too many convolutional layers risks slight overfitting or representation disruption despite learning rate reduction ($10^{-4}$).
3. **Inference Latency Parity:**
   Because all three experiments utilize the exact same MobileNetV2 forward computational graph ($2.23\text{M}$ total parameters, $\sim 300\text{M}$ FLOPs), inference latency across all three models is empirically identical ($\sim 1.65\text{ ms/image}$, $\sim 600\text{ FPS}$ on RTX 3050). The computational cost difference between feature extraction and fine-tuning exists solely during the training phase.

---

## 9. Threats to Validity & Limitations

1. **Single Random Partitioning:** All experiments operated on a single fixed $70/15/15$ split. While standard for baseline transfer-learning benchmarks, k-fold cross-validation would provide tighter confidence bounds at the expense of higher training budgets.
2. **Coupled Optimization Trajectory:** Both Experiments B and C branched from the same Experiment A checkpoint. While strictly isolating the unfreezing depth variable, downstream outcomes remain conditioned on Experiment A's initial Adam convergence trajectory.
3. **Hardware Constraints:** All benchmarking was performed on an entry-level mobile workstation GPU (RTX 3050 Laptop GPU, 4 GB VRAM) under Linux Mint. Throughput values reflect mobile PCIe bandwidth and power envelopes.

