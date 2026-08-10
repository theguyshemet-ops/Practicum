# Vulnerability Analysis of Vision Transformers vs. CNNs Under Frequency-Aware Physical Adversarial Constraints in Autonomous Perception

**MSc in Artificial Intelligence (Practicum) — National College of Ireland (NCI)**  
**Author**: Kunal Arya  
**Student ID**: 24243833  

---

## 📌 Executive Summary

This repository contains the complete, reproducible source code, datasets, evaluation results, figures, report deliverables, and Configuration Manual for the research project evaluating the physical adversarial robustness of **Vision Transformers (ViT)** versus **Convolutional Neural Networks (CNN)** under **2D Discrete Cosine Transform (DCT) frequency constraints**.

### 🌟 Key Findings
1. **High-Frequency Spectral Resilience**: Under high-frequency perturbations ($r > 16$), YOLOS-Small showed a **44.1% relative mAP decrease**, compared with a **53.9% decrease** for Faster R-CNN v2. This pattern supports H1 within the evaluated models, data, and attack configuration, indicating lower observed high-frequency sensitivity for YOLOS-Small.
2. **Low-Frequency Structural Vulnerability**: Low-frequency structural patches ($r \le 8$) reduced mAP by **more than 50%** in both detectors ($52.1\%$ drop in Faster R-CNN vs. $50.4\%$ in YOLOS-Small), supporting H2 across architectural boundaries.
3. **Shallower Degradation Slope (RSF)**: Sweep analysis across patch area ratios ($0.0$ to $0.5$) revealed that YOLOS-Small exhibited an **RSF slope approximately half that of Faster R-CNN** ($0.1192$ versus $0.2321$), indicating greater tolerance to expanding patch coverage.

---

## 📁 Repository Structure

```
MainFolder/
├── src/                          # Modular Python source code
│   ├── attacks/                  # 2D DCT filter, EoT pipeline, PGD patch optimizer, dual loss
│   │   ├── adversarial_loss.py   # Dual-objective loss (ReLU-hinge suppression + symmetric KL divergence)
│   │   ├── dct_filter.py         # Differentiable 2D DCT-II and inverse DCT-III frequency mask
│   │   ├── eot_pipeline.py       # 6-transform Expectation over Transformation pipeline
│   │   └── patch_optimizer.py    # PGD patch optimization engine
│   ├── data_prep/                # Dataset loaders
│   │   ├── carla_loader.py       # CARLA perception dataset loader (weather & distance filtering)
│   │   ├── gtsrb_loader.py       # GTSRB traffic sign patch loader
│   │   └── nuscenes_loader.py    # nuScenes-mini front-camera CAM_FRONT loader
│   ├── experiments/              # Experiment runners & visualization
│   │   ├── aggregate_results.py  # Result aggregator across random seeds (N=5)
│   │   ├── compare_with_references.py # Qualitative prior literature contextualization
│   │   ├── main_experiment.py   # Core experiment execution pipeline
│   │   ├── run_carla_experiments.py # CARLA cross-domain runner
│   │   ├── run_experiments.py   # nuScenes main experiment runner
│   │   └── visualize_results.py # 300 DPI publication plot generator
│   ├── models/                   # Model wrappers & factory
│   │   ├── faster_rcnn_wrapper.py# Faster R-CNN v2 wrapper with Grad-CAM hooks
│   │   ├── model_factory.py     # Unified model instantiation factory
│   │   └── yolos_wrapper.py      # YOLOS-Small wrapper with Attention Rollout hooks
│   ├── training/                 # Patch optimization utilities
│   └── utils/                    # Metrics, seed utils, and visualization helpers
├── data/                         # Datasets (nuScenes-mini, CARLA, GTSRB)
├── results/                      # Evaluation JSONs, LaTeX tables, and 300 DPI figures
│   ├── carla/                    # CARLA domain evaluation results
│   ├── evaluation/               # Aggregated metrics & statistical p-values
│   └── figures/                  # Publication-ready plots (Figures 1 to 8)
├── config_manual_latex/          # NCI Configuration Manual LaTeX source project
├── checkpoints/                  # Pre-trained model checkpoints
├── Research_Report.docx          # Master 21-body-page MSc Research Report
├── Dissertation_Presentation.pptx# 12-slide Viva Defense presentation deck
├── Configuration_Manual.docx     # Compiled Configuration Manual Word document
├── Configuration_Manual_LaTeX.zip# Overleaf-ready Configuration Manual package
├── requirements.txt              # Environment dependencies list
└── README.md                     # Repository documentation
```

---

## 🛠️ Environment Setup & Installation

### Prerequisites
* **Operating System**: Windows 11 / Linux (Ubuntu 22.04)
* **GPU Hardware**: NVIDIA GeForce GPU (minimum 6 GB VRAM budget, CUDA 12.1)
* **Python**: Python 3.11+ via Anaconda / Miniconda

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/theguyshemet-ops/Practicum.git
   cd Practicum
   ```

2. **Create and Activate Conda Environment**:
   ```bash
   conda create -n vit_vs_yolo python=3.11 -y
   conda activate vit_vs_yolo
   ```

3. **Install PyTorch with CUDA 12.1 Support**:
   ```bash
   pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Dataset Setup Instructions

### 1. nuScenes-mini Dataset Setup
1. Register and download the **nuScenes-mini** split (`v1.0-mini.tgz`) from [www.nuscenes.org](https://www.nuscenes.org).
2. Extract the archive contents into `data/nuscenes/`:
   ```
   data/nuscenes/
   └── v1.0-mini/
   ```

### 2. CARLA Autonomous Perception Dataset Setup
1. Place pre-rendered Unreal Engine CARLA frames and bounding box text annotations in `data/carla/`.
2. The dataset loader automatically parses weather subsets (`clear`, `rain`, `fog`) and vehicle distance bands ($5\text{m}$ to $30\text{m}$).

---

## 🚀 Running Experiments & Generating Results

### 1. Run Main Experiment Suite (nuScenes-mini)
Execute the frequency-stratified PGD attack optimization across 3 spectral bands (Low, High, Full) and 4 epsilon budgets ($\varepsilon \in \{0.1, 0.2, 0.3, 0.5\}$) over $N=5$ seeds:
```bash
python -m src.experiments.run_experiments --data_root data/nuscenes --results_dir results/ --num_steps 150 --alpha 0.005
```

### 2. Run CARLA Cross-Domain Validation Suite
Execute weather and distance ablation evaluations on the CARLA driving domain:
```bash
python -m src.experiments.run_carla_experiments --data_root data/carla --results_dir results/carla/
```

### 3. Aggregate Statistical Results
Aggregate JSON evaluation metrics, compute sample standard deviations ($s$), and generate Welch's $t$-test $p$-values:
```bash
python -m src.experiments.aggregate_results --results_dir results/
```

### 4. Generate 300 DPI Publication Plots
Generate all 8 high-resolution figures saved to `results/figures/`:
```bash
python -m src.experiments.visualize_results --results_dir results/
```

### 5. Run Literature Contextualization Analysis
Generate qualitative contextual comparison tables comparing empirical findings against published baselines:
```bash
python -m src.experiments.compare_with_references
```

---

## 🔍 Qualitative Diagnostics: Grad-CAM & Attention Rollout

To extract backbone feature maps and attention heatmaps:

* **Faster R-CNN Grad-CAM**: Registers backward hooks on `backbone.body.layer4` to compute gradient-weighted activation maps.
* **YOLOS-Small Attention Rollout**: Recursively computes cumulative attention flow across transformer encoder layers:
  $$\mathbf{A} = \prod_{l=1}^{L} \left(0.5 \, \mathbf{W}^{(l)} + 0.5 \, \mathbf{I}\right)$$

---

## 📜 License & Citation

This project is licensed under the MIT License