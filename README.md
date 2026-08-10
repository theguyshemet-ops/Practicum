# Vulnerability Analysis of ViT vs CNN in Autonomous Perception

This codebase implements the core technical foundation (Weeks 1-2) for a comparative research framework evaluating the robustness of a Multi-Head Self-Attention (MHSA) Vision Transformer against a Convolutional Neural Network (YOLOv10-S) under frequency-constrained physical adversarial patches.

---

# Technical Architecture & Directory Structure

```
vit-vs-cnn-robustness/
├── data/
│   ├── nuscenes/         # Standard nuScenes-mini split images and metadata
│   ├── gtsrb/            # German Traffic Sign Recognition Benchmark dataset
│   └── carla/            # CARLA physical domain scene renders (W6-7)
├── notebooks/            # High-performance fine-tuning on Google Colab A100
│   ├── colab_yolov10_finetuning.ipynb
│   └── colab_vitdet_finetuning.ipynb
├── src/
│   ├── data_prep/        # Week 1: Data ingestion and coordinate projections
│   │   ├── nuscenes_loader.py
│   │   └── gtsrb_loader.py
│   ├── models/           # Week 2: Unified detection wrappers & visualization hooks
│   │   ├── yolov10_wrapper.py
│   │   └── detr_vit_wrapper.py
│   ├── training/         # Week 2: Mixed-precision parameterizable fine-tuning loops
│   │   ├── train_yolov10.py
│   │   └── train_detr_vit.py
│   └── utils/
│       ├── metrics.py    # Standard evaluation metrics (mAP, RSF, Adversarial Delta)
│       └── setup_env.py  # Automated dependency checks and folder creation
├── requirements.txt      # Dependency lists
└── README.md             # Running and verification instructions
```

---

# Features & Implementation Details

# 1. Robust Coordinates Projection (`nuscenes_loader.py`)
- Real 3D-to-2D projection: Project 3D LiDAR/radar boxes to the front camera coordinate frame using intrinsic matrix multiplication and ego pose transformation.
- **Premium CARLA Road Generator**: If no physical dataset is found, it automatically switches to a CARLA rendering mode. It draws high-fidelity urban layouts with grass, perspective roads, dividing stripes, time-of-day skies, and drawn vehicles/pedestrians, along with perfect ground-truth coordinates. This guarantees the entire project is runnable instantly!

# 2. Grad-CAM and Attention Rollout Hooks (`src/models/`)
- **YOLOv10 Wrapper**: Registers backward hooks on key convolutional layers in the backbone, caching intermediate activations and gradients to support Grad-CAM diagnostic heatmaps.
- **DETR-ViT Wrapper**: Configures Hugging Face YOLOS with `output_attentions=True`, extracting the Multi-Head Self-Attention maps across all layers. Provides a standard `get_attention_rollout()` method to fuse matrices using mean/max strategies, enabling recursive trace maps of attention-hijacking.

# 3. VRAM-Safe Training Pipelines (`src/training/`)
- Implements PyTorch Automatic Mixed Precision (AMP) via `torch.cuda.amp.autocast` to execute training in FP16.
- Integrates gradient accumulation to simulate large training batch sizes while staying strictly under a 5.0 GB local VRAM ceiling.

---

# Local Verification & Quickstart

Follow these steps to run smoke tests and verify the environment on your Windows laptop (RTX 4050):

# Step 1: Initialize folders and verify dependencies
```bash
python src/utils/setup_env.py
```

# Step 2: Run Dataset Loader Smoke Tests
Verify that the datasets load, render CARLA scenes, and format bounding boxes correctly:
```bash
python src/data_prep/nuscenes_loader.py
python src/data_prep/gtsrb_loader.py
```

# Step 3: Run Model Forward-Pass Smoke Tests
Confirm that both model wrappers initialize, load default/mock configs, register diagnostic hooks, and execute inferences successfully:
```bash
python src/models/yolov10_wrapper.py
python src/models/detr_vit_wrapper.py
```

# Step 4: Run Fine-Tuning Dry Runs
Test the training loops by executing a 2-epoch dry run locally in CARLA mode to ensure memory safety (under 5 GB VRAM peak):
```bash
# YOLOv10-S fine-tuning dry run
python src/training/train_yolov10.py --epochs 2 --batch_size 4 --grad_accum 2 --use_carla

# DETR-ViT (YOLOS) fine-tuning dry run
python src/training/train_detr_vit.py --epochs 2 --batch_size 2 --grad_accum 4 --use_carla
```
--