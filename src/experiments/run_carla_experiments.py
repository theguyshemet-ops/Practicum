"""
run_carla_experiments.py - CARLA-specific experimental pipeline for ViT vs CNN robustness.

Orchestrates:
1. Patch optimisation across 3 frequency bands (low / high / full)
2. Epsilon sweep across 4 perturbation budgets (0.1, 0.2, 0.3, 0.5)
3. Evaluation of all patches on both Faster R-CNN and YOLOS-Small
4. RSF curve generation (mAP vs patch area ratio)
5. Adversarial Delta computation (ViT mAP - CNN mAP per band)
6. Weather ablation: evaluate clear-trained patches on rain/fog subsets
7. Distance ablation: evaluate patches at each distance (5, 10, 15, 20, 30 m)

All results are saved as structured JSON under a configurable results_dir
(default: results/carla/).

Usage:
    python -m src.experiments.run_carla_experiments \\
        --data_dir data/carla --results_dir results/carla

    # Quick smoke test:
    python -m src.experiments.run_carla_experiments \\
        --data_dir data/carla --results_dir results/carla --num_steps 5
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so `from src.xxx` imports work
# regardless of which directory the script is run from.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# -- Project imports -----------------------------------------------------
from src.models.faster_rcnn_wrapper import FasterRCNNWrapper
from src.models.detr_vit_wrapper import DetrVitWrapper
from src.attacks.patch_optimizer import PatchOptimizer
from src.attacks.patch_applier import PatchApplier
from src.attacks.dct_filter import DCTFrequencyMask, apply_dct_mask
from src.utils.metrics import (
    calculate_map,
    calculate_robustness_sensitivity_factor,
    calculate_adversarial_delta,
)


# =======================================================================
# Helper: CARLA DataLoader
# =======================================================================

def _build_carla_dataloader(
    data_dir: str,
    split: str,
    batch_size: int = 1,
    weather_filter: Optional[str] = None,
    distance_filter: Optional[float] = None,
):
    """
    Build a DataLoader from the CARLA dataset.
    Returns images (B, 3, 640, 640) and a list of target dicts.

    Parameters
    ----------
    data_dir : str
        Path to CARLA data directory.
    split : str
        Dataset split ('train' or 'val').
    batch_size : int
        Batch size (default: 1 for RTX 4050 6 GB VRAM).
    weather_filter : str, optional
        If set, only include images with this weather condition.
    distance_filter : float, optional
        If set, only include images at this distance (metres).
    """
    from src.data_prep.carla_loader import CARLADataset

    dataset = CARLADataset(
        data_root=data_dir,
        split=split,
        weather_filter=weather_filter,
        distance_filter=distance_filter,
    )
    print(f"[Data] Loaded CARLA {split}: {len(dataset)} samples"
          f" (weather={weather_filter}, distance={distance_filter})")

    def collate_fn(batch):
        """Custom collate: stack images, collect targets as list of dicts."""
        images = torch.stack([s["image"] for s in batch])
        targets = []
        for s in batch:
            targets.append({
                "boxes": s["bboxes"],
                "labels": s["labels"],
            })
        return images, targets

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return dataset, loader


# =======================================================================
# Stage 1: Patch Optimisation
# =======================================================================

def run_optimisation(
    rcnn: FasterRCNNWrapper,
    vit: DetrVitWrapper,
    images: torch.Tensor,
    bboxes_list: List[torch.Tensor],
    band: str,
    epsilon: float,
    num_steps: int,
    results_dir: str,
    device: str = "cuda",
) -> Tuple[torch.Tensor, dict]:
    """
    Run a single patch optimisation campaign.

    Parameters
    ----------
    rcnn, vit : detector wrappers (frozen)
    images : (N, 3, H, W) training images
    bboxes_list : list of (K_i, 4) target bounding boxes per image
    band : 'low' | 'high' | 'full'
    epsilon : perturbation budget
    num_steps : PGD iterations
    results_dir : directory to save outputs
    device : 'cuda' or 'cpu'

    Returns
    -------
    best_patch : (3, 256, 256) optimised patch tensor on CPU
    history : dict of per-step metrics
    """
    exp_name = f"patch_{band}_eps{epsilon:.1f}"
    exp_dir = os.path.join(results_dir, "experiments", f"{band}_eps{epsilon:.1f}")
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  Optimisation: band={band}, epsilon={epsilon}, steps={num_steps}")
    print(f"  Output: {exp_dir}")
    print(f"{'=' * 72}")

    optimizer = PatchOptimizer(
        rcnn_wrapper=rcnn,
        vit_wrapper=vit,
        patch_size=256,
        patch_ratio=0.3,
        placement_mode="centre",
        lr=0.02,
        num_steps=num_steps,
        tau=0.1,
        alpha=1.0,
        beta=1.0,
        gamma=0.3,
        epsilon=epsilon,
        device=device,
    )

    best_patch, history = optimizer.optimize(
        images=images,
        bboxes_list=bboxes_list,
        band=band,
        save_every=50,
        log_dir=exp_dir,
        experiment_name=exp_name,
    )

    # Free optimizer memory
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return best_patch, history


# =======================================================================
# Stage 2: Evaluation
# =======================================================================

def evaluate_single_patch(
    model,
    dataset,
    patch: Optional[torch.Tensor],
    model_name: str,
    device: str = "cuda",
    patch_ratio: float = 0.3,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate a single patch against a model on the full dataset.
    If patch is None, evaluates clean baseline.

    Returns (mAP, per_image_results).
    """
    from src.attacks.patch_applier import PatchApplier
    from src.utils.metrics import calculate_map

    applier = PatchApplier(placement_mode="centre", patch_ratio=patch_ratio)
    target_device = torch.device(device if torch.cuda.is_available() else "cpu")

    model.eval()
    predictions_list = []
    ground_truths_list = []
    per_image_results = []

    mode_str = "Clean" if patch is None else f"Patched (ratio={patch_ratio})"
    print(f"  [{model_name}] Evaluating: {mode_str} on {len(dataset)} images...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample["image"].to(target_device)

            if "bboxes" in sample and sample["bboxes"].numel() > 0:
                bboxes = sample["bboxes"].to(target_device)
                labels = sample["labels"].to(target_device)
            else:
                h, w = image.shape[1:]
                bboxes = torch.tensor([[0, 0, w, h]], dtype=torch.float32, device=target_device)
                labels = torch.tensor([0], dtype=torch.long, device=target_device)

            if patch is not None and bboxes.numel() > 0:
                patch_t = patch.to(target_device)
                if patch_t.ndim == 3:
                    patch_t = patch_t.unsqueeze(0)
                patched_img = applier(image.unsqueeze(0), patch_t, [bboxes])
                outputs = model(patched_img)
            else:
                outputs = model(image.unsqueeze(0))

            out = outputs[0]
            preds = []
            for b, s, l in zip(out["boxes"], out["scores"], out["labels"]):
                # Class-agnostic evaluation for cross-domain (GTSRB→CARLA):
                # Treat every detection as a generic "traffic_sign" (label=0)
                # because GTSRB class IDs won't match CARLA's 5 sign types.
                preds.append({
                    "box": b.cpu().tolist(),
                    "score": s.cpu().item(),
                    "label": 0,
                })
            predictions_list.append(preds)

            gts = []
            for b, l in zip(bboxes.cpu(), labels.cpu()):
                # Also set GT labels to 0 for class-agnostic matching
                gts.append({"box": b.tolist(), "label": 0})
            ground_truths_list.append(gts)

            max_conf = max([p["score"] for p in preds], default=0.0)
            per_image_results.append({
                "image_idx": idx,
                "num_detections": len(preds),
                "max_confidence": max_conf,
                "ground_truth_count": len(gts),
            })

    avg_det = np.mean([r["num_detections"] for r in per_image_results])
    avg_conf = np.mean([r["max_confidence"] for r in per_image_results])
    det_rate = np.mean([1.0 if r["num_detections"] > 0 else 0.0 for r in per_image_results])
    print(f"  [{model_name}] DetRate={det_rate:.2%} | Avg det={avg_det:.1f} | Avg conf={avg_conf:.4f}")

    # Return avg_conf as the primary "score" metric.
    # In cross-domain evaluation (GTSRB→CARLA), mAP is not viable because
    # models cannot localise small CARLA-rendered signs (IoU≈0). Average confidence
    # correctly measures the adversarial patch's suppression effect.
    return avg_conf, per_image_results


def evaluate_all_patches(
    rcnn: FasterRCNNWrapper,
    vit: DetrVitWrapper,
    dataset,
    patches: Dict[str, torch.Tensor],
    results_dir: str,
    device: str = "cuda",
) -> dict:
    """
    Evaluate all optimised patches against both models.
    Returns structured comparison dict.
    """
    eval_dir = os.path.join(results_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  Stage 2: Evaluation (Frequency Comparison)")
    print(f"{'=' * 72}")

    comparison = {"rcnn": {}, "yolos": {}}

    # Clean baselines
    for model, name, key in [(rcnn, "Faster R-CNN", "rcnn"), (vit, "YOLOS-Small", "yolos")]:
        score, _ = evaluate_single_patch(model, dataset, None, name, device)
        comparison[key]["clean"] = {"score": score}

    # Patched evaluations
    for patch_key, patch_tensor in patches.items():
        for model, name, key in [(rcnn, "Faster R-CNN", "rcnn"), (vit, "YOLOS-Small", "yolos")]:
            score, results = evaluate_single_patch(model, dataset, patch_tensor, f"{name}/{patch_key}", device)
            clean_score = comparison[key]["clean"]["score"]
            comparison[key][patch_key] = {
                "score": score,
                "score_drop": clean_score - score,
                "avg_detections": np.mean([r["num_detections"] for r in results]),
                "avg_max_confidence": np.mean([r["max_confidence"] for r in results]),
            }

    # Save
    freq_path = os.path.join(eval_dir, "frequency_comparison.json")
    with open(freq_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved: {freq_path}")

    return comparison


# =======================================================================
# Stage 3: RSF Curves
# =======================================================================

def compute_rsf_curves(
    rcnn: FasterRCNNWrapper,
    vit: DetrVitWrapper,
    dataset,
    patch: torch.Tensor,
    results_dir: str,
    device: str = "cuda",
    area_ratios: Optional[List[float]] = None,
) -> dict:
    """
    Compute RSF curves for both architectures using the full-band patch.
    """
    if area_ratios is None:
        area_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    eval_dir = os.path.join(results_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  Stage 3: RSF Curves (patch ratio sweep)")
    print(f"{'=' * 72}")

    rsf_data = {"rcnn": {"ratios": area_ratios, "scores": []},
                "yolos": {"ratios": area_ratios, "scores": []}}

    for model, name, key in [(rcnn, "Faster R-CNN", "rcnn"), (vit, "YOLOS-Small", "yolos")]:
        scores = []
        for ratio in area_ratios:
            if ratio == 0.0:
                score, _ = evaluate_single_patch(model, dataset, None, f"{name}/ratio=0.0", device)
            else:
                score, _ = evaluate_single_patch(
                    model, dataset, patch, f"{name}/ratio={ratio:.2f}", device,
                    patch_ratio=ratio
                )
            scores.append(score)

        rsf_data[key]["scores"] = scores
        rsf = calculate_robustness_sensitivity_factor(area_ratios, scores)
        rsf_data[key]["rsf"] = rsf
        print(f"  [{name}] RSF slope = {rsf:.4f}")

    rsf_path = os.path.join(eval_dir, "rsf_curves.json")
    with open(rsf_path, "w") as f:
        json.dump(rsf_data, f, indent=2)
    print(f"  Saved: {rsf_path}")

    return rsf_data


# =======================================================================
# Stage 4: Epsilon Sweep
# =======================================================================

def run_epsilon_sweep(
    rcnn: FasterRCNNWrapper,
    vit: DetrVitWrapper,
    dataset,
    all_patches: Dict[str, Dict[str, torch.Tensor]],
    results_dir: str,
    device: str = "cuda",
) -> dict:
    """
    Evaluate patches across all epsilon values and bands.

    Parameters
    ----------
    all_patches : dict
        Nested dict: {eps_str: {band: patch_tensor, ...}, ...}
        e.g. {"0.1": {"low": tensor, "high": tensor, "full": tensor}, ...}
    """
    eval_dir = os.path.join(results_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  Stage 4: Epsilon Sweep Evaluation")
    print(f"{'=' * 72}")

    sweep_results = {}

    for eps_str, band_patches in all_patches.items():
        sweep_results[eps_str] = {}
        for band, patch_tensor in band_patches.items():
            print(f"\n  epsilon={eps_str}, band={band}:")
            rcnn_score, _ = evaluate_single_patch(
                rcnn, dataset, patch_tensor, f"RCNN/epsilon={eps_str}/{band}", device
            )
            yolos_score, _ = evaluate_single_patch(
                vit, dataset, patch_tensor, f"YOLOS/epsilon={eps_str}/{band}", device
            )
            sweep_results[eps_str][band] = {
                "rcnn_score": rcnn_score,
                "yolos_score": yolos_score,
                "adversarial_delta": calculate_adversarial_delta(yolos_score, rcnn_score),
            }

    sweep_path = os.path.join(eval_dir, "epsilon_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n  Saved: {sweep_path}")

    return sweep_results


# =======================================================================
# Stage 5: Weather Ablation
# =======================================================================

def run_weather_ablation(
    rcnn: FasterRCNNWrapper,
    vit: DetrVitWrapper,
    patch: torch.Tensor,
    data_dir: str,
    results_dir: str,
    device: str = "cuda",
    weathers: Optional[List[str]] = None,
) -> dict:
    """
    Evaluate patches trained on 'clear' against rain and fog subsets.

    The patch is assumed to have been optimised on clear-weather CARLA data.
    We then measure transferability to rain and fog conditions.

    Parameters
    ----------
    patch : torch.Tensor
        The full-band patch optimised on clear-weather data.
    weathers : list of str
        Weather conditions to evaluate. Default: ['clear', 'rain', 'fog'].

    Returns
    -------
    dict
        Weather ablation results keyed by weather condition.
    """
    if weathers is None:
        weathers = ["clear", "rain", "fog"]

    eval_dir = os.path.join(results_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  Stage 5: Weather Ablation")
    print(f"{'=' * 72}")

    weather_results = {}

    for weather in weathers:
        print(f"\n  Weather: {weather}")
        dataset, _ = _build_carla_dataloader(
            data_dir, split="val", batch_size=1, weather_filter=weather
        )

        if len(dataset) == 0:
            print(f"    [SKIP] No samples for weather={weather}")
            continue

        weather_results[weather] = {}

        for model, name, key in [(rcnn, "Faster R-CNN", "rcnn"), (vit, "YOLOS-Small", "yolos")]:
            # Clean baseline for this weather
            clean_score, _ = evaluate_single_patch(
                model, dataset, None, f"{name}/{weather}/clean", device
            )
            # Patched evaluation
            patched_score, _ = evaluate_single_patch(
                model, dataset, patch, f"{name}/{weather}/patched", device
            )
            weather_results[weather][key] = {
                "clean_score": clean_score,
                "patched_score": patched_score,
                "score_drop": clean_score - patched_score,
                "num_samples": len(dataset),
            }

    weather_path = os.path.join(eval_dir, "weather_ablation.json")
    with open(weather_path, "w") as f:
        json.dump(weather_results, f, indent=2)
    print(f"\n  Saved: {weather_path}")

    return weather_results


# =======================================================================
# Stage 6: Distance Ablation
# =======================================================================

def run_distance_ablation(
    rcnn: FasterRCNNWrapper,
    vit: DetrVitWrapper,
    patch: torch.Tensor,
    data_dir: str,
    results_dir: str,
    device: str = "cuda",
    distances: Optional[List[float]] = None,
) -> dict:
    """
    Evaluate patches at each distance bin (5, 10, 15, 20, 30 m).

    Parameters
    ----------
    patch : torch.Tensor
        The full-band patch (typically epsilon=0.3).
    distances : list of float
        Distance bins in metres. Default: [5, 10, 15, 20, 30].

    Returns
    -------
    dict
        Distance ablation results keyed by distance string.
    """
    if distances is None:
        distances = [5.0, 10.0, 15.0, 20.0, 30.0]

    eval_dir = os.path.join(results_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  Stage 6: Distance Ablation")
    print(f"{'=' * 72}")

    distance_results = {}

    for dist in distances:
        dist_key = f"{dist:.0f}m"
        print(f"\n  Distance: {dist_key}")

        dataset, _ = _build_carla_dataloader(
            data_dir, split="val", batch_size=1, distance_filter=dist
        )

        if len(dataset) == 0:
            print(f"    [SKIP] No samples for distance={dist_key}")
            continue

        distance_results[dist_key] = {}

        for model, name, key in [(rcnn, "Faster R-CNN", "rcnn"), (vit, "YOLOS-Small", "yolos")]:
            # Clean baseline for this distance
            clean_score, _ = evaluate_single_patch(
                model, dataset, None, f"{name}/{dist_key}/clean", device
            )
            # Patched evaluation
            patched_score, _ = evaluate_single_patch(
                model, dataset, patch, f"{name}/{dist_key}/patched", device
            )
            distance_results[dist_key][key] = {
                "clean_score": clean_score,
                "patched_score": patched_score,
                "score_drop": clean_score - patched_score,
                "num_samples": len(dataset),
            }

    distance_path = os.path.join(eval_dir, "distance_ablation.json")
    with open(distance_path, "w") as f:
        json.dump(distance_results, f, indent=2)
    print(f"\n  Saved: {distance_path}")

    return distance_results


# =======================================================================
# Main Orchestrator
# =======================================================================

def run_carla_experiment(
    data_dir: str,
    results_dir: str,
    num_steps: int = 200,
    bands: Optional[List[str]] = None,
    epsilons: Optional[List[float]] = None,
    device: str = "cuda",
    batch_size: int = 1,
    skip_optim: bool = False,
    skip_eval: bool = False,
    skip_weather: bool = False,
    skip_distance: bool = False,
    rcnn_checkpoint: Optional[str] = None,
    vit_checkpoint: Optional[str] = None,
):
    """
    Run the complete CARLA experimental pipeline:
    1. Load data + models
    2. Optimise patches (3 bands x 4 epsilons)
    3. Evaluate all patches on both models
    4. Compute RSF curves
    5. Run epsilon sweep evaluation
    6. Weather ablation (clear -> rain, fog)
    7. Distance ablation (5, 10, 15, 20, 30 m)
    8. Save all results

    Parameters
    ----------
    data_dir : str
        Path to CARLA data directory.
    results_dir : str
        Path to output directory for all results.
    num_steps : int
        PGD optimisation iterations per campaign (default: 200).
    bands : list of str
        Frequency bands to optimise. Default: ['low', 'high', 'full'].
    epsilons : list of float
        Perturbation budgets for epsilon sweep. Default: [0.1, 0.2, 0.3, 0.5].
    device : str
        'cuda' or 'cpu'.
    batch_size : int
        Batch size for optimisation (default: 1, RTX 4050 6 GB VRAM).
    skip_optim : bool
        If True, skip optimisation and load existing patches from results_dir.
    skip_eval : bool
        If True, skip evaluation and load existing JSON results.
    skip_weather : bool
        If True, skip weather ablation.
    skip_distance : bool
        If True, skip distance ablation.
    """
    if bands is None:
        bands = ["low", "high", "full"]
    if epsilons is None:
        epsilons = [0.1, 0.2, 0.3, 0.5]

    os.makedirs(results_dir, exist_ok=True)
    start_total = time.time()

    print("+" + "=" * 70 + "+")
    print("|  ViT vs CNN Adversarial Robustness - CARLA Experiment Pipeline      |")
    print("+" + "=" * 70 + "+")
    print(f"  Data:     {data_dir}")
    print(f"  Results:  {results_dir}")
    print(f"  Steps:    {num_steps}")
    print(f"  Bands:    {bands}")
    print(f"  Epsilons: {epsilons}")
    print(f"  Device:   {device}")
    print()

    # -- 1. Load Data ----------------------------------------------------
    print("=" * 72)
    print("  Stage 0: Loading Data & Models")
    print("=" * 72)

    dataset, dataloader = _build_carla_dataloader(data_dir, split="val", batch_size=batch_size)

    # Collect all images + bboxes for optimisation (full val set)
    all_images = []
    all_bboxes = []
    for images_batch, targets_batch in dataloader:
        all_images.append(images_batch)
        for t in targets_batch:
            all_bboxes.append(t["boxes"])

    # Stack into optimization batches
    optim_images = torch.cat(all_images, dim=0)

    # Sort by bounding-box area (largest first) so the optimizer always
    # works with close-distance images where signs are big enough for the
    # patch applier to overlay a meaningful patch and propagate gradients.
    bbox_areas = []
    for b in all_bboxes:
        if b.numel() > 0:
            widths = b[:, 2] - b[:, 0]
            heights = b[:, 3] - b[:, 1]
            bbox_areas.append((widths * heights).max().item())
        else:
            bbox_areas.append(0.0)
    sorted_indices = sorted(range(len(bbox_areas)), key=lambda i: bbox_areas[i], reverse=True)
    optim_images = optim_images[sorted_indices]
    all_bboxes = [all_bboxes[i] for i in sorted_indices]

    print(f"  Optimisation images: {optim_images.shape}")
    print(f"  Total bounding boxes: {sum(b.shape[0] for b in all_bboxes)}")
    print(f"  Largest bbox area: {bbox_areas[sorted_indices[0]]:.0f} px^2, "
          f"Smallest: {bbox_areas[sorted_indices[-1]]:.0f} px^2")

    # -- 2. Load Models --------------------------------------------------
    rcnn_num_classes = 44 if rcnn_checkpoint else 91
    vit_num_classes = 44 if vit_checkpoint else 91

    rcnn = FasterRCNNWrapper(num_classes=rcnn_num_classes, conf_threshold=0.25, device=device, checkpoint_path=rcnn_checkpoint)
    vit = DetrVitWrapper(conf_threshold=0.25, device=device, num_classes=vit_num_classes, checkpoint_path=vit_checkpoint)

    # -- 3. Optimise Patches ---------------------------------------------
    # Dict: {eps_str: {band: patch_tensor}}
    all_patches: Dict[str, Dict[str, torch.Tensor]] = {}

    if not skip_optim:
        for eps in epsilons:
            eps_str = f"{eps:.1f}"
            all_patches[eps_str] = {}

            for band in bands:
                exp_dir = os.path.join(results_dir, "experiments", f"{band}_eps{eps_str}")
                patch_path = os.path.join(exp_dir, f"patch_{band}_eps{eps_str}_best.pt")

                # Check if already computed
                if os.path.exists(patch_path):
                    print(f"\n  [SKIP] Found existing: {patch_path}")
                    all_patches[eps_str][band] = torch.load(patch_path, weights_only=True)
                    continue

                # We optimise in batches to avoid OOM
                # Use the first `batch_size` images per step (the optimizer cycles internally)
                opt_imgs = optim_images[:batch_size]
                opt_bboxes = all_bboxes[:batch_size]

                patch, history = run_optimisation(
                    rcnn=rcnn,
                    vit=vit,
                    images=opt_imgs,
                    bboxes_list=opt_bboxes,
                    band=band,
                    epsilon=eps,
                    num_steps=num_steps,
                    results_dir=results_dir,
                    device=device,
                )
                all_patches[eps_str][band] = patch

                # Force cleanup between campaigns
                torch.cuda.empty_cache()
                gc.collect()
    else:
        # Load existing patches
        print("\n  [SKIP_OPTIM] Loading existing patches...")
        for eps in epsilons:
            eps_str = f"{eps:.1f}"
            all_patches[eps_str] = {}
            for band in bands:
                exp_dir = os.path.join(results_dir, "experiments", f"{band}_eps{eps_str}")
                patch_path = os.path.join(exp_dir, f"patch_{band}_eps{eps_str}_best.pt")
                if os.path.exists(patch_path):
                    all_patches[eps_str][band] = torch.load(patch_path, weights_only=True)
                    print(f"    Loaded: {patch_path}")
                else:
                    print(f"    WARNING: Not found: {patch_path}")

    # -- 4. Evaluate -----------------------------------------------------
    if not skip_eval:
        # Use default epsilon (0.3) patches for the main frequency comparison
        default_eps = "0.3"
        if default_eps in all_patches:
            freq_comparison = evaluate_all_patches(
                rcnn, vit, dataset, all_patches[default_eps], results_dir, device
            )

            # Compute Adversarial Delta per band
            eval_dir = os.path.join(results_dir, "evaluation")
            adv_delta = {}
            for band in bands:
                if band in freq_comparison["rcnn"] and band in freq_comparison["yolos"]:
                    rcnn_mAP = freq_comparison["rcnn"][band]["score"]
                    yolos_mAP = freq_comparison["yolos"][band]["score"]
                    adv_delta[band] = calculate_adversarial_delta(yolos_mAP, rcnn_mAP)

            delta_path = os.path.join(eval_dir, "adversarial_delta.json")
            with open(delta_path, "w") as f:
                json.dump(adv_delta, f, indent=2)
            print(f"  Saved Adversarial Delta: {delta_path}")

        # RSF curves (using full-band patch at epsilon=0.3)
        if default_eps in all_patches and "full" in all_patches[default_eps]:
            compute_rsf_curves(
                rcnn, vit, dataset, all_patches[default_eps]["full"],
                results_dir, device
            )

        # Epsilon sweep
        run_epsilon_sweep(rcnn, vit, dataset, all_patches, results_dir, device)

    # -- 5. Weather Ablation ---------------------------------------------
    if not skip_weather:
        default_eps = "0.3"
        if default_eps in all_patches and "full" in all_patches[default_eps]:
            run_weather_ablation(
                rcnn, vit, all_patches[default_eps]["full"],
                data_dir, results_dir, device
            )
        else:
            print("\n  [SKIP] Weather ablation: no full-band epsilon=0.3 patch available.")
    else:
        print("\n  [SKIP] Weather ablation skipped (--skip_weather)")

    # -- 6. Distance Ablation --------------------------------------------
    if not skip_distance:
        default_eps = "0.3"
        if default_eps in all_patches and "full" in all_patches[default_eps]:
            run_distance_ablation(
                rcnn, vit, all_patches[default_eps]["full"],
                data_dir, results_dir, device
            )
        else:
            print("\n  [SKIP] Distance ablation: no full-band epsilon=0.3 patch available.")
    else:
        print("\n  [SKIP] Distance ablation skipped (--skip_distance)")

    # -- 7. Summary ------------------------------------------------------
    elapsed = time.time() - start_total
    print(f"\n{'+' + '=' * 70 + '+'}")
    print(f"|  CARLA Experiment Complete!                                          |")
    print(f"+{'=' * 70}+")
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print(f"  Results:    {results_dir}")
    print(f"  Patches:    {len(all_patches)} epsilon x {len(bands)} bands = {sum(len(v) for v in all_patches.values())} total")

    # Save experiment config for reproducibility
    config = {
        "data_dir": data_dir,
        "results_dir": results_dir,
        "num_steps": num_steps,
        "bands": bands,
        "epsilons": epsilons,
        "device": device,
        "batch_size": batch_size,
        "dataset": "carla",
        "total_time_minutes": elapsed / 60,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config:     {config_path}")

    return all_patches


# =======================================================================
# CLI Entry Point
# =======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ViT vs CNN Adversarial Robustness - CARLA Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full experiment (200 steps):
    python -m src.experiments.run_carla_experiments --data_dir data/carla --results_dir results/carla

  Quick test (5 steps):
    python -m src.experiments.run_carla_experiments --data_dir data/carla --results_dir results/carla --num_steps 5

  Skip optimisation, re-run evaluation + ablations:
    python -m src.experiments.run_carla_experiments --data_dir data/carla --results_dir results/carla --skip_optim
        """
    )

    parser.add_argument(
        "--data_dir", type=str, default="data/carla",
        help="Path to CARLA data directory (default: data/carla)"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/carla",
        help="Output directory for all results (default: results/carla)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=200,
        help="PGD optimisation steps per campaign (default: 200)"
    )
    parser.add_argument(
        "--bands", type=str, default="low,high,full",
        help="Comma-separated frequency bands (default: low,high,full)"
    )
    parser.add_argument(
        "--epsilons", type=str, default="0.1,0.2,0.3,0.5",
        help="Comma-separated epsilon values for sweep (default: 0.1,0.2,0.3,0.5)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: 'cuda' or 'cpu' (default: cuda)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for optimisation (default: 1, RTX 4050 6 GB VRAM)"
    )
    parser.add_argument(
        "--skip_optim", action="store_true",
        help="Skip optimisation, use existing patch files"
    )
    parser.add_argument(
        "--skip_eval", action="store_true",
        help="Skip evaluation, use existing JSON results"
    )
    parser.add_argument(
        "--skip_weather", action="store_true",
        help="Skip weather ablation"
    )
    parser.add_argument(
        "--skip_distance", action="store_true",
        help="Skip distance ablation"
    )
    parser.add_argument(
        "--rcnn_checkpoint", type=str, default=None,
        help="Path to fine-tuned Faster R-CNN checkpoint (default: None)"
    )
    parser.add_argument(
        "--vit_checkpoint", type=str, default=None,
        help="Path to fine-tuned YOLOS checkpoint (default: None)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    bands = [b.strip() for b in args.bands.split(",")]
    epsilons = [float(e.strip()) for e in args.epsilons.split(",")]

    run_carla_experiment(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        num_steps=args.num_steps,
        bands=bands,
        epsilons=epsilons,
        device=args.device,
        batch_size=args.batch_size,
        skip_optim=args.skip_optim,
        skip_eval=args.skip_eval,
        skip_weather=args.skip_weather,
        skip_distance=args.skip_distance,
        rcnn_checkpoint=args.rcnn_checkpoint,
        vit_checkpoint=args.vit_checkpoint,
    )
