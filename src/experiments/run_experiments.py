"""
run_experiments.py - Full experimental pipeline for ViT vs CNN adversarial robustness.

Orchestrates:
1. Patch optimisation across 3 frequency bands (low / high / full)
2. Epsilon sweep across 4 perturbation budgets (0.1, 0.2, 0.3, 0.5)
3. Evaluation of all patches on both Faster R-CNN and YOLOS-Small
4. RSF curve generation (mAP vs patch area ratio)
5. Adversarial Delta computation (ViT mAP - CNN mAP per band)

All results are saved as structured JSON for downstream aggregation and visualisation.
"""

import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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
# Helper: nuScenes DataLoader that wraps the existing loader
# =======================================================================

def _build_nuscenes_dataloader(data_dir: str, split: str, batch_size: int = 2):
    """
    Build a DataLoader from the nuScenes 2D dataset.
    Returns images (B, 3, 640, 640) and a list of target dicts.
    """
    from src.data_prep.nuscenes_loader import NuScenes2DDataset

    dataset = NuScenes2DDataset(data_root=data_dir, split=split)
    print(f"[Data] Loaded nuScenes {split}: {len(dataset)} samples")

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
        patch_ratio=0.4,
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
    patch_ratio: float = 0.4,
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
                preds.append({
                    "box": b.cpu().tolist(),
                    "score": s.cpu().item(),
                    "label": l.cpu().item(),
                })
            predictions_list.append(preds)

            gts = []
            for b, l in zip(bboxes.cpu(), labels.cpu()):
                gts.append({"box": b.tolist(), "label": l.item()})
            ground_truths_list.append(gts)

            max_conf = max([p["score"] for p in preds], default=0.0)
            per_image_results.append({
                "image_idx": idx,
                "num_detections": len(preds),
                "max_confidence": max_conf,
                "ground_truth_count": len(gts),
            })

    mAP = calculate_map(predictions_list, ground_truths_list)
    avg_det = np.mean([r["num_detections"] for r in per_image_results])
    avg_conf = np.mean([r["max_confidence"] for r in per_image_results])
    print(f"  [{model_name}] mAP={mAP:.4f} | Avg det={avg_det:.1f} | Avg conf={avg_conf:.4f}")

    return mAP, per_image_results


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
        mAP, _ = evaluate_single_patch(model, dataset, None, name, device)
        comparison[key]["clean"] = {"mAP": mAP}

    # Patched evaluations
    for patch_key, patch_tensor in patches.items():
        for model, name, key in [(rcnn, "Faster R-CNN", "rcnn"), (vit, "YOLOS-Small", "yolos")]:
            mAP, results = evaluate_single_patch(model, dataset, patch_tensor, f"{name}/{patch_key}", device)
            clean_mAP = comparison[key]["clean"]["mAP"]
            comparison[key][patch_key] = {
                "mAP": mAP,
                "mAP_drop": clean_mAP - mAP,
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

    rsf_data = {"rcnn": {"ratios": area_ratios, "mAPs": []},
                "yolos": {"ratios": area_ratios, "mAPs": []}}

    for model, name, key in [(rcnn, "Faster R-CNN", "rcnn"), (vit, "YOLOS-Small", "yolos")]:
        mAPs = []
        for ratio in area_ratios:
            if ratio == 0.0:
                mAP, _ = evaluate_single_patch(model, dataset, None, f"{name}/ratio=0.0", device)
            else:
                mAP, _ = evaluate_single_patch(
                    model, dataset, patch, f"{name}/ratio={ratio:.2f}", device,
                    patch_ratio=ratio
                )
            mAPs.append(mAP)

        rsf_data[key]["mAPs"] = mAPs
        rsf = calculate_robustness_sensitivity_factor(area_ratios, mAPs)
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
            rcnn_mAP, _ = evaluate_single_patch(
                rcnn, dataset, patch_tensor, f"RCNN/epsilon={eps_str}/{band}", device
            )
            yolos_mAP, _ = evaluate_single_patch(
                vit, dataset, patch_tensor, f"YOLOS/epsilon={eps_str}/{band}", device
            )
            sweep_results[eps_str][band] = {
                "rcnn_mAP": rcnn_mAP,
                "yolos_mAP": yolos_mAP,
                "adversarial_delta": calculate_adversarial_delta(yolos_mAP, rcnn_mAP),
            }

    sweep_path = os.path.join(eval_dir, "epsilon_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n  Saved: {sweep_path}")

    return sweep_results


# =======================================================================
# Main Orchestrator
# =======================================================================

def run_full_experiment(
    data_dir: str,
    results_dir: str,
    num_steps: int = 300,
    bands: Optional[List[str]] = None,
    epsilons: Optional[List[float]] = None,
    device: str = "cuda",
    batch_size: int = 2,
    skip_optim: bool = False,
    skip_eval: bool = False,
):
    """
    Run the complete experimental pipeline:
    1. Load data + models
    2. Optimise patches (3 bands x 4 epsilons)
    3. Evaluate all patches on both models
    4. Compute RSF curves
    5. Run epsilon sweep evaluation
    6. Save all results

    Parameters
    ----------
    data_dir : str
        Path to nuScenes data directory.
    results_dir : str
        Path to output directory for all results.
    num_steps : int
        PGD optimisation iterations per campaign (default: 300).
    bands : list of str
        Frequency bands to optimise. Default: ['low', 'high', 'full'].
    epsilons : list of float
        Perturbation budgets for epsilon sweep. Default: [0.1, 0.2, 0.3, 0.5].
    device : str
        'cuda' or 'cpu'.
    batch_size : int
        Batch size for optimisation (default: 2).
    skip_optim : bool
        If True, skip optimisation and load existing patches from results_dir.
    skip_eval : bool
        If True, skip evaluation and load existing JSON results.
    """
    if bands is None:
        bands = ["low", "high", "full"]
    if epsilons is None:
        epsilons = [0.1, 0.2, 0.3, 0.5]

    os.makedirs(results_dir, exist_ok=True)
    start_total = time.time()

    print("+" + "=" * 70 + "+")
    print("|  ViT vs CNN Adversarial Robustness - Full Experiment Pipeline       |")
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

    dataset, dataloader = _build_nuscenes_dataloader(data_dir, split="val", batch_size=batch_size)

    # Collect all images + bboxes for optimisation (full val set)
    all_images = []
    all_bboxes = []
    for images_batch, targets_batch in dataloader:
        all_images.append(images_batch)
        for t in targets_batch:
            all_bboxes.append(t["boxes"])

    # Stack into optimization batches
    optim_images = torch.cat(all_images, dim=0)
    print(f"  Optimisation images: {optim_images.shape}")
    print(f"  Total bounding boxes: {sum(b.shape[0] for b in all_bboxes)}")

    # -- 2. Load Models --------------------------------------------------
    rcnn = FasterRCNNWrapper(conf_threshold=0.25, device=device)
    vit = DetrVitWrapper(conf_threshold=0.25, device=device)

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
                    rcnn_mAP = freq_comparison["rcnn"][band]["mAP"]
                    yolos_mAP = freq_comparison["yolos"][band]["mAP"]
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

    # -- 5. Summary ------------------------------------------------------
    elapsed = time.time() - start_total
    print(f"\n{'+' + '=' * 70 + '+'}")
    print(f"|  Experiment Complete!                                                |")
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
        "total_time_minutes": elapsed / 60,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config:     {config_path}")

    return all_patches


# =======================================================================
# Smoke Test
# =======================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("  run_experiments.py - Smoke Test (5 steps, mock data)")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Quick validation: just test the evaluation pipeline with mock model
    class _MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(1, 1)
        def forward(self, x):
            return [{
                "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], device=x.device),
                "scores": torch.tensor([0.85], device=x.device),
                "labels": torch.tensor([0], device=x.device, dtype=torch.long),
            } for _ in range(x.shape[0])]
        def parameters(self):
            return self.dummy.parameters()

    class _MockDataset:
        def __len__(self):
            return 4
        def __getitem__(self, idx):
            return {
                "image": torch.rand(3, 256, 256),
                "bboxes": torch.tensor([[80.0, 80.0, 220.0, 220.0]]),
                "labels": torch.tensor([0]),
            }

    mock_model = _MockModel()
    mock_dataset = _MockDataset()
    mock_patch = torch.rand(3, 64, 64)

    mAP, results = evaluate_single_patch(mock_model, mock_dataset, mock_patch, "MockModel", "cpu")
    print(f"\n  Mock evaluation mAP: {mAP:.4f}")
    print(f"  Per-image results: {len(results)} entries")

    clean_mAP, _ = evaluate_single_patch(mock_model, mock_dataset, None, "MockModel-Clean", "cpu")
    print(f"  Clean mAP: {clean_mAP:.4f}")

    print("\n  ✅ Smoke test passed!")
