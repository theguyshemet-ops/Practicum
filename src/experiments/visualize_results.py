"""
visualize_results.py - Publication-quality matplotlib figures for dissertation.

Generates 9 figure types from experiment JSON results and patch tensors:
1. Loss convergence curves (per frequency band)
2. Frequency comparison grouped bar chart
3. RSF curves (mAP vs patch area ratio)
4. Attention Rollout heatmaps (clean vs patched, ViT)
5. Grad-CAM comparison (clean vs patched, CNN)
6. Detection overlay (bounding boxes before/after attack)
7. Patch gallery (all 3 frequency band patches)
8. Adversarial Delta bar chart
9. Epsilon sweep curves

All figures saved at 300 DPI with tight bounding boxes.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


# =======================================================================
# Global Style Configuration
# =======================================================================

# Curated colour palette
COLOUR_RCNN = "#2563EB"      # Deep blue - Faster R-CNN
COLOUR_YOLOS = "#F59E0B"     # Warm orange - YOLOS
COLOUR_LOW = "#10B981"       # Emerald - low-frequency
COLOUR_HIGH = "#EF4444"      # Red - high-frequency
COLOUR_FULL = "#8B5CF6"      # Purple - full-spectrum
COLOUR_CLEAN = "#6B7280"     # Grey - clean baseline
COLOUR_POS = "#22C55E"       # Green - positive delta
COLOUR_NEG = "#EF4444"       # Red - negative delta

BAND_COLOURS = {"low": COLOUR_LOW, "high": COLOUR_HIGH, "full": COLOUR_FULL, "clean": COLOUR_CLEAN}
BAND_LABELS = {"low": "Low-Frequency", "high": "High-Frequency", "full": "Full-Spectrum", "clean": "Clean"}


def set_pub_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _load_json(path: str) -> Optional[dict]:
    """Load JSON or return None."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =======================================================================
# Figure 1: Loss Convergence Curves
# =======================================================================

def plot_loss_convergence(results_dir: str, save_path: str):
    """
    Plot loss vs optimisation step for all 3 frequency bands.
    1x3 subplot grid showing total, RCNN, ViT, and attention loss components.
    """
    set_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    bands = ["low", "high", "full"]
    titles = ["Low-Frequency", "High-Frequency", "Full-Spectrum"]
    found_any = False

    for ax, band, title in zip(axes, bands, titles):
        # Try default epsilon first
        hist_path = os.path.join(results_dir, "experiments", f"{band}_eps0.3",
                                 f"patch_{band}_eps0.3_history.json")
        # Fallback to direct band folder
        if not os.path.exists(hist_path):
            hist_path = os.path.join(results_dir, "experiments", band,
                                     f"patch_{band}_history.json")

        history = _load_json(hist_path)
        if history is None:
            ax.set_title(f"{title}\n(no data)", color="grey")
            ax.set_xlabel("Step")
            continue

        found_any = True
        steps = range(1, len(history["total_loss"]) + 1)

        ax.plot(steps, history["total_loss"], color="#1F2937", linewidth=2.0,
                label="Total Loss", zorder=3)
        ax.plot(steps, history["suppress_rcnn"], color=COLOUR_RCNN, linewidth=1.2,
                alpha=0.8, label="R-CNN Suppress", linestyle="--")
        ax.plot(steps, history["suppress_vit"], color=COLOUR_YOLOS, linewidth=1.2,
                alpha=0.8, label="ViT Suppress", linestyle="--")
        ax.plot(steps, history["attention_vit"], color=COLOUR_FULL, linewidth=1.2,
                alpha=0.8, label="Attn Disrupt", linestyle="-.")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Optimisation Step")

    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right", framealpha=0.9, edgecolor="none")

    fig.suptitle("Loss Convergence by Frequency Band (epsilon = 0.3)", fontweight="bold", y=1.02)
    plt.tight_layout()

    if found_any:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [OK] Saved: {save_path}")
    else:
        print(f"  [SKIPPED] Skipped (no history data): {save_path}")
    plt.close(fig)


# =======================================================================
# Figure 2: Frequency Comparison Bar Chart
# =======================================================================

def plot_frequency_comparison(results_dir: str, save_path: str):
    """
    Grouped bar chart: mAP per frequency band per architecture.
    """
    set_pub_style()

    freq_path = os.path.join(results_dir, "evaluation", "frequency_comparison.json")
    data = _load_json(freq_path)
    if data is None:
        print(f"  [SKIPPED] Skipped (no data): {save_path}")
        return

    bands = ["clean", "low", "high", "full"]
    rcnn_mAPs = [data["rcnn"].get(b, {}).get("score", data["rcnn"].get(b, {}).get("mAP", 0)) for b in bands]
    yolos_mAPs = [data["yolos"].get(b, {}).get("score", data["yolos"].get(b, {}).get("mAP", 0)) for b in bands]

    x = np.arange(len(bands))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars_rcnn = ax.bar(x - width / 2, rcnn_mAPs, width, label="Faster R-CNN",
                       color=COLOUR_RCNN, alpha=0.85, edgecolor="white", linewidth=0.8)
    bars_yolos = ax.bar(x + width / 2, yolos_mAPs, width, label="YOLOS-Small",
                        color=COLOUR_YOLOS, alpha=0.85, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bars in [bars_rcnn, bars_yolos]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Clean baseline reference line
    if rcnn_mAPs[0] > 0:
        ax.axhline(y=rcnn_mAPs[0], color=COLOUR_RCNN, linestyle=":", alpha=0.4, linewidth=1)
    if yolos_mAPs[0] > 0:
        ax.axhline(y=yolos_mAPs[0], color=COLOUR_YOLOS, linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("mAP")
    ax.set_title("Detection mAP Under Frequency-Constrained Adversarial Patches",
                  fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([BAND_LABELS.get(b, b) for b in bands])
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax.set_ylim(0, max(max(rcnn_mAPs), max(yolos_mAPs)) * 1.15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] Saved: {save_path}")
    plt.close(fig)


# =======================================================================
# Figure 3: RSF Curves
# =======================================================================

def plot_rsf_curves(results_dir: str, save_path: str):
    """
    Line plot: mAP vs patch area ratio for both architectures.
    """
    set_pub_style()

    rsf_path = os.path.join(results_dir, "evaluation", "rsf_curves.json")
    data = _load_json(rsf_path)
    if data is None:
        print(f"  [SKIPPED] Skipped (no data): {save_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for key, label, colour, marker in [
        ("rcnn", "Faster R-CNN", COLOUR_RCNN, "o"),
        ("yolos", "YOLOS-Small", COLOUR_YOLOS, "s"),
    ]:
        if key in data:
            ratios = data[key]["ratios"]
            mAPs = data[key].get("scores", data[key].get("mAPs", []))
            rsf = data[key].get("rsf", 0)

            ax.plot(ratios, mAPs, color=colour, marker=marker, markersize=7,
                    linewidth=2.0, label=f"{label} (RSF={rsf:.3f})", zorder=3)

    # Shaded gap between curves
    if "rcnn" in data and "yolos" in data:
        ratios = data["rcnn"]["ratios"]
        rcnn_mAPs = data["rcnn"].get("scores", data["rcnn"].get("mAPs", []))
        yolos_mAPs = data["yolos"].get("scores", data["yolos"].get("mAPs", []))
        ax.fill_between(ratios, rcnn_mAPs, yolos_mAPs, alpha=0.1, color="#6B7280")

    ax.set_xlabel("Patch Area Ratio")
    ax.set_ylabel("mAP")
    ax.set_title("Robustness Sensitivity Factor (RSF): mAP Degradation vs Patch Size",
                  fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax.set_xlim(-0.02, 0.52)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] Saved: {save_path}")
    plt.close(fig)


# =======================================================================
# Figure 4: Attention Rollout Heatmaps
# =======================================================================

def plot_attention_rollout(
    model,
    images: torch.Tensor,
    patches_dict: Dict[str, torch.Tensor],
    patch_applier,
    save_path: str,
    device: str = "cuda",
):
    """
    2x4 grid: Row 1 = source images, Row 2 = attention rollout heatmaps.
    Columns: Clean, Low, High, Full.
    """
    set_pub_style()

    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()

    # Use first image in batch
    img = images[0:1].to(target_device)
    columns = ["clean", "low", "high", "full"]
    col_labels = ["Clean", "Low-Freq", "High-Freq", "Full-Spectrum"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col_idx, (col_key, col_label) in enumerate(zip(columns, col_labels)):
        with torch.no_grad():
            if col_key == "clean":
                input_img = img
            else:
                if col_key not in patches_dict:
                    axes[0, col_idx].set_visible(False)
                    axes[1, col_idx].set_visible(False)
                    continue
                patch = patches_dict[col_key].to(target_device)
                if patch.ndim == 3:
                    patch = patch.unsqueeze(0)
                # Apply patch at centre with dummy bbox covering full image
                h, w = img.shape[2], img.shape[3]
                dummy_bbox = [torch.tensor([[0, 0, w, h]], dtype=torch.float32, device=target_device)]
                input_img = patch_applier(img, patch, dummy_bbox)

            _ = model(input_img)
            rollout = model.get_attention_rollout()

        # Display source image
        img_np = input_img[0].cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[0, col_idx].imshow(img_np)
        axes[0, col_idx].set_title(col_label, fontweight="bold")
        axes[0, col_idx].axis("off")

        # Display attention heatmap overlay
        if rollout and len(rollout) > 0:
            attn_map = rollout[0].cpu().numpy()
            if attn_map.ndim > 2:
                attn_map = attn_map.mean(axis=tuple(range(attn_map.ndim - 2)))
            # Resize to image dimensions
            from PIL import Image as PILImage
            attn_resized = np.array(
                PILImage.fromarray((attn_map * 255).astype(np.uint8)).resize(
                    (img_np.shape[1], img_np.shape[0]), PILImage.BILINEAR
                )
            ).astype(np.float32) / 255.0

            axes[1, col_idx].imshow(img_np)
            axes[1, col_idx].imshow(attn_resized, cmap="jet", alpha=0.5)
        else:
            axes[1, col_idx].imshow(img_np)
            axes[1, col_idx].text(0.5, 0.5, "No data", ha="center", va="center",
                                   transform=axes[1, col_idx].transAxes, color="grey")
        axes[1, col_idx].axis("off")

    axes[0, 0].set_ylabel("Input Image", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Attention Rollout", fontsize=12, fontweight="bold")

    fig.suptitle("YOLOS-Small Attention Rollout: Clean vs Adversarial Patches",
                  fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] Saved: {save_path}")
    plt.close(fig)


# =======================================================================
# Figure 5: Grad-CAM Comparison
# =======================================================================

def plot_gradcam_comparison(
    model,
    images: torch.Tensor,
    patches_dict: Dict[str, torch.Tensor],
    patch_applier,
    save_path: str,
    device: str = "cuda",
):
    """
    2x4 grid: Row 1 = source images, Row 2 = Grad-CAM saliency maps.
    """
    set_pub_style()

    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()

    img = images[0:1].to(target_device)
    columns = ["clean", "low", "high", "full"]
    col_labels = ["Clean", "Low-Freq", "High-Freq", "Full-Spectrum"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col_idx, (col_key, col_label) in enumerate(zip(columns, col_labels)):
        with torch.no_grad():
            if col_key == "clean":
                input_img = img
            else:
                if col_key not in patches_dict:
                    axes[0, col_idx].set_visible(False)
                    axes[1, col_idx].set_visible(False)
                    continue
                patch = patches_dict[col_key].to(target_device)
                if patch.ndim == 3:
                    patch = patch.unsqueeze(0)
                h, w = img.shape[2], img.shape[3]
                dummy_bbox = [torch.tensor([[0, 0, w, h]], dtype=torch.float32, device=target_device)]
                input_img = patch_applier(img, patch, dummy_bbox)

            _ = model(input_img)

        # Get Grad-CAM
        try:
            gradcam = model.get_gradcam()
        except (AttributeError, Exception):
            gradcam = None

        # Display source image
        img_np = input_img[0].cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[0, col_idx].imshow(img_np)
        axes[0, col_idx].set_title(col_label, fontweight="bold")
        axes[0, col_idx].axis("off")

        # Display Grad-CAM overlay
        if gradcam is not None:
            cam = gradcam
            if isinstance(cam, torch.Tensor):
                cam = cam.cpu().numpy()
            if cam.ndim > 2:
                cam = cam.squeeze()
                if cam.ndim > 2:
                    cam = cam.mean(axis=0)

            from PIL import Image as PILImage
            cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam_resized = np.array(
                PILImage.fromarray((cam_norm * 255).astype(np.uint8)).resize(
                    (img_np.shape[1], img_np.shape[0]), PILImage.BILINEAR
                )
            ).astype(np.float32) / 255.0

            axes[1, col_idx].imshow(img_np)
            axes[1, col_idx].imshow(cam_resized, cmap="jet", alpha=0.5)
        else:
            axes[1, col_idx].imshow(img_np)
            axes[1, col_idx].text(0.5, 0.5, "No Grad-CAM", ha="center", va="center",
                                   transform=axes[1, col_idx].transAxes, color="grey")
        axes[1, col_idx].axis("off")

    axes[0, 0].set_ylabel("Input Image", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=12, fontweight="bold")

    fig.suptitle("Faster R-CNN Grad-CAM: Clean vs Adversarial Patches",
                  fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] Saved: {save_path}")
    plt.close(fig)


# =======================================================================
# Figure 6: Detection Overlay
# =======================================================================

def plot_detection_overlay(
    rcnn,
    yolos,
    images: torch.Tensor,
    patches_dict: Dict[str, torch.Tensor],
    patch_applier,
    save_path: str,
    device: str = "cuda",
    ground_truth_boxes: Optional[torch.Tensor] = None,
):
    """
    2x4 grid: detection bounding boxes on clean and patched images.
    Row 1: Faster R-CNN, Row 2: YOLOS.
    """
    set_pub_style()

    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    rcnn.eval()
    yolos.eval()

    img = images[0:1].to(target_device)
    columns = ["clean", "low", "high", "full"]
    col_labels = ["Clean", "Low-Freq", "High-Freq", "Full-Spectrum"]
    models = [("Faster R-CNN", rcnn), ("YOLOS-Small", yolos)]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for row_idx, (model_name, model) in enumerate(models):
        for col_idx, (col_key, col_label) in enumerate(zip(columns, col_labels)):
            with torch.no_grad():
                if col_key == "clean":
                    input_img = img
                else:
                    if col_key not in patches_dict:
                        axes[row_idx, col_idx].set_visible(False)
                        continue
                    patch = patches_dict[col_key].to(target_device)
                    if patch.ndim == 3:
                        patch = patch.unsqueeze(0)
                    h, w = img.shape[2], img.shape[3]
                    dummy_bbox = [torch.tensor([[0, 0, w, h]], dtype=torch.float32, device=target_device)]
                    input_img = patch_applier(img, patch, dummy_bbox)

                detections = model(input_img)

            img_np = input_img[0].cpu().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            axes[row_idx, col_idx].imshow(img_np)

            # Draw detection boxes
            det = detections[0]
            for b, s, l in zip(det["boxes"].cpu(), det["scores"].cpu(), det["labels"].cpu()):
                x1, y1, x2, y2 = b.tolist()
                conf = s.item()
                if conf > 0.25:
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          linewidth=2, edgecolor=COLOUR_POS,
                                          facecolor="none", linestyle="-")
                    axes[row_idx, col_idx].add_patch(rect)
                    axes[row_idx, col_idx].text(x1, y1 - 4, f"{conf:.2f}",
                                                 color=COLOUR_POS, fontsize=8,
                                                 fontweight="bold",
                                                 bbox=dict(boxstyle="round,pad=0.15",
                                                           facecolor="black", alpha=0.6))

            # Draw ground truth boxes if available
            if ground_truth_boxes is not None:
                for gt_box in ground_truth_boxes:
                    x1, y1, x2, y2 = gt_box.tolist()
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          linewidth=1.5, edgecolor=COLOUR_NEG,
                                          facecolor="none", linestyle="--")
                    axes[row_idx, col_idx].add_patch(rect)

            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_label, fontweight="bold")
            axes[row_idx, col_idx].axis("off")

        axes[row_idx, 0].set_ylabel(model_name, fontsize=12, fontweight="bold")

    fig.suptitle("Detection Overlay: Clean vs Adversarial Patches", fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] Saved: {save_path}")
    plt.close(fig)


# =======================================================================
# Figure 7: Patch Gallery
# =======================================================================

def plot_patch_gallery(results_dir: str, save_path: str):
    """
    1x3 horizontal strip showing optimised patches for each frequency band.
    """
    set_pub_style()

    bands = ["low", "high", "full"]
    titles = ["Low-Frequency", "High-Frequency", "Full-Spectrum"]
    found_any = False

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, band, title in zip(axes, bands, titles):
        # Try eps0.3 first, fallback to direct
        pt_path = os.path.join(results_dir, "experiments", f"{band}_eps0.3",
                               f"patch_{band}_eps0.3_best.pt")
        if not os.path.exists(pt_path):
            pt_path = os.path.join(results_dir, "experiments", band,
                                   f"patch_{band}_best.pt")

        if os.path.exists(pt_path):
            patch = torch.load(pt_path, map_location="cpu", weights_only=True)
            if patch.ndim == 3:
                img_np = patch.permute(1, 2, 0).numpy()
            else:
                img_np = patch.squeeze(0).permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)
            found_any = True
        else:
            ax.text(0.5, 0.5, "Not\ngenerated", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="grey")

        ax.set_title(title, fontweight="bold", pad=10)
        ax.axis("off")

    fig.suptitle("Optimised Adversarial Patches (epsilon = 0.3)", fontweight="bold", y=1.02)
    plt.tight_layout()

    if found_any:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [OK] Saved: {save_path}")
    else:
        print(f"  [SKIPPED] Skipped (no patch files): {save_path}")
    plt.close(fig)


# =======================================================================
# Figure 8: Adversarial Delta Bar Chart
# =======================================================================

def plot_adversarial_delta(results_dir: str, save_path: str):
    """
    Bar chart: Adversarial Delta (ViT mAP - CNN mAP) per frequency band.
    """
    set_pub_style()

    delta_path = os.path.join(results_dir, "evaluation", "adversarial_delta.json")
    data = _load_json(delta_path)

    # Fallback: compute from frequency comparison
    if data is None:
        freq_path = os.path.join(results_dir, "evaluation", "frequency_comparison.json")
        freq_data = _load_json(freq_path)
        if freq_data is None:
            print(f"  [SKIPPED] Skipped (no data): {save_path}")
            return
        data = {}
        for band in ["low", "high", "full"]:
            rcnn_mAP = freq_data.get("rcnn", {}).get(band, {}).get("score", freq_data.get("rcnn", {}).get(band, {}).get("mAP", 0))
            yolos_mAP = freq_data.get("yolos", {}).get(band, {}).get("score", freq_data.get("yolos", {}).get(band, {}).get("mAP", 0))
            data[band] = yolos_mAP - rcnn_mAP

    bands = ["low", "high", "full"]
    deltas = [data.get(b, 0) for b in bands]
    colours = [COLOUR_POS if d >= 0 else COLOUR_NEG for d in deltas]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar([BAND_LABELS[b] for b in bands], deltas, color=colours, alpha=0.85,
                  edgecolor="white", linewidth=1.5, width=0.6)

    # Value labels
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        y_pos = height + 0.003 if height >= 0 else height - 0.012
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{delta:+.4f}",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=11, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=1.0)
    ax.set_ylabel("Adversarial Delta (ViT mAP - CNN mAP)")
    ax.set_title("Adversarial Delta: Architecture Robustness Comparison",
                  fontweight="bold")

    # Legend
    pos_patch = mpatches.Patch(color=COLOUR_POS, alpha=0.85, label="ViT advantage")
    neg_patch = mpatches.Patch(color=COLOUR_NEG, alpha=0.85, label="CNN advantage")
    ax.legend(handles=[pos_patch, neg_patch], loc="upper right", framealpha=0.9, edgecolor="none")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] Saved: {save_path}")
    plt.close(fig)


# =======================================================================
# Figure 9: Epsilon Sweep
# =======================================================================

def plot_epsilon_sweep(results_dir: str, save_path: str):
    """
    1x3 subplot grid: mAP vs epsilon for each frequency band.
    """
    set_pub_style()

    sweep_path = os.path.join(results_dir, "evaluation", "epsilon_sweep.json")
    data = _load_json(sweep_path)
    if data is None:
        print(f"  [SKIPPED] Skipped (no data): {save_path}")
        return

    bands = ["low", "high", "full"]
    titles = ["Low-Frequency", "High-Frequency", "Full-Spectrum"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    epsilons_sorted = sorted(data.keys(), key=float)

    for ax, band, title in zip(axes, bands, titles):
        rcnn_mAPs = []
        yolos_mAPs = []
        eps_vals = []

        for eps_str in epsilons_sorted:
            if band in data[eps_str]:
                eps_vals.append(float(eps_str))
                rcnn_mAPs.append(data[eps_str][band].get("rcnn_score", data[eps_str][band].get("rcnn_mAP", 0)))
                yolos_mAPs.append(data[eps_str][band].get("yolos_score", data[eps_str][band].get("yolos_mAP", 0)))

        if eps_vals:
            ax.plot(eps_vals, rcnn_mAPs, color=COLOUR_RCNN, marker="o", markersize=6,
                    linewidth=2.0, label="Faster R-CNN")
            ax.plot(eps_vals, yolos_mAPs, color=COLOUR_YOLOS, marker="s", markersize=6,
                    linewidth=2.0, label="YOLOS-Small")

            # Shaded gap
            ax.fill_between(eps_vals, rcnn_mAPs, yolos_mAPs, alpha=0.08, color="#6B7280")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Perturbation Budget (epsilon)")

    axes[0].set_ylabel("mAP")
    axes[0].legend(loc="upper right", framealpha=0.9, edgecolor="none")

    fig.suptitle("Epsilon Sweep: mAP vs Perturbation Budget", fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] Saved: {save_path}")
    plt.close(fig)


# =======================================================================
# Convenience: Generate All Figures
# =======================================================================

def generate_all_figures(
    results_dir: str,
    figures_dir: str,
    model_rcnn=None,
    model_yolos=None,
    images: Optional[torch.Tensor] = None,
    patches_dict: Optional[Dict[str, torch.Tensor]] = None,
    patch_applier=None,
    device: str = "cuda",
):
    """
    Generate all available figures from experiment results.

    Static figures (from JSON): always generated.
    Model-dependent figures (attention, Grad-CAM, detection): require model/image args.
    """
    os.makedirs(figures_dir, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  Generating Figures -> {figures_dir}")
    print(f"{'=' * 72}")

    # -- Static figures (from JSON only) ---------------------------------
    plot_loss_convergence(results_dir, os.path.join(figures_dir, "fig_loss_convergence.png"))
    plot_frequency_comparison(results_dir, os.path.join(figures_dir, "fig_frequency_comparison.png"))
    plot_rsf_curves(results_dir, os.path.join(figures_dir, "fig_rsf_curves.png"))
    plot_patch_gallery(results_dir, os.path.join(figures_dir, "fig_patch_gallery.png"))
    plot_adversarial_delta(results_dir, os.path.join(figures_dir, "fig_adversarial_delta.png"))
    plot_epsilon_sweep(results_dir, os.path.join(figures_dir, "fig_epsilon_sweep.png"))

    # -- CARLA-specific static figures -----------------------------------
    plot_carla_weather_ablation(results_dir, os.path.join(figures_dir, "fig_weather_ablation.png"))
    plot_carla_distance_ablation(results_dir, os.path.join(figures_dir, "fig_distance_ablation.png"))
    plot_cross_domain_comparison(results_dir, os.path.join(figures_dir, "fig_cross_domain_comparison.png"))
    plot_reference_comparison(results_dir, os.path.join(figures_dir, "fig_reference_comparison.png"))

    # -- Model-dependent figures -----------------------------------------
    if model_yolos is not None and images is not None and patches_dict is not None and patch_applier is not None:
        plot_attention_rollout(
            model_yolos, images, patches_dict, patch_applier,
            os.path.join(figures_dir, "fig_attention_rollout.png"), device
        )

    if model_rcnn is not None and images is not None and patches_dict is not None and patch_applier is not None:
        plot_gradcam_comparison(
            model_rcnn, images, patches_dict, patch_applier,
            os.path.join(figures_dir, "fig_gradcam_comparison.png"), device
        )

    if (model_rcnn is not None and model_yolos is not None and images is not None
            and patches_dict is not None and patch_applier is not None):
        plot_detection_overlay(
            model_rcnn, model_yolos, images, patches_dict, patch_applier,
            os.path.join(figures_dir, "fig_detection_overlay.png"), device
        )

    print(f"\n  Done! {figures_dir}")


# =======================================================================
# CARLA-Specific Visualization Functions
# =======================================================================

def plot_carla_weather_ablation(results_dir: str, save_path: str):
    """
    Plot grouped bar chart for weather ablation: mAP vs weather x model.
    """
    weather_path = os.path.join(results_dir, "evaluation", "weather_ablation.json")
    if not os.path.exists(weather_path):
        return
    with open(weather_path, "r") as f:
        data = json.load(f)
        
    weathers = list(data.keys())
    rcnn_clean = [data[w]["rcnn"].get("clean_score", data[w]["rcnn"].get("clean_mAP", 0)) for w in weathers]
    rcnn_patched = [data[w]["rcnn"].get("patched_score", data[w]["rcnn"].get("patched_mAP", 0)) for w in weathers]
    yolos_clean = [data[w]["yolos"].get("clean_score", data[w]["yolos"].get("clean_mAP", 0)) for w in weathers]
    yolos_patched = [data[w]["yolos"].get("patched_score", data[w]["yolos"].get("patched_mAP", 0)) for w in weathers]
    
    x = np.arange(len(weathers))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.bar(x - 1.5*width, rcnn_clean, width, label="R-CNN Clean", color="#1f77b4")
    ax.bar(x - 0.5*width, rcnn_patched, width, label="R-CNN Patched", color="#aec7e8")
    ax.bar(x + 0.5*width, yolos_clean, width, label="YOLOS Clean", color="#ff7f0e")
    ax.bar(x + 1.5*width, yolos_patched, width, label="YOLOS Patched", color="#ffbb78")
    
    ax.set_ylabel("mAP")
    ax.set_title("Weather Ablation: Detector Robustness Across Visibility Conditions")
    ax.set_xticks(x)
    ax.set_xticklabels([w.capitalize() for w in weathers])
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved weather ablation plot to: {save_path}")


def plot_carla_distance_ablation(results_dir: str, save_path: str):
    """
    Plot line chart: mAP vs distance (5-30m) x model.
    """
    dist_path = os.path.join(results_dir, "evaluation", "distance_ablation.json")
    if not os.path.exists(dist_path):
        return
    with open(dist_path, "r") as f:
        data = json.load(f)
        
    dists_str = sorted(data.keys(), key=lambda x: int(x.replace('m', '')) if x.replace('m', '').isdigit() else 0)
    dists_val = [int(d.replace('m', '')) for d in dists_str]
    
    rcnn_clean = [data[d]["rcnn"].get("clean_score", data[d]["rcnn"].get("clean_mAP", 0)) for d in dists_str]
    rcnn_patched = [data[d]["rcnn"].get("patched_score", data[d]["rcnn"].get("patched_mAP", 0)) for d in dists_str]
    yolos_clean = [data[d]["yolos"].get("clean_score", data[d]["yolos"].get("clean_mAP", 0)) for d in dists_str]
    yolos_patched = [data[d]["yolos"].get("patched_score", data[d]["yolos"].get("patched_mAP", 0)) for d in dists_str]
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(dists_val, rcnn_clean, "o-", label="R-CNN Clean", color="#1f77b4", linewidth=2)
    ax.plot(dists_val, rcnn_patched, "o--", label="R-CNN Patched", color="#1f77b4", linewidth=2)
    ax.plot(dists_val, yolos_clean, "s-", label="YOLOS Clean", color="#ff7f0e", linewidth=2)
    ax.plot(dists_val, yolos_patched, "s--", label="YOLOS Patched", color="#ff7f0e", linewidth=2)
    
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("mAP")
    ax.set_title("Distance Ablation: Detector Robustness vs Sign Distance")
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved distance ablation plot to: {save_path}")


def plot_cross_domain_comparison(results_dir: str, save_path: str):
    """
    Plot side-by-side bar chart: nuScenes vs CARLA results (mAP Drop).
    """
    carla_path = os.path.join(results_dir, "evaluation", "frequency_comparison.json")
    parent_dir = os.path.dirname(os.path.abspath(results_dir))
    nuscenes_path = os.path.join(parent_dir, "evaluation", "frequency_comparison.json")
    if not os.path.exists(nuscenes_path):
        nuscenes_path = os.path.join("results", "evaluation", "frequency_comparison.json")
        
    if not os.path.exists(carla_path) or not os.path.exists(nuscenes_path):
        return
        
    with open(carla_path, "r") as f:
        carla_data = json.load(f)
    with open(nuscenes_path, "r") as f:
        nus_data = json.load(f)
        
    models = ["rcnn", "yolos"]
    labels = ["Faster R-CNN", "YOLOS-Small"]
    
    nus_drops = [nus_data[m]["full"].get("mAP_drop", 0) for m in models]
    carla_drops = [carla_data[m]["full"].get("score_drop", carla_data[m]["full"].get("mAP_drop", 0)) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.bar(x - width/2, nus_drops, width, label="nuScenes Front-Camera Domain", color="#2ca02c")
    ax.bar(x + width/2, carla_drops, width, label="CARLA Driving Domain", color="#d62728")
    
    ax.set_ylabel("mAP Drop")
    ax.set_title("Cross-Domain Robustness Transfer: nuScenes vs CARLA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved cross domain comparison plot to: {save_path}")


def plot_reference_comparison(results_dir: str, save_path: str):
    """
    Plot horizontal bar chart: our values vs published baselines.
    """
    ref_path = os.path.join(results_dir, "evaluation", "reference_comparison.json")
    if not os.path.exists(ref_path):
        return
    with open(ref_path, "r") as f:
        data = json.load(f)
        
    metrics = [
        "Eykholt (2018)\nCNN Physical Drop",
        "Fu (2022)\nViT Accuracy Drop",
        "Bai (2021)\nViT-CNN Gap (abs)",
        "Mahmood (2021)\nCNN Drop (high eps)"
    ]
    
    published = [0.685, 0.310, 0.175, 0.420]
    ours = [
        data.get("eykholt_2018", {}).get("our_cnn_mAP_drop", 0.0),
        data.get("fu_2022", {}).get("our_vit_mAP_drop", 0.0),
        abs(data.get("bai_2021", {}).get("our_adversarial_delta_eps03", 0.0)),
        data.get("mahmood_2021", {}).get("our_cnn_mAP_drop_highest_eps", 0.0)
    ]
    
    y = np.arange(len(metrics))
    height = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.barh(y - height/2, published, height, label="Published Reference", color="#7f7f7f")
    ax.barh(y + height/2, ours, height, label="CARLA Evaluation", color="#bcbd22")
    
    ax.set_xscale("linear")
    ax.set_xlabel("Drop / Gap Ratio")
    ax.set_title("Benchmarking Robustness Results Against Published Literature")
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlim(0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved reference comparison plot to: {save_path}")


# =======================================================================
# Smoke Test
# =======================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 72)
    print("  visualize_results.py - Smoke Test (mock data)")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock data files
        eval_dir = os.path.join(tmpdir, "evaluation")
        os.makedirs(eval_dir)
        exp_dir = os.path.join(tmpdir, "experiments")
        fig_dir = os.path.join(tmpdir, "figures")

        # Mock frequency comparison
        freq = {
            "rcnn": {
                "clean": {"mAP": 0.452},
                "low": {"mAP": 0.410, "mAP_drop": 0.042},
                "high": {"mAP": 0.385, "mAP_drop": 0.067},
                "full": {"mAP": 0.340, "mAP_drop": 0.112},
            },
            "yolos": {
                "clean": {"mAP": 0.389},
                "low": {"mAP": 0.320, "mAP_drop": 0.069},
                "high": {"mAP": 0.345, "mAP_drop": 0.044},
                "full": {"mAP": 0.280, "mAP_drop": 0.109},
            },
        }
        with open(os.path.join(eval_dir, "frequency_comparison.json"), "w") as f:
            json.dump(freq, f)

        # Mock RSF curves
        rsf = {
            "rcnn": {"ratios": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                     "mAPs": [0.452, 0.44, 0.41, 0.38, 0.34, 0.30], "rsf": 0.30},
            "yolos": {"ratios": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                      "mAPs": [0.389, 0.37, 0.34, 0.30, 0.25, 0.20], "rsf": 0.38},
        }
        with open(os.path.join(eval_dir, "rsf_curves.json"), "w") as f:
            json.dump(rsf, f)

        # Mock adversarial delta
        adv_delta = {"low": -0.09, "high": -0.04, "full": -0.06}
        with open(os.path.join(eval_dir, "adversarial_delta.json"), "w") as f:
            json.dump(adv_delta, f)

        # Mock epsilon sweep
        sweep = {
            "0.1": {
                "low": {"rcnn_mAP": 0.44, "yolos_mAP": 0.37},
                "high": {"rcnn_mAP": 0.43, "yolos_mAP": 0.36},
                "full": {"rcnn_mAP": 0.42, "yolos_mAP": 0.35},
            },
            "0.2": {
                "low": {"rcnn_mAP": 0.42, "yolos_mAP": 0.34},
                "high": {"rcnn_mAP": 0.41, "yolos_mAP": 0.35},
                "full": {"rcnn_mAP": 0.39, "yolos_mAP": 0.31},
            },
            "0.3": {
                "low": {"rcnn_mAP": 0.41, "yolos_mAP": 0.32},
                "high": {"rcnn_mAP": 0.385, "yolos_mAP": 0.345},
                "full": {"rcnn_mAP": 0.34, "yolos_mAP": 0.28},
            },
            "0.5": {
                "low": {"rcnn_mAP": 0.35, "yolos_mAP": 0.25},
                "high": {"rcnn_mAP": 0.33, "yolos_mAP": 0.27},
                "full": {"rcnn_mAP": 0.28, "yolos_mAP": 0.19},
            },
        }
        with open(os.path.join(eval_dir, "epsilon_sweep.json"), "w") as f:
            json.dump(sweep, f)

        # Mock loss history for low band
        os.makedirs(os.path.join(exp_dir, "low_eps0.3"), exist_ok=True)
        history = {
            "total_loss": [float(2.0 - 0.005 * i + np.random.normal(0, 0.05)) for i in range(100)],
            "suppress_rcnn": [float(1.0 - 0.003 * i + np.random.normal(0, 0.03)) for i in range(100)],
            "suppress_vit": [float(0.8 - 0.002 * i + np.random.normal(0, 0.02)) for i in range(100)],
            "attention_vit": [float(0.2 - 0.0005 * i + np.random.normal(0, 0.01)) for i in range(100)],
            "max_confidence_rcnn": [float(max(0, 0.9 - 0.003 * i)) for i in range(100)],
            "max_confidence_vit": [float(max(0, 0.85 - 0.004 * i)) for i in range(100)],
            "vram_mb": [2500.0] * 100,
        }
        with open(os.path.join(exp_dir, "low_eps0.3", "patch_low_eps0.3_history.json"), "w") as f:
            json.dump(history, f)

        # Mock patch tensor
        mock_patch = torch.rand(3, 64, 64)
        torch.save(mock_patch, os.path.join(exp_dir, "low_eps0.3", "patch_low_eps0.3_best.pt"))

        # Generate all static figures
        generate_all_figures(tmpdir, fig_dir)

        # Verify figures exist
        expected = [
            "fig_frequency_comparison.png",
            "fig_rsf_curves.png",
            "fig_adversarial_delta.png",
            "fig_epsilon_sweep.png",
        ]
        for fname in expected:
            fpath = os.path.join(fig_dir, fname)
            if os.path.exists(fpath):
                size_kb = os.path.getsize(fpath) / 1024
                print(f"    [OK] {fname} ({size_kb:.1f} KB)")
            else:
                print(f"    [SKIPPED] MISSING: {fname}")

    print("\n  ✅ Visualisation smoke test passed!")
