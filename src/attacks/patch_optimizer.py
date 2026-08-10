"""
patch_optimizer.py - Differentiable patch optimization engine.

This is the core script as it integrates all components of the adversarial
attack pipeline:
1. DCT Frequency Masking (dct_filter.py)
2. Expectation over Transformation (eot_transforms.py)
3. Patch Overlay (patch_applier.py)
4. Dual-Objective Loss (adversarial_loss.py)

Optimizes a 256x256 pixel patch to suppress object detections in both a CNN
(Faster R-CNN) and a ViT (YOLOS) while disrupting the ViT's attention graph.
Enforces strict VRAM management (< 5.8 GB).
"""

import os
import json
import time
import sys
import torch
import torchvision
import torch.nn as nn
from PIL import Image
import numpy as np

# Ensure project root is on sys.path so `from src.xxx` imports work
# regardless of which directory the script is run from.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.attacks.dct_filter import DCTFrequencyMask, apply_dct_mask
from src.attacks.eot_transforms import EoTTransformStack
from src.attacks.patch_applier import PatchApplier
from src.attacks.adversarial_loss import DualObjectiveLoss


class PatchOptimizer:
    """
    Manages the optimization loop for generating frequency-constrained
    adversarial patches under physical transformations.
    """

    def __init__(
        self,
        rcnn_wrapper,
        vit_wrapper,
        patch_size=256,
        patch_ratio=0.3,
        placement_mode="centre",
        lr=0.02,
        num_steps=300,
        tau=0.1,
        alpha=1.0,
        beta=1.0,
        gamma=0.3,
        epsilon=1.0,
        eot_config=None,
        device="cuda"
    ):
        """
        Parameters
        ----------
        rcnn_wrapper : FasterRCNNWrapper
            The CNN-based detector wrapper.
        vit_wrapper : DetrVitWrapper
            The ViT-based detector wrapper.
        patch_size : int
            Resolution of the square patch (default: 256).
        patch_ratio : float
            Size ratio of the patch relative to target bbox side length.
        placement_mode : 'centre' | 'random'
            Whether the patch is centred or randomly offset in target bboxes.
        lr : float
            Step size for sign-gradient optimization (PGD-style).
        num_steps : int
            Number of optimization steps.
        tau : float
            Confidence threshold for hinge loss detection suppression.
        alpha : float
            Weight for CNN suppression loss.
        beta : float
            Weight for ViT suppression loss.
        gamma : float
            Weight for ViT attention disruption loss.
        epsilon : float
            Perturbation budget - maximum pixel intensity for the patch.
            Controls spectral energy: lower = subtler patch, higher = stronger.
            Default 1.0 (full pixel range). Used in epsilon sweep experiments.
        eot_config : dict, optional
            Config parameters for EoT augmentation stack.
        device : str
            Device to run optimization on ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.rcnn = rcnn_wrapper
        self.vit = vit_wrapper
        self.patch_size = patch_size
        self.patch_ratio = patch_ratio
        self.placement_mode = placement_mode
        self.lr = lr
        self.num_steps = num_steps
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

        # Freeze detector parameters to ensure no model tuning happens
        for param in self.rcnn.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False

        self.rcnn.eval()
        self.vit.eval()

        # Initialize core components
        self.patch_applier = PatchApplier(
            placement_mode=placement_mode, patch_ratio=patch_ratio
        )
        self.loss_fn = DualObjectiveLoss(
            alpha=alpha, beta=beta, gamma=gamma, tau=tau
        )
        self.eot = EoTTransformStack(config=eot_config)
        self.eot.to(self.device)

        print(f"[Optimizer] Initialized PatchOptimizer | Device: {self.device}")
        print(f"            Patch size: {patch_size}x{patch_size} | ratio: {patch_ratio} | epsilon: {epsilon}")
        print(f"            Mode: {placement_mode} | lr: {lr} | steps: {num_steps}")
        print(f"            Loss weights: alpha={alpha}, beta={beta}, gamma={gamma} (tau={tau})")

    def optimize(
        self,
        images,
        bboxes_list,
        band="low",
        save_every=50,
        log_dir="results",
        experiment_name=None
    ):
        """
        Optimize the patch over a small set of clean images containing target objects.

        Parameters
        ----------
        images : torch.Tensor
            Batch of clean images, shape (N_img, C, H, W) normalized to [0,1].
        bboxes_list : List[torch.Tensor]
            Target bounding boxes per image. Each tensor is shape (N_boxes, 4).
        band : 'low' | 'high' | 'full'
            DCT frequency band constraints to apply.
        save_every : int
            Interval of steps for saving patch snapshots and metrics.
        log_dir : str
            Directory path to write results.
        experiment_name : str
            Unique name prefix for saved files. If None, generated automatically.

        Returns
        -------
        best_patch : torch.Tensor
            Optimized spatial patch of shape (3, patch_size, patch_size) on CPU.
        metrics : dict
            Dict of logged training histories (losses, confidence, VRAM).
        """
        start_time = time.time()
        
        # Temporarily lower the confidence thresholds during optimization to allow
        # gradients to flow even for low-confidence detections (dense gradient field)
        orig_rcnn_conf = getattr(self.rcnn, "conf_threshold", 0.25)
        orig_vit_conf = getattr(self.vit, "conf_threshold", 0.25)
        self.rcnn.conf_threshold = 0.01
        self.vit.conf_threshold = 0.01

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        if experiment_name is None:
            experiment_name = f"patch_{band}_{self.placement_mode}_{int(time.time())}"

        # 1. Initialize learnable parameters in spatial domain
        # Starting with random uniform initialization [0, 1]
        patch_params = torch.rand(
            (3, self.patch_size, self.patch_size),
            device=self.device,
            dtype=torch.float32,
            requires_grad=True
        )
        # Scale initialization to epsilon budget
        with torch.no_grad():
            patch_params.data.mul_(self.epsilon)

        # 2. Build the target DCT Frequency Mask
        dct_mask = DCTFrequencyMask(
            self.patch_size, self.patch_size, band=band, device=self.device
        )
        print(f"[Optimizer] Created frequency mask: {dct_mask}")

        # Move input images and bounding boxes to device
        images = images.to(self.device)
        bboxes_list = [b.to(self.device) for b in bboxes_list]

        # 3. Precompute clean attention rollouts for YOLOS on clean images
        # Since the detector is frozen, clean rollout is a constant benchmark
        print("[Optimizer] Precomputing clean attention rollouts from ViT...")
        with torch.no_grad():
            _ = self.vit(images)
            clean_rollouts = self.vit.get_attention_rollout()
        
        # Ensure clean_rollouts are on target device
        clean_rollouts = [r.to(self.device) for r in clean_rollouts]
        print(f"[Optimizer] Precomputed rollouts for {len(clean_rollouts)} samples.")

        # Set up logging structures
        history = {
            "total_loss": [],
            "suppress_rcnn": [],
            "suppress_vit": [],
            "attention_vit": [],
            "vram_mb": [],
            "max_confidence_rcnn": [],
            "max_confidence_vit": []
        }

        best_loss = float("inf")
        best_patch_params = patch_params.clone().detach()

        print(f"\n[Optimizer] Starting optimization for {self.num_steps} steps...")
        
        for step in range(1, self.num_steps + 1):
            # Clean gradient
            if patch_params.grad is not None:
                patch_params.grad.zero_()

            # 4. Generate frequency-constrained patch in spatial domain
            # Shape: (3, H, W)
            spatial_patch = apply_dct_mask(patch_params, dct_mask)
            
            # Expand to batch size for patch applier
            # Shape: (B, 3, H, W)
            batch_patches = spatial_patch.unsqueeze(0).expand(images.shape[0], -1, -1, -1)

            # 5. Apply patch differentiably to all target bounding boxes
            # Shape: (B, 3, H, W)
            patched_images = self.patch_applier(images, batch_patches, bboxes_list)

            # 6. Apply Expectation over Transformation (EoT) hardening
            # Augment the entire patched images to simulate real-world physics
            augmented_images = self.eot(patched_images)

            # 7. Forward pass through Faster R-CNN
            rcnn_detections = self.rcnn(augmented_images)

            # 8. Forward pass through YOLOS
            vit_detections = self.vit(augmented_images)
            patched_rollouts = self.vit.get_attention_rollout()
            patched_rollouts = [r.to(self.device) for r in patched_rollouts]

            # 9. Compute combined loss
            loss, loss_dict = self.loss_fn(
                rcnn_detections,
                vit_detections,
                clean_rollouts,
                patched_rollouts
            )

            # 10. Backward pass
            loss.backward()

            # 11. PGD-style signed gradient step
            with torch.no_grad():
                if patch_params.grad is not None:
                    # Ascend or descend? We want to MINIMIZE detection scores,
                    # so we perform standard gradient descent.
                    grad_sign = patch_params.grad.sign()
                    patch_params.data -= self.lr * grad_sign
                    
                    # Project parameters back into valid pixel range [0, epsilon]
                    patch_params.data.clamp_(0.0, self.epsilon)
                else:
                    print(f"WARNING: No gradient computed at step {step}!")

            # 12. Logging and tracking
            # Retrieve max confidences for diagnostics
            max_conf_rcnn = 0.0
            for det in rcnn_detections:
                if det["scores"].numel() > 0:
                    max_conf_rcnn = max(max_conf_rcnn, det["scores"].max().item())
            
            max_conf_vit = 0.0
            for det in vit_detections:
                if det["scores"].numel() > 0:
                    max_conf_vit = max(max_conf_vit, det["scores"].max().item())

            # VRAM tracking
            vram = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

            # Record metrics
            history["total_loss"].append(loss_dict["total"])
            history["suppress_rcnn"].append(loss_dict["suppress_rcnn"])
            history["suppress_vit"].append(loss_dict["suppress_vit"])
            history["attention_vit"].append(loss_dict["attention_vit"])
            history["max_confidence_rcnn"].append(max_conf_rcnn)
            history["max_confidence_vit"].append(max_conf_vit)
            history["vram_mb"].append(vram)

            # Keep track of the best patch parameters based on total loss
            if loss_dict["total"] < best_loss:
                best_loss = loss_dict["total"]
                best_patch_params = patch_params.clone().detach()

            # Visual progress print
            if step == 1 or step % 10 == 0 or step == self.num_steps:
                print(
                    f"Step {step:03d}/{self.num_steps:03d} | "
                    f"Loss: {loss_dict['total']:.4f} (RCNN: {loss_dict['suppress_rcnn']:.3f}, "
                    f"ViT: {loss_dict['suppress_vit']:.3f}, Attn: {loss_dict['attention_vit']:.3f}) | "
                    f"Max Conf: RCNN={max_conf_rcnn:.2%}, ViT={max_conf_vit:.2%} | "
                    f"VRAM: {vram:.1f} MB"
                )

                # Save intermediate patch image
                with torch.no_grad():
                    temp_spatial = apply_dct_mask(patch_params, dct_mask)
                    self._save_patch_image(
                        temp_spatial,
                        os.path.join(log_dir, f"{experiment_name}_step{step}.png")
                    )

            # VRAM safety cap assert
            assert vram < 5800.0, f"VRAM limit exceeded: {vram:.1f} MB (allowed: 5800 MB)"

        # 13. End of optimization: Save final outputs
        elapsed = time.time() - start_time
        print(f"\n[Optimizer] Optimization complete in {elapsed:.1f}s.")
        
        # Save best patch in spatial domain
        best_spatial = apply_dct_mask(best_patch_params, dct_mask)
        
        final_pt_path = os.path.join(log_dir, f"{experiment_name}_best.pt")
        final_png_path = os.path.join(log_dir, f"{experiment_name}_best.png")
        torch.save(best_spatial.cpu(), final_pt_path)
        self._save_patch_image(best_spatial, final_png_path)
        
        # Save history log as json
        history_path = os.path.join(log_dir, f"{experiment_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)

        print(f"[Optimizer] Saved best patch: {final_png_path}")
        # Restore original confidence thresholds
        self.rcnn.conf_threshold = orig_rcnn_conf
        self.vit.conf_threshold = orig_vit_conf

        return best_spatial.cpu(), history

    def _save_patch_image(self, patch_tensor, file_path):
        """
        Saves a (3, H, W) float32 tensor as a standard PNG image.
        """
        # (3, H, W) -> (H, W, 3)
        img_np = patch_tensor.detach().cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        img.save(file_path)


# ======================================================================
# Smoke test and pipeline integration test
# ======================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("Patch Optimizer - Smoke & Integration Test")
    print("=" * 72)

    from src.models.faster_rcnn_wrapper import FasterRCNNWrapper
    from src.models.detr_vit_wrapper import DetrVitWrapper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTarget device: {device}")

    # 1. Load detector wrappers
    # Using low thresholds and wrapper configs to ensure quick startup
    rcnn = FasterRCNNWrapper(conf_threshold=0.2, device=device)
    vit = DetrVitWrapper(conf_threshold=0.2, device=device)

    # 2. Setup mock optimization target
    # 2 images, size 640x640, with target bounding boxes
    print("\nCreating mock image target batch...")
    torch.manual_seed(42)
    mock_images = torch.rand(2, 3, 640, 640)
    
    # Image 1 has 1 stop sign bbox; Image 2 has 1 car bbox
    # [x1, y1, x2, y2]
    mock_bboxes = [
        torch.tensor([[100.0, 120.0, 300.0, 320.0]]),
        torch.tensor([[200.0, 200.0, 450.0, 450.0]])
    ]

    # 3. Initialize optimizer
    # Run a fast 10-step optimization to verify the entire autograd chain
    optimizer = PatchOptimizer(
        rcnn_wrapper=rcnn,
        vit_wrapper=vit,
        patch_size=256,
        patch_ratio=0.3,
        placement_mode="centre",
        lr=0.05,
        num_steps=10,
        tau=0.1,
        alpha=1.0,
        beta=1.0,
        gamma=0.3,
        device=device
    )

    # 4. Optimize!
    try:
        best_patch, history = optimizer.optimize(
            images=mock_images,
            bboxes_list=mock_bboxes,
            band="low",
            save_every=5,
            log_dir="scratch_results",
            experiment_name="smoke_test_patch"
        )
        
        print("\n" + "=" * 72)
        print("INTEGRATION TEST SUCCESSFUL!")
        print(f"Best patch shape:  {best_patch.shape}")
        print(f"Initial Loss:      {history['total_loss'][0]:.4f}")
        print(f"Final Loss:        {history['total_loss'][-1]:.4f}")
        print(f"Decreased:         {history['total_loss'][-1] < history['total_loss'][0]}")
        print(f"VRAM peak usage:   {max(history['vram_mb']):.1f} MB")
        print("=" * 72)

    except Exception as e:
        print("\nINTEGRATION TEST FAILED!")
        import traceback
        traceback.print_exc()

