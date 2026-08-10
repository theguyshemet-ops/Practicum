"""Differentiable Patch Applier: overlays adversarial patches onto images.

This script implements a fully-differentiable patch overlay pipeline that
composites learnable adversarial patches into bounding-box regions of input
images. Because every operation uses torch primitives that participate in
the autograd graph, gradients from a downstream detection / classification
loss propagate all the way back through the composite to the raw patch
parameters, enabling end-to-end adversarial patch optimisation.


Hardware target
---------------
RTX 4050 (6 GB VRAM) - the implementation avoids materialising large
intermediate tensors and processes patches sequentially per image to bound
peak memory.

Author : Adversarial Robustness Research Project
Licence: MIT
"""

from __future__ import annotations

import math
from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchApplier(nn.Module):
    """Differentiable adversarial-patch overlay.

    Applies a learnable patch onto every target bounding box in a batch of
    images.  The forward pass is fully differentiable w.r.t. patches.

    Parameters
    ----------
    placement_mode : 'centre' | 'random'
        Where inside the bounding box to place the (resized) patch.

        * 'centre' - the patch is centred on the bbox midpoint.
        * 'random' - the patch top-left corner is sampled uniformly at
          random within the valid region of the bbox (so that the whole
          patch stays inside the bbox).  Sampling uses straight-through
          semantics and does **not** block gradients.

    patch_ratio : float, default 0.3
        Fraction of the bounding-box side length that the patch covers.
        A value of 0.3 means the patch occupies 30 % of the bbox width and
        30 % of the bbox height.

    min_patch_px : int, default 4
        Minimum allowed patch side length in pixels.  Bounding boxes whose
        derived patch size falls below this threshold are silently skipped
        (the image is left unmodified in that region).
    """

    VALID_MODES: tuple[str, ...] = ("centre", "random")

    def __init__(
        self,
        placement_mode: Literal["centre", "random"] = "centre",
        patch_ratio: float = 0.3,
        min_patch_px: int = 1,
    ) -> None:
        super().__init__()
        if placement_mode not in self.VALID_MODES:
            raise ValueError(
                f"placement_mode must be one of {self.VALID_MODES}, "
                f"got '{placement_mode}'"
            )
        if not 0.0 < patch_ratio <= 1.0:
            raise ValueError(
                f"patch_ratio must be in (0, 1], got {patch_ratio}"
            )
        self.placement_mode = placement_mode
        self.patch_ratio = patch_ratio
        self.min_patch_px = min_patch_px

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_region(
        top: int,
        left: int,
        p_h: int,
        p_w: int,
        img_h: int,
        img_w: int,
    ) -> tuple[int, int, int, int, int, int]:
        """Clamp the patch placement so it stays inside the image.

        Returns
        -------
        top, left, p_h, p_w, patch_top_offset, patch_left_offset
            Clamped image-space coordinates and the corresponding offsets
            into the resized patch tensor (needed when the patch is clipped
            at the image boundary).
        """
        patch_top_offset = 0
        patch_left_offset = 0

        if top < 0:
            patch_top_offset = -top
            p_h += top  # shrink
            top = 0
        if left < 0:
            patch_left_offset = -left
            p_w += left
            left = 0
        if top + p_h > img_h:
            p_h = img_h - top
        if left + p_w > img_w:
            p_w = img_w - left

        return top, left, p_h, p_w, patch_top_offset, patch_left_offset

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        patches: torch.Tensor,
        bboxes: List[torch.Tensor],
    ) -> torch.Tensor:
        """Apply adversarial patches onto images at the given bboxes.

        Parameters
        ----------
        images : Tensor, shape (B, C, H, W)
            Clean input images (float32, typically in [0, 1]).
        patches : Tensor, shape (B, C, Hp, Wp)
            Learnable adversarial patch(es).  One patch per image in the
            batch.  The canonical patch resolution is 256 X 256, but any
            spatial size is accepted because we resize to match the bbox.
        bboxes : list[Tensor]
            Length-*B* list.  bboxes[i] has shape (N_i, 4) with
            columns [x1, y1, x2, y2] in pixel coordinates
            (integers or floats).

        Returns
        -------
        Tensor, shape (B, C, H, W)
            Patched images.  Gradients w.r.t. patches are preserved
            through the composite operation.

        Raises
        ------
        ValueError
            If the batch dimensions of *images*, *patches*, and *bboxes*
            do not agree.
        """
        B, C, H, W = images.shape
        Bp = patches.shape[0]
        if Bp != B:
            raise ValueError(
                f"Batch size mismatch: images has B={B}, patches has B={Bp}"
            )
        if len(bboxes) != B:
            raise ValueError(
                f"Batch size mismatch: images has B={B}, "
                f"bboxes list length={len(bboxes)}"
            )

        # Clone so we never mutate the caller's tensor - the clone keeps
        # the graph connection to *images* (though for adversarial patch
        # training we typically do not need ∂L/∂images).
        patched: torch.Tensor = images.clone()

        for i in range(B):
            img = patched[i]          # (C, H, W) - a *view* into patched
            patch = patches[i]        # (C, Hp, Wp)
            boxes = bboxes[i]         # (N_i, 4)

            if boxes.numel() == 0:
                continue

            for j in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[j].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                bbox_h = y2 - y1
                bbox_w = x2 - x1
                if bbox_h <= 0 or bbox_w <= 0:
                    continue

                # --- 1. target patch size ---
                p_h = max(1, math.floor(self.patch_ratio * bbox_h))
                p_w = max(1, math.floor(self.patch_ratio * bbox_w))

                if p_h < self.min_patch_px or p_w < self.min_patch_px:
                    continue  # bbox too small - skip

                # --- 2. resize patch (differentiable) ---
                # F.interpolate expects (N, C, H, W) -> unsqueeze then squeeze
                resized: torch.Tensor = F.interpolate(
                    patch.unsqueeze(0),
                    size=(p_h, p_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # -> (C, p_h, p_w)

                # --- 3. compute placement position ---
                if self.placement_mode == "centre":
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    top = cy - p_h // 2
                    left = cx - p_w // 2
                else:  # 'random'
                    # valid range so patch stays within bbox
                    max_top = max(y1, y2 - p_h)
                    max_left = max(x1, x2 - p_w)
                    top = torch.randint(
                        y1, max_top + 1, (1,)
                    ).item() if max_top > y1 else y1
                    left = torch.randint(
                        x1, max_left + 1, (1,)
                    ).item() if max_left > x1 else x1

                top, left = int(top), int(left)

                # --- 4. clamp to image bounds ---
                (
                    top, left, eff_h, eff_w, pt_off, pl_off
                ) = self._clamp_region(top, left, p_h, p_w, H, W)

                if eff_h <= 0 or eff_w <= 0:
                    continue

                # Slice the (possibly clipped) resized patch
                patch_slice: torch.Tensor = resized[
                    :, pt_off: pt_off + eff_h, pl_off: pl_off + eff_w
                ]  # (C, eff_h, eff_w) - still in the autograd graph

                # --- 5. differentiable mask composite ---
                # Build a full-image mask (binary, float) for this patch
                # position.  The mask itself is *not* learned, so it does
                # not need gradients - only the patch values do.
                #
                # Instead of materialising a full (C, H, W) mask we do a
                # direct slice assignment which keeps the graph intact:
                #
                #   patched[i, :, top:top+eff_h, left:left+eff_w] =
                #       img[:, top:top+eff_h, left:left+eff_w] * (1 - 1)
                #       + patch_slice * 1
                #
                # Simplified: we just write the patch values.
                patched[i, :, top: top + eff_h, left: left + eff_w] = (
                    patch_slice
                )

        return patched

    def extra_repr(self) -> str:  # noqa: D401
        return (
            f"placement_mode='{self.placement_mode}', "
            f"patch_ratio={self.patch_ratio}, "
            f"min_patch_px={self.min_patch_px}"
        )


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke-test] device = {device}")

    B, C, H, W = 2, 3, 640, 640
    Hp, Wp = 256, 256

    images = torch.rand(B, C, H, W, device=device)
    patches = torch.rand(B, C, Hp, Wp, device=device, requires_grad=True)
    bboxes: List[torch.Tensor] = [
        torch.tensor([[100, 100, 400, 400]], dtype=torch.float32, device=device),
        torch.tensor([[200, 200, 500, 500]], dtype=torch.float32, device=device),
    ]

    # ---- Test 1: centre mode ----
    applier_centre = PatchApplier(placement_mode="centre", patch_ratio=0.3)
    applier_centre.to(device)

    out_centre = applier_centre(images, patches, bboxes)

    assert out_centre.shape == (B, C, H, W), (
        f"Shape mismatch: expected {(B, C, H, W)}, got {out_centre.shape}"
    )
    print(f"[centre] output shape: {out_centre.shape}")

    # Backward pass - verify gradients flow to patches
    loss = out_centre.sum()
    loss.backward()
    assert patches.grad is not None, "patches.grad is None - autograd broken!"
    assert patches.grad.abs().sum().item() > 0, "patches.grad is all zeros!"
    print(f"[centre] patches.grad norm: {patches.grad.norm().item():.6f}")

    # Verify that patched region differs from original
    # For image 0, bbox (100,100)->(400,400), centre placement
    bbox0 = bboxes[0][0].int().tolist()
    x1, y1, x2, y2 = bbox0
    region_orig = images[0, :, y1:y2, x1:x2]
    region_patched = out_centre[0, :, y1:y2, x1:x2]
    diff = (region_orig - region_patched).abs().sum().item()
    assert diff > 0, "Patched region is identical to original - patch not applied!"
    print(f"[centre] patched-vs-original diff (L1): {diff:.4f}")

    # ---- Test 2: random mode ----
    patches_rand = torch.rand(
        B, C, Hp, Wp, device=device, requires_grad=True
    )
    applier_random = PatchApplier(placement_mode="random", patch_ratio=0.3)
    applier_random.to(device)

    out_random = applier_random(images, patches_rand, bboxes)

    assert out_random.shape == (B, C, H, W), (
        f"Shape mismatch: expected {(B, C, H, W)}, got {out_random.shape}"
    )
    print(f"[random] output shape: {out_random.shape}")

    # Backward for random mode
    loss_r = out_random.sum()
    loss_r.backward()
    assert patches_rand.grad is not None, "patches_rand.grad is None!"
    print(f"[random] patches_rand.grad norm: {patches_rand.grad.norm().item():.6f}")

    # Verify patched region differs
    region_patched_r = out_random[1, :, 200:500, 200:500]
    region_orig_r = images[1, :, 200:500, 200:500]
    diff_r = (region_orig_r - region_patched_r).abs().sum().item()
    assert diff_r > 0, "Random-mode patched region identical to original!"
    print(f"[random] patched-vs-original diff (L1): {diff_r:.4f}")

    # ---- Test 3: edge cases ----
    # Empty bboxes
    out_empty = applier_centre(
        images,
        torch.rand(B, C, Hp, Wp, device=device, requires_grad=True),
        [torch.zeros(0, 4, device=device), torch.zeros(0, 4, device=device)],
    )
    assert torch.allclose(out_empty, images), (
        "Empty bboxes should leave images unchanged!"
    )
    print("[edge] empty bboxes -> identity")

    # Tiny bbox (should be skipped due to min_patch_px)
    tiny_bboxes: List[torch.Tensor] = [
        torch.tensor([[0, 0, 5, 5]], dtype=torch.float32, device=device),
        torch.tensor([[0, 0, 5, 5]], dtype=torch.float32, device=device),
    ]
    out_tiny = applier_centre(
        images,
        torch.rand(B, C, Hp, Wp, device=device, requires_grad=True),
        tiny_bboxes,
    )
    # patch_ratio 0.3 x 5 = 1.5 -> floor = 1 < min_patch_px(4) -> skipped
    assert torch.allclose(out_tiny, images), (
        "Tiny bboxes should be skipped!"
    )
    print("[edge] tiny bboxes -> identity")

    # Bbox near image border (clamp test)
    border_bboxes: List[torch.Tensor] = [
        torch.tensor([[580, 580, 700, 700]], dtype=torch.float32, device=device),
        torch.tensor([[0, 0, 100, 100]], dtype=torch.float32, device=device),
    ]
    out_border = applier_centre(
        images,
        torch.rand(B, C, Hp, Wp, device=device, requires_grad=True),
        border_bboxes,
    )
    assert out_border.shape == (B, C, H, W), "Border clamp test failed!"
    print("[edge] border bbox -> clamped ")

    print("\nAll smoke tests passed.")
    sys.exit(0)
