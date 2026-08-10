"""Expectation over Transformation (EoT) - differentiable augmentations.

This module provides image-level augmentations that are applied during
adversarial-patch optimisation so the resulting patch generalises to
real-world imaging variations (camera angle, lighting, blur, …).

Every function operates on batched tensors (B, C, H, W) and is fully
differentiable through torch.autograd.  No OpenCV / PIL dependency -
only torch.nn.functional and pure tensor arithmetic.

Typical usage during patch training::

    eot = EoTTransformStack()          # default config
    augmented = eot(patched_images)    # stochastic, differentiable
    loss = detector(augmented)
    loss.backward()                    # gradients flow back through EoT

"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Helper: build a 2x3 affine matrix and warp
# ---------------------------------------------------------------------------

def _affine_warp(
    images: Tensor,
    theta: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tensor:
    """Apply a batch of 2X3 affine matrices to *images* via grid sampling.

    Parameters
    ----------
    images : Tensor (B, C, H, W)
    theta : Tensor (B, 2, 3)
        Affine transformation matrices.
    mode : str
        Interpolation mode for ``F.grid_sample``.
    padding_mode : str
        Padding mode for ``F.grid_sample``.

    Returns
    -------
    Tensor (B, C, H, W)
    """
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images, grid, mode=mode, padding_mode=padding_mode, align_corners=False
    )


# ===================================================================
# 1-a  Random rotation
# ===================================================================

def random_rotation(images: Tensor, max_angle: float = 15.0) -> Tensor:
    r"""Rotate each image in the batch by a uniformly-sampled angle.

    Parameters
    ----------
    images : Tensor (B, C, H, W)
        Input batch of images, values in [0, 1].
    max_angle : float
        Maximum absolute rotation angle in **degrees**.

    Returns
    -------
    Tensor (B, C, H, W)
        Rotated images.
    """
    B = images.shape[0]
    device = images.device
    dtype = images.dtype

    # Sample one angle per image (in radians)
    angles = (
        torch.empty(B, device=device, dtype=dtype).uniform_(-max_angle, max_angle)
        * (math.pi / 180.0)
    )

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    zeros = torch.zeros(B, device=device, dtype=dtype)

    # (B, 2, 3) affine matrices
    theta = torch.stack(
        [cos_a, -sin_a, zeros,
         sin_a,  cos_a, zeros],
        dim=1,
    ).reshape(B, 2, 3)

    return _affine_warp(images, theta)


# ===================================================================
# 1-b  Random scale
# ===================================================================

def random_scale(
    images: Tensor,
    min_scale: float = 0.8,
    max_scale: float = 1.2,
) -> Tensor:
    r"""Uniformly scale each image in the batch.

    Parameters
    ----------
    images : Tensor (B, C, H, W)
    min_scale, max_scale : float
        Bounds for the uniform scale factor.

    Returns
    -------
    Tensor (B, C, H, W)
    """
    B = images.shape[0]
    device = images.device
    dtype = images.dtype

    s = torch.empty(B, device=device, dtype=dtype).uniform_(min_scale, max_scale)
    inv_s = 1.0 / s
    zeros = torch.zeros(B, device=device, dtype=dtype)

    theta = torch.stack(
        [inv_s, zeros, zeros,
         zeros, inv_s, zeros],
        dim=1,
    ).reshape(B, 2, 3)

    return _affine_warp(images, theta)


# ===================================================================
# 1-c  Gaussian blur
# ===================================================================

def gaussian_blur(
    images: Tensor,
    sigma_range: Tuple[float, float] = (0.5, 2.0),
) -> Tensor:
    r"""Apply Gaussian blur with a randomly-sampled sigma.

    Parameters
    ----------
    images : Tensor (B, C, H, W)
    sigma_range : tuple of float
        ``(sigma_min, sigma_max)`` for the uniform sample.

    Returns
    -------
    Tensor (B, C, H, W)
    """
    device = images.device
    dtype = images.dtype
    C = images.shape[1]

    # --- sample sigma ---------------------------------------------------
    sigma = random.uniform(sigma_range[0], sigma_range[1])

    # --- build 1-D kernel ------------------------------------------------
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1  # ensure odd
    half = kernel_size // 2

    # coordinate grid centred at 0
    x = torch.arange(kernel_size, device=device, dtype=dtype) - half
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()  # normalise  (differentiable)

    # --- depthwise separable convolution ---------------------------------
    # Horizontal pass  (1, 1, 1, K)  ->  replicate C times  ->  (C, 1, 1, K)
    weight_h = kernel_1d.reshape(1, 1, 1, kernel_size).expand(C, 1, 1, kernel_size)
    # Vertical pass    (1, 1, K, 1)
    weight_v = kernel_1d.reshape(1, 1, kernel_size, 1).expand(C, 1, kernel_size, 1)

    pad_h = (half, half, 0, 0)       # left, right, top, bottom
    pad_v = (0, 0, half, half)

    out = F.pad(images, pad_h, mode="reflect")
    out = F.conv2d(out, weight_h, groups=C)
    out = F.pad(out, pad_v, mode="reflect")
    out = F.conv2d(out, weight_v, groups=C)

    return out


# ===================================================================
# 1-d  Random brightness
# ===================================================================

def random_brightness(images: Tensor, max_delta: float = 0.15) -> Tensor:
    r"""Additive brightness shift.

    Parameters
    ----------
    images : Tensor (B, C, H, W)
    max_delta : float
        Maximum absolute brightness shift.

    Returns
    -------
    Tensor (B, C, H, W)
    """
    delta = (
        torch.empty(1, device=images.device, dtype=images.dtype)
        .uniform_(-max_delta, max_delta)
    )
    return (images + delta).clamp(0.0, 1.0)


# ===================================================================
# 1-e  Random colour jitter (per-channel multiplicative)
# ===================================================================

def random_color_jitter(images: Tensor, max_delta: float = 0.1) -> Tensor:
    r"""Per-channel multiplicative colour jitter.

    Parameters
    ----------
    images : Tensor (B, C, H, W)
    max_delta : float
        Half-width of the multiplicative factor range.

    Returns
    -------
    Tensor (B, C, H, W)
    """
    C = images.shape[1]
    factors = (
        torch.empty(1, C, 1, 1, device=images.device, dtype=images.dtype)
        .uniform_(1.0 - max_delta, 1.0 + max_delta)
    )
    return (images * factors).clamp(0.0, 1.0)


# ===================================================================
# 1-f  Random perspective warp
# ===================================================================

def random_perspective(
    images: Tensor,
    distortion_scale: float = 0.05,
) -> Tensor:
    r"""Perspective warp by perturbing the four corner points.

    Each corner of the image is displaced by up to
    ``distortion_scale X image_size`` pixels.  A perspective (projective)
    homography :math:`H` is computed from the four correspondences, then
    applied via ``F.grid_sample``.

    Parameters
    ----------
    images : Tensor (B, C, H, W)
    distortion_scale : float
        Fraction of image size used as maximum corner displacement.

    Returns
    -------
    Tensor (B, C, H, W)
    """
    B, _C, H, W = images.shape
    device = images.device
    dtype = images.dtype

    half_h, half_w = H / 2.0, W / 2.0

    # Source corners in normalised coords [-1, 1]
    src = torch.tensor(
        [[-1.0, -1.0],
         [ 1.0, -1.0],
         [ 1.0,  1.0],
         [-1.0,  1.0]],
        device=device, dtype=dtype,
    )  # (4, 2)

    # Random perturbations in normalised coords
    # distortion_scale * image_size  ->  in normalised space  = distortion_scale * 2
    d = distortion_scale * 2.0
    offsets = torch.empty(B, 4, 2, device=device, dtype=dtype).uniform_(-d, d)
    dst = src.unsqueeze(0) + offsets  # (B, 4, 2)

    # --- Solve for 3x3 homography per sample via DLT ---------------------
    # We map *output* (dst) to *input* (src) so we can use grid_sample.
    # 8 equations per sample, 8 unknowns (h33 = 1).
    src_exp = src.unsqueeze(0).expand(B, 4, 2)  # (B, 4, 2)

    x_s, y_s = src_exp[..., 0], src_exp[..., 1]  # (B, 4)
    x_d, y_d = dst[..., 0], dst[..., 1]           # (B, 4)

    ones = torch.ones_like(x_d)
    zeros = torch.zeros_like(x_d)

    # Each point contributes two rows to A h = 0 (where h33=1 is moved to rhs).
    # Row 1: x_d  y_d  1  0  0  0  -x_s*x_d  -x_s*y_d  | x_s
    # Row 2: 0  0  0  x_d  y_d  1  -y_s*x_d  -y_s*y_d  | y_s
    A_rows = []
    b_rows = []
    for i in range(4):
        xd_i, yd_i = x_d[:, i], y_d[:, i]
        xs_i, ys_i = x_s[:, i], y_s[:, i]

        row1 = torch.stack(
            [xd_i, yd_i, ones[:, i],
             zeros[:, i], zeros[:, i], zeros[:, i],
             -xs_i * xd_i, -xs_i * yd_i],
            dim=1,
        )  # (B, 8)
        row2 = torch.stack(
            [zeros[:, i], zeros[:, i], zeros[:, i],
             xd_i, yd_i, ones[:, i],
             -ys_i * xd_i, -ys_i * yd_i],
            dim=1,
        )
        A_rows.extend([row1, row2])
        b_rows.extend([xs_i, ys_i])

    A = torch.stack(A_rows, dim=1)  # (B, 8, 8)
    b = torch.stack(b_rows, dim=1).unsqueeze(-1)  # (B, 8, 1)

    h = torch.linalg.solve(A, b).squeeze(-1)  # (B, 8)
    H_mat = torch.cat(
        [h, torch.ones(B, 1, device=device, dtype=dtype)], dim=1
    ).reshape(B, 3, 3)  # (B, 3, 3)

    # --- Build sampling grid from H -------------------------------------
    # Grid of normalised output coordinates
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij",
    )
    ones_grid = torch.ones_like(gx)
    coords = torch.stack([gx, gy, ones_grid], dim=-1).reshape(-1, 3)  # (H*W, 3)

    # Apply homography: src_coords = H @ dst_coords
    coords_t = coords.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 3)
    mapped = torch.bmm(coords_t, H_mat.transpose(1, 2))  # (B, H*W, 3)

    # Perspective divide
    mapped_xy = mapped[..., :2] / (mapped[..., 2:3] + 1e-8)  # (B, H*W, 2)
    grid = mapped_xy.reshape(B, H, W, 2)

    return F.grid_sample(
        images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


# ===================================================================
# 2.  EoTTransformStack
# ===================================================================

# Registry mapping config keys to callables and their keyword argument names.
_TRANSFORM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "rotation": {
        "fn": random_rotation,
        "default_params": {"max_angle": 15.0},
    },
    "scale": {
        "fn": random_scale,
        "default_params": {"min_scale": 0.8, "max_scale": 1.2},
    },
    "blur": {
        "fn": gaussian_blur,
        "default_params": {"sigma_range": (0.5, 2.0)},
    },
    "brightness": {
        "fn": random_brightness,
        "default_params": {"max_delta": 0.15},
    },
    "color_jitter": {
        "fn": random_color_jitter,
        "default_params": {"max_delta": 0.1},
    },
    "perspective": {
        "fn": random_perspective,
        "default_params": {"distortion_scale": 0.05},
    },
}

DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    name: {"enabled": True, **entry["default_params"]}
    for name, entry in _TRANSFORM_REGISTRY.items()
}


class EoTTransformStack(nn.Module):
    """Differentiable Expectation-over-Transformation augmentation stack.

    At every forward pass the enabled transforms are applied in a **freshly
    randomised order** with **freshly sampled** parameters, exactly mimicking
    the stochastic sampling required by the EoT formulation.

    Parameters
    ----------
    config : dict, optional
        Dictionary keyed by transform name ("rotation", "scale",
        "blur", "brightness", "color_jitter", "perspective").
        Each value is itself a dict with an "enabled" flag plus any
        keyword arguments to pass to the underlying function.

        If None, all transforms are enabled with their default parameters
        (see DEFAULT_CONFIG).

    Example
    -------
    >>> stack = EoTTransformStack({"rotation": {"enabled": True, "max_angle": 30.0},
    ...                            "blur": {"enabled": False}})
    >>> out = stack(images)
    """

    def __init__(self, config: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        super().__init__()
        if config is None:
            config = DEFAULT_CONFIG

        self._transforms: list[Tuple[str, Any, Dict[str, Any]]] = []

        for name, entry in _TRANSFORM_REGISTRY.items():
            cfg = config.get(name, {"enabled": False})
            if not cfg.get("enabled", False):
                continue
            # Collect keyword args (everything except "enabled")
            kwargs = {k: v for k, v in cfg.items() if k != "enabled"}
            # Fill in defaults for any missing keys
            for k, v in entry["default_params"].items():
                kwargs.setdefault(k, v)
            self._transforms.append((name, entry["fn"], kwargs))

    # ------------------------------------------------------------------
    def forward(self, images: Tensor) -> Tensor:
        """Apply all enabled transforms in a random order.

        Parameters
        ----------
        images : Tensor (B, C, H, W)
            Input images, typically the scene with an adversarial patch
            already applied.

        Returns
        -------
        Tensor (B, C, H, W)
            Augmented images.  The computational graph is preserved for
            back-propagation.
        """
        order = list(range(len(self._transforms)))
        random.shuffle(order)

        out = images
        for idx in order:
            _name, fn, kwargs = self._transforms[idx]
            out = fn(out, **kwargs)
        return out

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        lines = [f"{self.__class__.__name__}("]
        for name, _fn, kwargs in self._transforms:
            lines.append(f"  {name}: {kwargs}")
        lines.append(")")
        return "\n".join(lines)


# ===================================================================
# 3.  Smoke test
# ===================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 3, 256, 256
    images = torch.rand(B, C, H, W, device=device, requires_grad=True)

    print("=" * 70)
    print("EoT Transforms - Smoke Test")
    print(f"Device : {device}")
    print(f"Input  : shape={tuple(images.shape)}  requires_grad={images.requires_grad}")
    print("=" * 70)

    # --- Individual transforms -------------------------------------------
    individual_transforms = [
        ("random_rotation",    random_rotation,    {"max_angle": 15.0}),
        ("random_scale",       random_scale,       {"min_scale": 0.8, "max_scale": 1.2}),
        ("gaussian_blur",      gaussian_blur,      {"sigma_range": (0.5, 2.0)}),
        ("random_brightness",  random_brightness,  {"max_delta": 0.15}),
        ("random_color_jitter", random_color_jitter, {"max_delta": 0.1}),
        ("random_perspective", random_perspective,  {"distortion_scale": 0.05}),
    ]

    all_ok = True
    for name, fn, kwargs in individual_transforms:
        out = fn(images, **kwargs)
        shape_ok = out.shape == images.shape
        grad_ok = out.grad_fn is not None
        status = "PASS" if (shape_ok and grad_ok) else "FAIL"
        if not (shape_ok and grad_ok):
            all_ok = False
        print(
            f"  [{status}] {name:25s}  "
            f"shape={tuple(out.shape)}  grad_fn={out.grad_fn is not None}"
        )

    # --- Full EoTTransformStack ------------------------------------------
    print("-" * 70)
    print("EoTTransformStack (all defaults):")
    stack = EoTTransformStack()
    print(stack)

    out_stack = stack(images)
    shape_ok = out_stack.shape == images.shape
    grad_ok = out_stack.grad_fn is not None
    status = "PASS" if (shape_ok and grad_ok) else "FAIL"
    if not (shape_ok and grad_ok):
        all_ok = False
    print(
        f"  [{status}] stack forward     "
        f"shape={tuple(out_stack.shape)}  grad_fn={out_stack.grad_fn is not None}"
    )

    # --- Backward pass ---------------------------------------------------
    loss = out_stack.sum()
    loss.backward()
    grad_exists = images.grad is not None
    grad_nonzero = images.grad is not None and images.grad.abs().sum().item() > 0
    status = "PASS" if (grad_exists and grad_nonzero) else "FAIL"
    if not (grad_exists and grad_nonzero):
        all_ok = False
    print(
        f"  [{status}] backward pass     "
        f"grad exists={grad_exists}  grad nonzero={grad_nonzero}"
    )

    # --- Summary ---------------------------------------------------------
    print("=" * 70)
    if all_ok:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)
