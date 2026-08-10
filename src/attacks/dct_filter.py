"""
dct_filter.py - Differentiable 2D Discrete Cosine Transform filter.

Core novelty module for frequency-constrained adversarial patch generation.
Constrains learnable patch parameters to specific DCT frequency bands (low,
high, or full) so that the resulting spatial-domain patch only contains
energy in the chosen band.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal, Union

import torch
import torch.nn.functional as F  # noqa: N812 - kept for consistency

# ---------------------------------------------------------------------------
# Basis matrix construction
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _basis_matrix(N: int, device: torch.device) -> torch.Tensor:
    """Build the orthonormal DCT-II basis matrix of size (N, N).

    The matrix is registered as a non-leaf buffer (requires_grad=False)
    so it never accumulates gradients but participates in the autograd graph through matmul with learnable inputs.

    Parameters
    ----------
    N: int
        Spatial dimension (height or width).
    device: torch.device
        Target device for the tensor.

    Returns
    -------
    torch.Tensor
        Orthonormal basis matrix of shape (N, N), dtype float64.
        Using float64 ensures round-trip reconstruction error stays
        below 1e-10 even for large N (e.g. 256).  The calling functions handle dtype casting so the autograd graph remains intact.
    """
    k = torch.arange(N, dtype=torch.float64, device=device).unsqueeze(1)  # (N, 1)
    n = torch.arange(N, dtype=torch.float64, device=device).unsqueeze(0)  # (1, N)

    T = math.sqrt(2.0 / N) * torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
    T[0, :] *= 1.0 / math.sqrt(2.0)  # orthonormality scaling for k=0
    return T


# ---------------------------------------------------------------------------
# 2D DCT / IDCT
# ---------------------------------------------------------------------------

def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """Differentiable 2-D orthonormal DCT-II via matrix multiplication.

    Computes X = T_h · x · T_w^T independently for each channel (and each sample in a batch).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (C, H, W) or (B, C, H, W).

    Returns
    -------
    torch.Tensor
        DCT coefficient tensor, same shape as x.
    """
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True

    orig_dtype = x.dtype
    B, C, H, W = x.shape
    T_h = _basis_matrix(H, x.device)  # (H, H) float64
    T_w = _basis_matrix(W, x.device)  # (W, W) float64

    # Upcast to float64 for numerical precision; cast is differentiable.
    x_flat = x.reshape(B * C, H, W).to(torch.float64)

    # Forward DCT: T_h @ x @ T_w^T
    out = torch.bmm(
        T_h.unsqueeze(0).expand(B * C, -1, -1),
        torch.bmm(x_flat, T_w.t().unsqueeze(0).expand(B * C, -1, -1)),
    )
    out = out.reshape(B, C, H, W).to(orig_dtype)

    if squeeze:
        out = out.squeeze(0)
    return out


def idct_2d(X: torch.Tensor) -> torch.Tensor:
    """Differentiable 2-D orthonormal inverse DCT (DCT-III) via matrix
    multiplication.

    Computes x = T_h^T · X · T_w which is the exact inverse of
    :func:dct_2d because the basis matrices are orthogonal.

    Parameters
    ----------
    X : torch.Tensor
        DCT coefficient tensor of shape (C, H, W) or (B, C, H, W).

    Returns
    -------
    torch.Tensor
        Reconstructed spatial-domain tensor, same shape as X.
    """
    squeeze = False
    if X.ndim == 3:
        X = X.unsqueeze(0)
        squeeze = True

    orig_dtype = X.dtype
    B, C, H, W = X.shape
    T_h = _basis_matrix(H, X.device)  # (H, H) float64
    T_w = _basis_matrix(W, X.device)  # (W, W) float64

    # Upcast to float64 for numerical precision; cast is differentiable.
    X_flat = X.reshape(B * C, H, W).to(torch.float64)

    # Inverse DCT: T_h^T @ X @ T_w
    out = torch.bmm(
        T_h.t().unsqueeze(0).expand(B * C, -1, -1),
        torch.bmm(X_flat, T_w.unsqueeze(0).expand(B * C, -1, -1)),
    )
    out = out.reshape(B, C, H, W).to(orig_dtype)

    if squeeze:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Frequency mask
# ---------------------------------------------------------------------------

class DCTFrequencyMask:
    """Binary mask that selects DCT coefficients by Manhattan distance from DC.

    The DC component sits at index (0, 0) (top-left).  The Manhattan
    distance of coefficient (u, v) is simply d = u + v.

    Parameters
    ----------
    height : int
        Spatial height of the DCT block.
    width : int
        Spatial width of the DCT block.
    band : 'low' | 'high' | 'full'
        Which frequency band to retain.
    device : torch.device or str
        Target device for the mask tensor.

    Attributes
    ----------
    mask : torch.Tensor
        Binary mask of shape (1, H, W) broadcastable over channels and
        batch dimensions.
    threshold : int
        Manhattan-distance threshold T = int(0.25 * (H + W - 2)).

    Notes
    -----
    The threshold is chosen so that roughly 25 % of the H x W
    coefficients satisfy d <= T (the exact fraction depends on the aspect ratio, but is close for square inputs).
    """

    def __init__(
        self,
        height: int,
        width: int,
        band: Literal["low", "high", "full"],
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        if band not in ("low", "high", "full"):
            raise ValueError(
                f"band must be 'low', 'high', or 'full', got '{band}'"
            )

        self.height = height
        self.width = width
        self.band = band
        self.device = torch.device(device)

        # Threshold: ~25 % of coefficients lie in the low band.
        self.threshold: int = int(0.25 * (height + width - 2))

        # Build Manhattan-distance grid
        u = torch.arange(height, device=self.device).unsqueeze(1)  # (H, 1)
        v = torch.arange(width, device=self.device).unsqueeze(0)   # (1, W)
        dist = u + v  # (H, W)

        if band == "low":
            mask_2d = (dist <= self.threshold).float()
        elif band == "high":
            mask_2d = (dist > self.threshold).float()
        else:  # 'full'
            mask_2d = torch.ones(height, width, device=self.device)

        # Shape (1, H, W) - broadcastable over (B, C, H, W)
        self.mask: torch.Tensor = mask_2d.unsqueeze(0)

    # Convenience ----------------------------------------------------------

    def to(self, device: Union[torch.device, str]) -> "DCTFrequencyMask":
        """Return a copy of the mask on *device*."""
        new = DCTFrequencyMask.__new__(DCTFrequencyMask)
        new.height = self.height
        new.width = self.width
        new.band = self.band
        new.threshold = self.threshold
        new.device = torch.device(device)
        new.mask = self.mask.to(new.device)
        return new

    def __repr__(self) -> str:
        frac = self.mask.sum().item() / (self.height * self.width)
        return (
            f"DCTFrequencyMask(h={self.height}, w={self.width}, "
            f"band='{self.band}', threshold={self.threshold}, "
            f"coverage={frac:.1%})"
        )


# ---------------------------------------------------------------------------
# Apply mask pipeline
# ---------------------------------------------------------------------------

def apply_dct_mask(
    patch_params: torch.Tensor,
    mask: Union[DCTFrequencyMask, torch.Tensor],
) -> torch.Tensor:
    """Project learnable patch parameters to a frequency-constrained spatial patch.

    Pipeline (fully differentiable)::

        spatial_patch = clamp( IDCT2( DCT2(patch_params) . mask ), 0, 1 )

    Parameters
    ----------
    patch_params : torch.Tensor
        Learnable parameters in spatial domain, shape (C, H, W) or
        (B, C, H, W).  Must have requires_grad=True for gradient flow.
    mask : DCTFrequencyMask or torch.Tensor
        Frequency-domain mask.  If a DCTFrequencyMask instance, its
        .mask attribute (shape (1, H, W)) is used.  A raw tensor of compatible shape is also accepted.

    Returns
    -------
    torch.Tensor
        Frequency-constrained patch in [0, 1], same shape as patch_params.
    """
    # Resolve mask tensor
    mask_tensor = mask.mask if isinstance(mask, DCTFrequencyMask) else mask

    # Forward DCT
    coeffs = dct_2d(patch_params)

    # Element-wise masking (broadcast over batch & channels)
    coeffs_masked = coeffs * mask_tensor

    # Inverse DCT back to spatial domain
    patch = idct_2d(coeffs_masked)

    # Clamp to valid pixel range
    patch = patch.clamp(0.0, 1.0)

    return patch


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  DCT Filter - Smoke Tests")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    H, W, C = 256, 256, 3
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 1. Round-trip: IDCT2(DCT2(x)) ≈ x
    # ------------------------------------------------------------------
    print("Test 1: DCT round-trip (3-D input)")
    x_3d = torch.randn(C, H, W, device=device)
    x_rec_3d = idct_2d(dct_2d(x_3d))
    err_3d = (x_3d - x_rec_3d).abs().max().item()
    print(f"Max reconstruction error (C,H,W):    {err_3d:.2e}")
    assert err_3d < 1e-5, f"Round-trip error too large: {err_3d}"

    print("\nTest 1b: DCT round-trip (4-D batched input)")
    B = 2
    x_4d = torch.randn(B, C, H, W, device=device)
    x_rec_4d = idct_2d(dct_2d(x_4d))
    err_4d = (x_4d - x_rec_4d).abs().max().item()
    print(f"  Max reconstruction error (B,C,H,W):  {err_4d:.2e}")
    assert err_4d < 1e-5, f"Round-trip error too large: {err_4d}"
    print("  PASS")

    # ------------------------------------------------------------------
    # 2. Mask complementarity: mask_low + mask_high == mask_full
    # ------------------------------------------------------------------
    print("\nTest 2: Mask complementarity")
    m_low  = DCTFrequencyMask(H, W, "low",  device=device)
    m_high = DCTFrequencyMask(H, W, "high", device=device)
    m_full = DCTFrequencyMask(H, W, "full", device=device)

    print(f"{m_low}")
    print(f"{m_high}")
    print(f"{m_full}")

    complement = m_low.mask + m_high.mask
    assert torch.allclose(complement, m_full.mask), "Complementarity FAILED"
    print("mask_low + mask_high == mask_full | PASS")

    # ------------------------------------------------------------------
    # 3. Gradient flow through apply_dct_mask
    # ------------------------------------------------------------------
    print("\nTest 3: Gradient flow")
    params = torch.randn(B, C, H, W, device=device, requires_grad=True)
    patch = apply_dct_mask(params, m_low)
    loss = patch.sum()
    loss.backward()

    grad_norm = params.grad.norm().item()
    grad_nonzero = (params.grad != 0).sum().item()
    total_elems = params.grad.numel()

    print(f"Gradient L2 norm: {grad_norm:.4f}")
    print(f"Non-zero grad elements: {grad_nonzero} / {total_elems}")
    assert grad_nonzero > 0, "No gradients flowed!"
    print(" PASS")

    # ------------------------------------------------------------------
    # 4. Output range of apply_dct_mask is [0, 1]
    # ------------------------------------------------------------------
    print("\nTest 4: Output range check")
    patch_check = apply_dct_mask(
        torch.randn(C, H, W, device=device, requires_grad=True), m_full
    )
    print(f"min={patch_check.min().item():.4f}  max={patch_check.max().item():.4f}")
    assert patch_check.min() >= 0.0 and patch_check.max() <= 1.0, "Range violated"
    print("PASS")

    print("\n" + "=" * 72)
    print("All smoke tests passed.")
    print("=" * 72)
