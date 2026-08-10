"""
Adversarial Loss Script
Dual-objective loss function for adversarial patch optimisation.

Objective 1 - Detection Suppression (both Faster R-CNN and YOLOS):
    L_suppress = Σ max(0, s_i - τ)
    Pushes all detection confidence scores below threshold τ.

Objective 2 - ViT Attention Disruption (YOLOS only):
    L_attention = KL(A_clean || A_patched) + KL(A_patched || A_clean)
    Symmetric KL-divergence between clean and patched attention rollout maps.
    Forces the patch to maximally distort the ViT's global attention graph.

Combined loss:
    L_total = α · L_suppress_rcnn + β · L_suppress_vit + γ · L_attention_vit

Design choice (from implementation plan):
    The hinge-based formulation is
    numerically stable and penalises ALL confident detections.

References:
    - Eykholt et al. (2018) - RP2 adversarial patch loss
    - Fu et al. (2022) - Patch-Fool attention manipulation
    - Park & Kim (2022) - MHSA low-pass filtering behaviour
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionSuppressionLoss(nn.Module):
    """
    Hinge-based detection suppression loss.

    For each detection with confidence score s_i:
        L = Σ max(0, s_i - τ)

    This penalises all detections above the confidence threshold τ,
    driving the model to produce no confident predictions in patched regions.

    Args:
        tau (float): Confidence threshold below which detections are
                     considered suppressed. Default 0.1 (aggressive suppression).
    """

    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, detections):
        """
        Compute suppression loss from model detection outputs.

        Args:
            detections: List[Dict] from model wrapper, each containing:
                - scores: Tensor (N,) - detection confidence scores

        Returns:
            loss: Scalar tensor (differentiable)
        """
        total_loss = torch.tensor(0.0, requires_grad=True)
        device = None

        for det in detections:
            scores = det['scores']
            if scores.numel() == 0:
                continue

            if device is None:
                device = scores.device
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Hinge loss: penalise scores above threshold
            hinge = F.relu(scores - self.tau)
            total_loss = total_loss + hinge.sum()

        return total_loss


class AttentionDisruptionLoss(nn.Module):
    """
    Symmetric KL-divergence loss between clean and patched attention rollout maps.

    Measures how much the adversarial patch distorts the ViT's global
    attention pattern. Higher KL-divergence = more disruption = better attack.

    We MAXIMISE disruption, so the loss is negated (minimising negative KL
    = maximising KL).

    Mathematical formulation:
        L_attn = -[ KL(A_clean || A_patched) + KL(A_patched || A_clean) ]

    The symmetric form ensures the loss captures both directions of
    distributional shift, avoiding edge cases where one-sided KL is
    small despite significant attention redistribution.

    Args:
        eps (float): Small constant for numerical stability in log operations.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, clean_rollout, patched_rollout):
        """
        Compute symmetric KL-divergence between attention rollout maps.

        Args:
            clean_rollout: List[Tensor] - rollout maps from clean images.
                           Each tensor shape: (seq_len, seq_len)
            patched_rollout: List[Tensor] - rollout maps from patched images.
                             Same shape as clean_rollout.

        Returns:
            loss: Scalar tensor (differentiable). Negative symmetric KL.
        """
        if clean_rollout is None or patched_rollout is None:
            return torch.tensor(0.0, requires_grad=True)

        total_kl = torch.tensor(0.0, requires_grad=True)
        count = 0

        for clean_map, patched_map in zip(clean_rollout, patched_rollout):
            # Ensure both are on the same device
            device = patched_map.device
            clean_map = clean_map.to(device)

            if total_kl.device != device:
                total_kl = torch.tensor(0.0, device=device, requires_grad=True)

            # Normalise rows to valid probability distributions
            clean_prob = clean_map / (clean_map.sum(dim=-1, keepdim=True) + self.eps)
            patched_prob = patched_map / (patched_map.sum(dim=-1, keepdim=True) + self.eps)

            # Clamp for numerical stability in log
            clean_prob = clean_prob.clamp(min=self.eps)
            patched_prob = patched_prob.clamp(min=self.eps)

            # KL(clean || patched) = Σ clean * log(clean / patched)
            kl_forward = (clean_prob * (clean_prob.log() - patched_prob.log())).sum()

            # KL(patched || clean) = Σ patched * log(patched / clean)
            kl_backward = (patched_prob * (patched_prob.log() - clean_prob.log())).sum()

            # Symmetric KL
            symmetric_kl = kl_forward + kl_backward
            total_kl = total_kl + symmetric_kl
            count += 1

        if count > 0:
            total_kl = total_kl / count

        # Negate: minimising this loss = maximising attention disruption
        return -total_kl


class DualObjectiveLoss(nn.Module):
    """
    Combined adversarial loss for dual-model patch optimisation.

    L_total = α · L_suppress_rcnn + β · L_suppress_vit + γ · L_attention_vit

    The three components target different aspects of detection robustness:
        1. L_suppress_rcnn - suppress Faster R-CNN detections (CNN pathway)
        2. L_suppress_vit  - suppress YOLOS detections (ViT pathway)
        3. L_attention_vit  - disrupt YOLOS attention graph (ViT-specific)

    Args:
        alpha (float): Weight for Faster R-CNN suppression loss. Default 1.0.
        beta (float):  Weight for YOLOS suppression loss. Default 1.0.
        gamma (float): Weight for attention disruption loss. Default 0.3.
                       Lower because KL operates on normalised distributions
                       (smaller gradients than detection scores).
        tau (float):   Confidence threshold for suppression. Default 0.1.
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.3, tau=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.suppress_loss = DetectionSuppressionLoss(tau=tau)
        self.attention_loss = AttentionDisruptionLoss()

    def forward(self, rcnn_detections, vit_detections,
                clean_rollout=None, patched_rollout=None):
        """
        Compute the combined dual-objective loss.

        Args:
            rcnn_detections: List[Dict] - Faster R-CNN outputs.
                             Each dict has 'boxes', 'scores', 'labels'.
            vit_detections:  List[Dict] - YOLOS outputs. Same format.
            clean_rollout:   List[Tensor] - Attention rollout from clean images.
                             Optional; if None, attention loss is skipped.
            patched_rollout: List[Tensor] - Attention rollout from patched images.
                             Optional; if None, attention loss is skipped.

        Returns:
            total_loss: Scalar tensor (differentiable)
            loss_dict:  Dict with individual loss components for logging:
                        {'suppress_rcnn', 'suppress_vit', 'attention_vit', 'total'}
        """
        # Detection suppression losses
        l_suppress_rcnn = self.suppress_loss(rcnn_detections)
        l_suppress_vit = self.suppress_loss(vit_detections)

        # Attention disruption loss (ViT-specific)
        l_attention = self.attention_loss(clean_rollout, patched_rollout)

        # Combined weighted loss
        total = (self.alpha * l_suppress_rcnn
                 + self.beta * l_suppress_vit
                 + self.gamma * l_attention)

        loss_dict = {
            'suppress_rcnn': l_suppress_rcnn.item(),
            'suppress_vit': l_suppress_vit.item(),
            'attention_vit': l_attention.item(),
            'total': total.item(),
        }

        return total, loss_dict


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Adversarial Loss Module - Smoke Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Test 1: Detection Suppression Loss ---
    print("\nDetection Suppression Loss")
    suppress = DetectionSuppressionLoss(tau=0.1)

    # Simulated detections with gradients
    scores_1 = torch.tensor([0.95, 0.80, 0.05, 0.30], device=device, requires_grad=True)
    scores_2 = torch.tensor([0.60, 0.15], device=device, requires_grad=True)
    detections = [
        {'scores': scores_1, 'boxes': torch.zeros(4, 4, device=device), 'labels': torch.zeros(4, device=device, dtype=torch.long)},
        {'scores': scores_2, 'boxes': torch.zeros(2, 4, device=device), 'labels': torch.zeros(2, device=device, dtype=torch.long)},
    ]

    loss = suppress(detections)
    loss.backward()
    print(f"Suppression loss: {loss.item():.4f}")
    print(f"Expected: max(0, 0.95-0.1) + max(0, 0.80-0.1) + max(0, 0.05-0.1) + max(0, 0.30-0.1) + max(0, 0.60-0.1) + max(0, 0.15-0.1)")
    expected = (0.85 + 0.70 + 0.0 + 0.20) + (0.50 + 0.05)
    print(f"Expected value: {expected:.4f}")
    print(f"Match: {abs(loss.item() - expected) < 1e-4}")
    print(f"scores_1.grad: {scores_1.grad}")
    print(f"scores_2.grad: {scores_2.grad}")
    assert scores_1.grad is not None, "Gradient must flow to scores_1"
    assert scores_2.grad is not None, "Gradient must flow to scores_2"
    print("Detection suppression loss PASSED")

    # --- Test 2: Attention Disruption Loss ---
    print("\nAttention Disruption Loss")
    attn_loss = AttentionDisruptionLoss()

    seq_len = 100
    clean_rollout = [torch.softmax(torch.randn(seq_len, seq_len, device=device), dim=-1)]
    patched_tensor = torch.randn(seq_len, seq_len, device=device, requires_grad=True)
    patched_rollout = [torch.softmax(patched_tensor, dim=-1)]

    loss = attn_loss(clean_rollout, patched_rollout)
    loss.backward()
    print(f"Attention loss (negated KL): {loss.item():.4f}")
    print(f"Loss should be negative (minimising = maximising disruption): {loss.item() < 0}")
    assert patched_tensor.grad is not None, "Gradient must flow to patched rollout input tensor"
    print(f"patched_tensor grad norm: {patched_tensor.grad.norm().item():.6f}")
    print("Attention disruption loss PASSED")

    # --- Test 3: None rollout handling ---
    print("\n[Test 3] None rollout handling")
    loss_none = attn_loss(None, None)
    print(f"Loss with None rollouts: {loss_none.item():.4f} (should be 0.0)")
    assert loss_none.item() == 0.0, "None rollouts should produce zero loss"
    print("None rollout handling PASSED")

    # --- Test 4: Combined Dual-Objective Loss ---
    print("\nDual-Objective Loss")
    dual_loss = DualObjectiveLoss(alpha=1.0, beta=1.0, gamma=0.3, tau=0.1)

    # Fresh detections with grad
    rcnn_scores = torch.tensor([0.9, 0.7], device=device, requires_grad=True)
    vit_scores = torch.tensor([0.8, 0.6], device=device, requires_grad=True)

    rcnn_det = [{'scores': rcnn_scores, 'boxes': torch.zeros(2, 4, device=device), 'labels': torch.zeros(2, device=device, dtype=torch.long)}]
    vit_det = [{'scores': vit_scores, 'boxes': torch.zeros(2, 4, device=device), 'labels': torch.zeros(2, device=device, dtype=torch.long)}]

    clean_r = [torch.softmax(torch.randn(50, 50, device=device), dim=-1)]
    patched_r_tensor = torch.randn(50, 50, device=device, requires_grad=True)
    patched_r = [torch.softmax(patched_r_tensor, dim=-1)]

    total, loss_dict = dual_loss(rcnn_det, vit_det, clean_r, patched_r)
    total.backward()

    print(f"Total loss: {total.item():.4f}")
    print(f"Components: {loss_dict}")
    print(f"rcnn_scores.grad: {rcnn_scores.grad}")
    print(f"vit_scores.grad: {vit_scores.grad}")
    assert rcnn_scores.grad is not None, "Gradient must flow to RCNN scores"
    assert vit_scores.grad is not None, "Gradient must flow to ViT scores"
    print("Dual-objective loss PASSED")

    # --- Test 5: Empty detections ---
    print("\n[Test 5] Empty detections")
    empty_det = [{'scores': torch.zeros(0, device=device), 'boxes': torch.zeros(0, 4, device=device), 'labels': torch.zeros(0, device=device, dtype=torch.long)}]
    total_empty, dict_empty = dual_loss(empty_det, empty_det, None, None)
    print(f"Total loss (empty): {total_empty.item():.4f} (should be 0.0)")
    assert total_empty.item() == 0.0, "Empty detections should produce zero loss"
    print("Empty detections PASSED")

    print("\n" + "=" * 60)
    print("All adversarial loss tests PASSED!")
    print("=" * 60)
