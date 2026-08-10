"""
Faster R-CNN Wrapper Module
fasterrcnn_resnet50_fpn_v2 (COCO-pretrained, 91 classes).

Provides the same output interface:
    List[Dict] with keys 'boxes' (N,4), 'scores' (N,), 'labels' (N,)

Includes Grad-CAM hook infrastructure for interpretability diagnostics.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


class FasterRCNNWrapper(nn.Module):
    """
    A PyTorch wrapper for Faster R-CNN (ResNet-50-FPN v2), providing
    standardised object detection outputs and Grad-CAM hook support.

    Output interface matches YOLOv10Wrapper exactly:
        List[Dict] with keys 'boxes', 'scores', 'labels'
    """

    # Full 91-class COCO label list (index 0 = __background__)
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
        'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush',
    ]

    def __init__(self, num_classes=91, conf_threshold=0.25, device='cuda', checkpoint_path=None):
        super(FasterRCNNWrapper, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.num_classes = num_classes

        print(f"Initializing Faster R-CNN (ResNet-50-FPN v2) on {self.device}")

        # Load COCO-pretrained Faster R-CNN v2
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)

        # Replace head if num_classes is different from COCO 91
        if num_classes != 91:
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.to(self.device)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            print(f"Loading Faster R-CNN checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)

        # Hook containers for Grad-CAM
        self.gradients = None
        self.features = None
        self.hook_handles = []

        print(f"Faster R-CNN loaded successfully | {num_classes} classes | device={self.device}")

    # ------------------------------------------------------------------
    # Grad-CAM hook infrastructure
    # ------------------------------------------------------------------

    def _save_gradient(self, grad):
        """Backward hook callback - stores gradient for Grad-CAM."""
        self.gradients = grad

    def _forward_hook(self, module, input, output):
        """
        Forward hook callback - captures feature maps from the target layer.

        Safely handles tuple/list outputs by extracting the first tensor,
        which is common for residual blocks and FPN layers.
        """
        if isinstance(output, tuple):
            feat = output[0]
        elif isinstance(output, list):
            feat = output[0]
        else:
            feat = output

        self.features = feat

        # Register backward hook on the tensor to capture gradients
        # (only possible when the tensor participates in the computation graph)
        if isinstance(feat, torch.Tensor) and feat.requires_grad:
            feat.register_hook(self._save_gradient)

    def register_gradcam_hooks(self):
        """
        Register a forward hook on the last layer of
        ``self.model.backbone.body.layer4`` (the final ResNet-50 conv block).

        This is the standard target for Grad-CAM on ResNet-based detectors:
        it captures the highest-level spatial feature maps before FPN.
        """
        # backbone.body is the ResNet trunk; layer4 is a nn.Sequential of
        # Bottleneck blocks. We hook the *last* Bottleneck block.
        target_layer = self.model.backbone.body.layer4[-1]

        handle = target_layer.register_forward_hook(self._forward_hook)
        self.hook_handles.append(handle)
        print(
            f"Grad-CAM hook registered on backbone.body.layer4[-1] "
            f"({target_layer.__class__.__name__})"
        )

    def remove_hooks(self):
        """Remove all registered hooks and clear cached tensors."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.gradients = None
        self.features = None
        print("Hook handles removed successfully.")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, images, targets=None):
        """
        Standardised forward pass.

        Args:
            images: Tensor of shape ``(B, C, H, W)`` normalised to [0, 1].
            targets: Optional list of dicts (one per image), each containing:
                - ``'boxes'``: Tensor ``(N, 4)``
                - ``'labels'``: Tensor ``(N,)``

        Returns:
            **eval mode** - ``List[Dict]`` (one per image), each containing:
                - ``'boxes'``:  Tensor ``(N, 4)`` in ``[x1, y1, x2, y2]``
                - ``'scores'``: Tensor ``(N,)``
                - ``'labels'``: Tensor ``(N,)`` (int64)

            **training mode** - ``Dict[str, Tensor]`` loss dictionary as
            returned by the underlying Faster R-CNN model.
        """
        device = images.device

        # ---- torchvision Faster R-CNN expects a LIST of (C,H,W) tensors ----
        image_list = [images[i] for i in range(images.shape[0])]

        # ---- Training mode: model returns loss dict ----
        if self.training:
            if targets is None:
                raise ValueError("In training mode, 'targets' must be provided to Faster R-CNN.")
            loss_dict = self.model(image_list, targets)
            return loss_dict

        # ---- Eval mode: model returns List[Dict] with detections ----
        raw_outputs = self.model(image_list)

        outputs = []
        for det in raw_outputs:
            boxes = det['boxes']
            scores = det['scores']
            labels = det['labels']

            # Filter by confidence threshold
            mask = scores >= self.conf_threshold
            outputs.append({
                'boxes': boxes[mask].to(device),
                'scores': scores[mask].to(device),
                'labels': labels[mask].to(device),
            })

        return outputs


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Faster R-CNN Wrapper - Smoke Test (REAL MODEL)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # 1. Load model
    wrapper = FasterRCNNWrapper(num_classes=91, conf_threshold=0.25, device=device)
    wrapper.eval()
    print(f"Model loaded on {device} - eval mode")

    # 2. Register Grad-CAM hooks
    wrapper.register_gradcam_hooks()

    # 3. Forward pass with random input (batch of 2, 640x640)
    print("\nRunning forward pass with random (2, 3, 640, 640) input...")
    dummy_input = torch.rand(2, 3, 640, 640, device=device)
    with torch.no_grad():
        outputs = wrapper(dummy_input)

    print(f"Number of outputs: {len(outputs)}")
    for idx, out in enumerate(outputs):
        print(
            f"  Image {idx}: "
            f"boxes={out['boxes'].shape}, "
            f"scores={out['scores'].shape}, "
            f"labels={out['labels'].shape}"
        )

    # Verify output interface matches YOLOv10Wrapper
    assert isinstance(outputs, list), "Output must be a list"
    for out in outputs:
        assert set(out.keys()) == {'boxes', 'scores', 'labels'}, \
            f"Unexpected keys: {out.keys()}"
        assert out['boxes'].ndim == 2 and out['boxes'].shape[1] == 4, \
            f"boxes shape mismatch: {out['boxes'].shape}"
        assert out['scores'].ndim == 1, f"scores shape mismatch: {out['scores'].shape}"
        assert out['labels'].ndim == 1, f"labels shape mismatch: {out['labels'].shape}"

    # 4. Verify Grad-CAM hooks captured features
    if wrapper.features is not None:
        print(f"\nGrad-CAM feature map shape: {wrapper.features.shape}")
    else:
        print("\nGrad-CAM features: None (expected under torch.no_grad())")

    # 5. Test with gradients enabled to verify backward hook wiring
    print("\nRunning forward pass WITH gradients for Grad-CAM verification...")
    wrapper.features = None
    wrapper.gradients = None

    grad_input = torch.rand(1, 3, 640, 640, device=device)
    # Temporarily allow gradients through the model
    detections = wrapper(grad_input)

    if wrapper.features is not None:
        print(f"Grad-CAM feature map shape (grad-enabled): {wrapper.features.shape}")
        if wrapper.features.requires_grad:
            # Drive a backward pass to verify gradient capture
            target_score = detections[0]['scores']
            if len(target_score) > 0:
                target_score[0].backward(retain_graph=True)
                if wrapper.gradients is not None:
                    print(f"Grad-CAM gradient shape: {wrapper.gradients.shape}")
                else:
                    print("Grad-CAM gradient: None (backward hook did not fire)")
            else:
                print("No detections above threshold - skipping backward test")
        else:
            print("Features do not require grad - backward hook not applicable")
    else:
        print("Grad-CAM features: None")

    # 6. Verify COCO class names
    print(f"\nCOCO_CLASSES count: {len(FasterRCNNWrapper.COCO_CLASSES)}")
    assert len(FasterRCNNWrapper.COCO_CLASSES) == 91, \
        f"Expected 91 COCO classes, got {len(FasterRCNNWrapper.COCO_CLASSES)}"
    print(f"First 5 classes: {FasterRCNNWrapper.COCO_CLASSES[:5]}")

    # 7. Clean up
    wrapper.remove_hooks()

    print("\n" + "=" * 60)
    print("Faster R-CNN Wrapper smoke test PASSED!")
    print("=" * 60)
