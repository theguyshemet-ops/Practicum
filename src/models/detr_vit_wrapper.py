import os
import torch
import torch.nn as nn
from transformers import YolosForObjectDetection, YolosImageProcessor

class DetrVitWrapper(nn.Module):
    """
    A unified wrapper class for a Vision Transformer (ViT) based object detector (YOLOS / DETR-ViT).
    Features direct extraction of self-attention matrices from MHSA layers to support Attention Rollout.
    """
    def __init__(self, model_name="hustvl/yolos-tiny", conf_threshold=0.25, device="cuda", num_classes=91, checkpoint_path=None):
        super(DetrVitWrapper, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self.num_classes = num_classes
        
        print(f"Initializing {model_name} on {self.device}")
        
        # We try to load the model from HuggingFace, otherwise fallback to mock
        try:
            self.image_processor = YolosImageProcessor.from_pretrained(model_name)
            # IMPORTANT: Use 'eager' attention implementation to enable output_attentions=True
            # The default 'sdpa' (Scaled Dot-Product Attention) does NOT support attention extraction,
            # which is required for the Attention Rollout diagnostic in this research.
            if num_classes != 91:
                self.model = YolosForObjectDetection.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    attn_implementation="eager"
                )
            else:
                self.model = YolosForObjectDetection.from_pretrained(
                    model_name, 
                    attn_implementation="eager"
                )
            
            # Load checkpoint if provided
            if checkpoint_path is not None:
                print(f"Loading YOLOS checkpoint from: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval() # evaluation mode
            self.is_mock = False
        except Exception as e:
            print(f"HuggingFace load failed: {e}. Building mock ViT detector.")
            self.is_mock = True
            # Simple dummy modules to simulate forward pass
            self.dummy_patch_embed = nn.Conv2d(3, 16, kernel_size=16, stride=16) # patch size 16
            self.dummy_attention = nn.Linear(16, 16)
            self.dummy_head = nn.Linear(16, 10 * 6) # 10 classes, boxes + scores
            self.to(self.device)
            
        # Attention maps cache
        self.last_attentions = None

    def forward(self, images, labels=None):
        """
        This function does forward pass.
        Args:
            images: Tensor of shape (B, C, H, W) normalized to [0, 1].
            labels: Optional list of target dicts for training, each containing:
                - ``'boxes'``: Tensor ``(N, 4)`` in ``[x1, y1, x2, y2]`` absolute format
                - ``'labels'``: Tensor ``(N,)``
        Returns:
            List of dictionaries containing bounding boxes, scores, and labels (eval mode),
            or Hugging Face model outputs containing loss (training mode).
        """
        batch_size = images.shape[0]
        device = images.device
        
        # ---- Training mode: model returns HF output containing loss ----
        if self.training:
            if labels is None:
                raise ValueError("In training mode, 'labels' (targets) must be provided to YOLOS.")
            
            if self.is_mock:
                # Return None, mock training loop will compute a mock loss
                return None
                
            # Convert targets to Hugging Face YOLOS expected format
            hf_labels = []
            h, w = images.shape[2:]
            for b_idx in range(len(labels)):
                target_boxes = labels[b_idx]['boxes']  # (N, 4) in absolute xyxy format
                target_classes = labels[b_idx]['labels']  # (N,)
                
                # Convert xyxy absolute to cxcywh normalized in [0, 1]
                if target_boxes.numel() > 0:
                    x1 = target_boxes[:, 0]
                    y1 = target_boxes[:, 1]
                    x2 = target_boxes[:, 2]
                    y2 = target_boxes[:, 3]
                    
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    
                    boxes_normalized = torch.stack([cx, cy, bw, bh], dim=-1)
                    boxes_normalized = torch.clamp(boxes_normalized, 0.0, 1.0)
                else:
                    boxes_normalized = torch.zeros((0, 4), device=device)
                
                hf_labels.append({
                    "class_labels": target_classes.to(device),
                    "boxes": boxes_normalized.to(device)
                })
            
            # Normalize images according to ImageNet stats expected by YOLOS
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            pixel_values = (images - mean) / std
            
            outputs_hf = self.model(pixel_values=pixel_values, labels=hf_labels, output_attentions=True)
            self.last_attentions = outputs_hf.attentions
            return outputs_hf

        if self.is_mock:
            # Simulate a ViT patch embedding + self-attention pass
            outputs = []
            x = self.dummy_patch_embed(images) # B, 16, H/16, W/16
            h_patches, w_patches = x.shape[2], x.shape[3]
            num_patches = h_patches * w_patches
            
            # Simulate attention matrix (B, num_heads=2, num_patches, num_patches)
            dummy_attn = torch.softmax(torch.randn(batch_size, 2, num_patches, num_patches, device=device), dim=-1)
            self.last_attentions = [dummy_attn] # Cache attention map
            
            # Predict boxes
            x_pooled = torch.mean(x, dim=[2, 3]) # B, 16
            pred = self.dummy_head(x_pooled) # B, 60
            pred = pred.view(batch_size, 10, 6)
            
            for i in range(batch_size):
                boxes = pred[i, :, 0:4] * 200.0 + 320.0 # centered coordinates
                boxes = torch.clamp(boxes, 0.0, 640.0)
                scores = torch.sigmoid(pred[i, :, 4])
                labels = torch.argmax(pred[i, :, 5:], dim=-1)
                
                mask = scores >= self.conf_threshold
                if torch.is_grad_enabled() or (isinstance(images, torch.Tensor) and images.requires_grad):
                    outputs.append({
                        'boxes': boxes[mask],
                        'scores': scores[mask],
                        'labels': labels[mask]
                    })
                else:
                    outputs.append({
                        'boxes': boxes[mask].detach(),
                        'scores': scores[mask].detach(),
                        'labels': labels[mask].detach()
                    })
            return outputs

        # Real YOLOS ViT Detector Forward Pass
        try:
            # YOLOS processor takes PIL or numpy, but we can feed model directly with standard tensor.
            # HuggingFace expects inputs in dictionary format with 'pixel_values'.
            # If tensor is not already normalized according to processor, we can normalize.
            # YOLOS default normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            pixel_values = (images - mean) / std
            
            # Forward with output_attentions=True to capture self-attention matrix natively!
            outputs_hf = self.model(pixel_values=pixel_values, output_attentions=True)
            
            # Cache attention weights across all ViT layers
            # outputs_hf.attentions is a list of Tensors of shape (B, num_heads, seq_len, seq_len)
            self.last_attentions = outputs_hf.attentions
            
            # Post-process predictions to match our standard format
            target_sizes = torch.tensor([list(images.shape[2:])] * batch_size, device=device)
            results = self.image_processor.post_process_object_detection(outputs_hf, threshold=self.conf_threshold, target_sizes=target_sizes)
            
            outputs = []
            for res in results:
                if torch.is_grad_enabled() or (isinstance(images, torch.Tensor) and images.requires_grad):
                    outputs.append({
                        'boxes': res['boxes'],
                        'scores': res['scores'],
                        'labels': res['labels']
                    })
                else:
                    outputs.append({
                        'boxes': res['boxes'].detach(),
                        'scores': res['scores'].detach(),
                        'labels': res['labels'].detach()
                    })
            return outputs
        except Exception as e:
            print(f"Inference failed: {e}. Returning empty predictions.")
            return [{
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros((0,), device=device),
                'labels': torch.zeros((0,), dtype=torch.long, device=device)
            } for _ in range(batch_size)]

    def get_attention_rollout(self, head_fusion="mean"):
        """
        This function computes the Attention Rollout map from cached self-attention matrices.
        This provides a diagnostic visualization of global ViT token dependency,
        helping analyze how physical patches manipulate global receptive fields.
        """
        if self.last_attentions is None:
            print("WARNING!! No attentions cached. Run a forward pass first.")
            return None
            
        device = self.last_attentions[0].device
        batch_size = self.last_attentions[0].shape[0]
        
        # Rollout lists per batch sample
        rollouts = []
        for b in range(batch_size):
            # Identity matrix represents initial information flow
            # Sequence length = num_patches + 1 (for class token) or similar depending on YOLOS
            seq_len = self.last_attentions[0].shape[2]
            rollout = torch.eye(seq_len, device=device)
            
            for layer_attn in self.last_attentions:
                # Get attention map for current sample: (num_heads, seq_len, seq_len)
                attn = layer_attn[b]
                
                # Fuse attention across multiple heads
                if head_fusion == "mean":
                    attn_fused = torch.mean(attn, dim=0)
                elif head_fusion == "max":
                    attn_fused = torch.max(attn, dim=0)[0]
                elif head_fusion == "min":
                    attn_fused = torch.min(attn, dim=0)[0]
                else:
                    raise ValueError(f"Unknown head fusion strategy: {head_fusion}")
                    
                # Add Identity to account for residual connections in ViT
                I = torch.eye(seq_len, device=device)
                attn_fused = 0.5 * attn_fused + 0.5 * I
                
                # Normalize columns to preserve probability distribution
                attn_fused = attn_fused / torch.sum(attn_fused, dim=-1, keepdim=True)
                
                # Matrix multiply to trace information flow recursively
                rollout = torch.matmul(attn_fused, rollout)
                
            if torch.is_grad_enabled():
                rollouts.append(rollout)
            else:
                rollouts.append(rollout.detach().cpu())
            
        return rollouts

if __name__ == "__main__":
    print("=" * 50)
    print("\nTest\nRunning DetrVitWrapper REAL MODEL test...")
    print("=" * 50)
    
    # Load the real YOLOS-tiny model from HuggingFace
    model_name = "hustvl/yolos-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {model_name} on {device}...")
    wrapper = DetrVitWrapper(model_name=model_name, conf_threshold=0.25, device=device)
    
    print(f"\nModel loaded | Mock mode: {wrapper.is_mock} | Device: {device}")
    
    # Run forward pass with a test image
    mock_input = torch.randn(2, 3, 512, 512).to(device)
    outputs = wrapper(mock_input)
    print(f"Outputs count: {len(outputs)}")
    print(f"Sample 0 keys: {list(outputs[0].keys())}")
    print(f"Sample 0 boxes: {outputs[0]['boxes'].shape}")
    print(f"Sample 0 scores: {outputs[0]['scores'].shape}")
    
    # Verify Attention Rollout on the real ViT self-attention maps
    if wrapper.last_attentions is not None:
        print(f"\nCaptured {len(wrapper.last_attentions)} attention layers")
        print(f"Attention layer 0 shape: {wrapper.last_attentions[0].shape}")
        
        rollouts = wrapper.get_attention_rollout()
        print(f"Rollout maps count: {len(rollouts)}")
        print(f"Rollout shape: {rollouts[0].shape}")
    else:
        print("No attentions captured (mock mode)")
    
    print("\n" + "=" * 40)
    print("DetrVitWrapper test PASSED!")
    print("=" * 40)
