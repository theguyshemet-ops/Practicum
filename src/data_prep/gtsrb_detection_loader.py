"""
GTSRB Traffic Sign Detection Dataset Loader

Loads full-frame GTSRB images with ROI bounding boxes for object detection.
Maps class labels to 1-based indexing (0 = background, 1-43 = traffic sign classes).
"""

import os
import csv
import cv2
import torch
from torch.utils.data import Dataset


class GTSRBDetectionDataset(Dataset):
    """
    PyTorch Dataset for German Traffic Sign Detection on the GTSRB dataset.
    
    Loads full images and converts the ROI bounding boxes to absolute coordinates
    scaled to the target image size (default 640x640).
    """

    # Default path relative to project root
    DEFAULT_DATA_ROOT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "gtsrb"
    )

    # 43 standard GTSRB sign classes
    CLASS_NAMES = [
        "speed_limit_20", "speed_limit_30", "speed_limit_50", "speed_limit_60", "speed_limit_70",
        "speed_limit_80", "end_of_speed_limit_80", "speed_limit_100", "speed_limit_120", "no_passing",
        "no_passing_heavy", "priority_intersection", "priority_road", "yield", "stop",
        "no_vehicles", "vehicles_prohibited_heavy", "no_entry", "general_danger", "dangerous_curve_left",
        "dangerous_curve_right", "double_curve", "bumpy_road", "slippery_road", "road_narrows_right",
        "road_works", "traffic_signals", "pedestrians", "children_crossing", "bicycles_crossing",
        "ice_snow", "wild_animals", "end_of_all_limits", "turn_right_ahead", "turn_left_ahead",
        "ahead_only", "go_straight_or_right", "go_straight_or_left", "keep_right", "keep_left",
        "roundabout_mandatory", "end_of_no_passing", "end_of_no_passing_heavy"
    ]

    def __init__(self, data_root=None, split="train", img_size=(640, 640), transform=None):
        self.split = split.lower()
        self.img_size = img_size
        self.transform = transform
        self.classes = ["background"] + self.CLASS_NAMES

        # Resolve data root
        if data_root is None:
            data_root = self.DEFAULT_DATA_ROOT
        self.data_root = data_root

        # Validate dataset paths
        csv_path = self._get_csv_path()
        if csv_path is None or not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"GTSRB CSV not found for split '{self.split}' at {self.data_root}. "
                f"Expected file: {csv_path}"
            )

        # Load samples
        print(f"[GTSRB Detection] Loading dataset ({self.split} split)...")
        self.samples = self._load_samples(csv_path)
        print(f"[GTSRB Detection] Loaded {len(self.samples)} samples.")

    def _get_csv_path(self):
        """Return the path to the CSV annotation file for the split."""
        if self.split in ("train", "training"):
            return os.path.join(self.data_root, "Train.csv")
        elif self.split in ("test", "testing", "val", "validation"):
            return os.path.join(self.data_root, "Test.csv")
        return None

    def _load_samples(self, csv_path):
        """
        Parse the CSV file.
        Format: Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
        """
        samples_list = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row["Path"].strip()
                img_path = os.path.join(self.data_root, rel_path)

                # Parse dimensions and ROI
                try:
                    w_orig = int(row["Width"])
                    h_orig = int(row["Height"])
                    roi = [
                        int(row["Roi.X1"]), int(row["Roi.Y1"]),
                        int(row["Roi.X2"]), int(row["Roi.Y2"])
                    ]
                    class_id = int(row["ClassId"])
                except (KeyError, ValueError) as e:
                    print(f"Warning: failed to parse row: {row}. Error: {e}")
                    continue

                samples_list.append({
                    "img_path": img_path,
                    "class_id": class_id,
                    "w_orig": w_orig,
                    "h_orig": h_orig,
                    "roi": roi,
                })
        return samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        # Read image
        img = cv2.imread(sample_info["img_path"])
        if img is None:
            raise IOError(f"Failed to read image: {sample_info["img_path"]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h_orig, w_orig = img.shape[:2]
        w_target, h_target = self.img_size

        # Resize image
        img_resized = cv2.resize(img, self.img_size)

        # Scale ROI bounding box coordinates
        x1_orig, y1_orig, x2_orig, y2_orig = sample_info["roi"]
        x1 = x1_orig * (w_target / w_orig)
        y1 = y1_orig * (h_target / h_orig)
        x2 = x2_orig * (w_target / w_orig)
        y2 = y2_orig * (h_target / h_orig)

        # Clamp box to target dimensions
        x1 = max(0.0, min(x1, float(w_target)))
        y1 = max(0.0, min(y1, float(h_target)))
        x2 = max(0.0, min(x2, float(w_target)))
        y2 = max(0.0, min(y2, float(h_target)))

        # Ensure coordinates are valid bounding box
        if x2 <= x1:
            x2 = min(x1 + 1.0, float(w_target))
        if y2 <= y1:
            y2 = min(y1 + 1.0, float(h_target))

        # Check if albumentations transforms are provided
        if self.transform:
            transformed = self.transform(
                image=img_resized,
                bboxes=[[x1, y1, x2, y2]],
                class_labels=[sample_info["class_id"] + 1]
            )
            img_resized = transformed["image"]
            if len(transformed["bboxes"]) > 0:
                x1, y1, x2, y2 = transformed["bboxes"][0]
                label_val = transformed["class_labels"][0]
            else:
                label_val = sample_info["class_id"] + 1
        else:
            label_val = sample_info["class_id"] + 1

        # Normalize image to [0, 1] tensor of shape (3, H, W)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # Labels are 1-based, where class 0 is reserved for background.
        bboxes_tensor = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        labels_tensor = torch.tensor([label_val], dtype=torch.long)

        return {
            "image": img_tensor,
            "bboxes": bboxes_tensor,
            "labels": labels_tensor,
            "metadata": {
                "img_path": sample_info["img_path"],
                "class_id": sample_info["class_id"],
                "original_size": (w_orig, h_orig),
                "roi": sample_info["roi"]
            }
        }


if __name__ == "__main__":
    print("=" * 60)
    print("GTSRBDetectionDataset - Verification")
    print("=" * 60)
    
    try:
        dataset = GTSRBDetectionDataset(split="train")
        print(f"Loaded {len(dataset)} training samples.")
        
        sample = dataset[0]
        print("Sample 0 verification:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Bboxes:      {sample['bboxes']}")
        print(f"  Labels:      {sample['labels']} ({dataset.classes[sample['labels'][0].item()]})")
        print(f"  Metadata:    {sample['metadata']}")
        
        dataset_test = GTSRBDetectionDataset(split="test")
        print(f"Loaded {len(dataset_test)} test samples.")
        print("Verification completed successfully!")
    except Exception as e:
        print(f"Verification failed: {e}")
    print("=" * 60)
