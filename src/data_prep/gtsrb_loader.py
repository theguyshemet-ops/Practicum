"""
German Traffic Sign Recognition Benchmark (GTSRB) Dataset Loader

Loads the real GTSRB dataset with ROI-cropped sign images for classification.
Used in this research as the target object set for adversarial patch application.

Required directory structure:
    data/gtsrb/
    ├-- Train.csv           # Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
    ├-- Test.csv            # Same format
    ├-- Train/
    |   ├-- 0/              # Class folders 0-42 containing .png images
    |   ├-- 1/
    |   └-- ...
    └-- Test/               # Flat folder of .png test images
"""

import os
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class GTSRBDataset(Dataset):
    """
    PyTorch Dataset for the German Traffic Sign Recognition Benchmark.
    
    Loads real traffic sign images from the GTSRB dataset, crops them to
    the annotated Region of Interest (ROI), and resizes to a uniform resolution.
    
    Args:
        data_root (str): Path to GTSRB root directory. Defaults to data/gtsrb
                         relative to the project root.
        split (str): 'train' or 'test' (also accepts 'val'/'validation' -> maps to test).
        img_size (tuple): Output image resolution (W, H). Default: (128, 128).
        transform: Optional albumentations transform pipeline.
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
    
    def __init__(self, data_root=None, split="train", img_size=(128, 128), transform=None):
        self.split = split.lower()
        self.img_size = img_size
        self.transform = transform
        self.classes = self.CLASS_NAMES
        
        # Resolve data root
        if data_root is None:
            data_root = self.DEFAULT_DATA_ROOT
        self.data_root = data_root
        
        # Validate dataset exists
        csv_path = self._get_csv_path()
        if csv_path is None or not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"GTSRB CSV not found for split '{self.split}' at {self.data_root}. "
                f"Expected file: {csv_path}"
            )
        
        # Load samples
        print(f"[GTSRB Loader] Loading real GTSRB dataset ({self.split} split)...")
        self.samples = self._load_samples(csv_path)
        num_classes = len(set(s['label'] for s in self.samples))
        print(f"[GTSRB Loader] Loaded {len(self.samples)} samples across {num_classes} classes.")
    
    def _get_csv_path(self):
        """Return the path to the CSV annotation file for the current split."""
        if self.split == "train":
            return os.path.join(self.data_root, "Train.csv")
        elif self.split in ("test", "val", "validation"):
            return os.path.join(self.data_root, "Test.csv")
        return None
    
    def _load_samples(self, csv_path):
        """
        Parse the root-level CSV annotation file.
        
        CSV columns: Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path
        The 'Path' column contains relative paths like 'Train/20/00020_00000_00000.png'.
        """
        samples_list = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row['Path'].strip()
                img_path = os.path.join(self.data_root, rel_path)
                
                # Parse ROI bounding box
                try:
                    roi = [
                        int(row['Roi.X1']), int(row['Roi.Y1']),
                        int(row['Roi.X2']), int(row['Roi.Y2'])
                    ]
                except (KeyError, ValueError):
                    roi = None
                
                samples_list.append({
                    'img_path': img_path,
                    'label': int(row['ClassId']),
                    'roi': roi,
                })
        
        if len(samples_list) == 0:
            raise ValueError(f"No samples parsed from {csv_path}")
            
        return samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load real image
        img = cv2.imread(sample_info['img_path'])
        if img is None:
            raise IOError(f"Failed to read image: {sample_info['img_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop to ROI bounding box (extracts just the sign face)
        roi = sample_info.get('roi')
        if roi is not None:
            x1, y1, x2, y2 = roi
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                img = img[y1:y2, x1:x2]
        
        # Resize to target resolution
        img_resized = cv2.resize(img, self.img_size)
        
        # Apply optional transform
        if self.transform:
            transformed = self.transform(image=img_resized)
            img_resized = transformed['image']
            
        # Convert to tensor: (C, H, W), normalised to [0, 1]
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': img_tensor,
            'label': torch.tensor(sample_info['label'], dtype=torch.long)
        }
    
    def get_class_distribution(self):
        """Return a dictionary mapping class_id -> sample count."""
        dist = {}
        for s in self.samples:
            lbl = s['label']
            dist[lbl] = dist.get(lbl, 0) + 1
        return dict(sorted(dist.items()))


if __name__ == "__main__":
    print("=" * 60)
    print("GTSRBDataset - Real Data Verification")
    print("=" * 60)
    
    # Test training split
    dataset_train = GTSRBDataset(split="train", img_size=(128, 128))
    print(f"\n[TRAIN] Samples: {len(dataset_train)}")
    
    dist = dataset_train.get_class_distribution()
    print(f"[TRAIN] Classes: {len(dist)}")
    print(f"[TRAIN] Min/Max samples per class: {min(dist.values())} / {max(dist.values())}")
    
    sample = dataset_train[0]
    print(f"[TRAIN] Image shape: {sample['image'].shape}")
    print(f"[TRAIN] Label: {sample['label'].item()} ({dataset_train.classes[sample['label'].item()]})")
    print(f"[TRAIN] Pixel range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    
    # Test test split
    dataset_test = GTSRBDataset(split="test", img_size=(128, 128))
    print(f"\n[TEST] Samples: {len(dataset_test)}")
    
    sample_t = dataset_test[0]
    print(f"[TEST] Image shape: {sample_t['image'].shape}")
    print(f"[TEST] Label: {sample_t['label'].item()} ({dataset_test.classes[sample_t['label'].item()]})")
    
    print("\n" + "=" * 60)
    print("GTSRB loader verification PASSED")
    print("=" * 60)
