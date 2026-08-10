"""
nuScenes-mini 2D Object Detection Dataset Loader

Parses the real nuScenes-mini dataset, projects 3D bounding box annotations
into 2D camera frames, and provides a standard PyTorch Dataset interface.

Required directory structure:
    data/nuscenes/
    ├-- v1.0-mini/          # Metadata JSON files (sample.json, scene.json, etc.)
    ├-- samples/            # Camera images by sensor (CAM_FRONT, CAM_BACK, etc.)
    ├-- sweeps/             # Intermediate sweep frames
    └-- maps/               # Map data

Dependencies:
    - nuscenes-devkit (pip install nuscenes-devkit)
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


class NuScenes2DDataset(Dataset):
    """
    PyTorch Dataset for nuScenes-mini with 3D->2D bounding box projection.
    
    Loads real camera images from the nuScenes-mini dataset and projects
    3D bounding box annotations onto the 2D image plane using the 
    official camera intrinsic matrices and ego-pose transformations.
    
    Args:
        data_root (str): Path to the nuScenes dataset root. Defaults to data/nuscenes
                         relative to the project root.
        version (str): Dataset version string. Default: 'v1.0-mini'.
        split (str): 'train' (first 8 scenes) or 'val' (last 2 scenes).
        camera_name (str): Camera sensor to use. Default: 'CAM_FRONT'.
        img_size (tuple): Output image resolution (W, H). Default: (640, 640).
        transform: Optional albumentations transform pipeline.
    """
    
    # Default path relative to project root
    DEFAULT_DATA_ROOT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "nuscenes"
    )
    
    # 10 object classes used in this research (mapped from nuScenes fine-grained categories)
    CLASSES = [
        'car', 'pedestrian', 'truck', 'bus', 'trailer',
        'construction_vehicle', 'motorcycle', 'bicycle', 'barrier', 'traffic_cone'
    ]
    
    def __init__(self, data_root=None, version="v1.0-mini", split="train",
                 camera_name="CAM_FRONT", img_size=(640, 640), transform=None):
        self.version = version
        self.split = split
        self.camera_name = camera_name
        self.img_size = img_size
        self.transform = transform
        
        # Resolve data root
        if data_root is None:
            data_root = self.DEFAULT_DATA_ROOT
        self.data_root = data_root
        
        # Build class mapping
        self.classes = self.CLASSES
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Validate dataset exists
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"nuScenes data root not found: {self.data_root}")
        version_dir = os.path.join(self.data_root, version)
        if not os.path.exists(version_dir):
            raise FileNotFoundError(f"nuScenes version folder not found: {version_dir}")
        
        # Initialise nuScenes API
        print(f"[NuScenes Loader] Loading nuScenes-mini ({self.version}) from: {self.data_root}")
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        self.samples = self._load_samples()
        print(f"[NuScenes Loader] Loaded {len(self.samples)} {self.split} samples "
              f"({self.camera_name}, {len(self.nusc.scene)} total scenes).")

    def _load_samples(self):
        """Parse scenes and camera tokens from the nuScenes database, filtered by split."""
        all_scenes = self.nusc.scene
        
        # Split: first 8 scenes for training, last 2 for validation
        if self.split == "train":
            scenes = all_scenes[:8]
        else:
            scenes = all_scenes[8:]
            
        scene_tokens = {s['token'] for s in scenes}
        
        samples_list = []
        for sample in self.nusc.sample:
            if sample['scene_token'] not in scene_tokens:
                continue
                
            cam_token = sample['data'][self.camera_name]
            cam_data = self.nusc.get('sample_data', cam_token)
            
            samples_list.append({
                'cam_token': cam_token,
                'img_path': os.path.join(self.data_root, cam_data['filename']),
                'sample_token': sample['token']
            })
            
        return samples_list

    def _project_3d_to_2d(self, cam_token):
        """
        Project nuScenes 3D bounding boxes to 2D image coordinates.
        
        Uses the camera intrinsic matrix to project the 8 corners of each
        3D bounding box onto the image plane, then computes the enclosing
        2D bounding box in pixel coordinates.
        
        Args:
            cam_token: Camera sample_data token.
            
        Returns:
            List of dicts with keys 'class', 'class_id', 'box_2d'.
        """
        cam_data = self.nusc.get('sample_data', cam_token)
        im_size = (cam_data['width'], cam_data['height'])
        
        # Get projected 3D boxes using the official nuScenes API
        _, boxes, camera_intrinsic = self.nusc.get_sample_data(cam_token)
        
        objects_2d = []
        for box in boxes:
            # Map nuScenes fine-grained categories to our 10 research classes
            raw_name = self._map_category(box.name)
            if raw_name is None:
                continue
            
            # Project 3D box corners (3x8 matrix) to image plane
            corners_3d = box.corners()
            
            # Skip boxes with any corner behind the camera
            if np.any(corners_3d[2, :] <= 0):
                continue
            
            # Perspective projection: (3x3) @ (3x8) -> (3x8), then normalise by depth
            corners_img = camera_intrinsic @ corners_3d
            corners_img = corners_img / corners_img[2, :]
            
            # Compute enclosing 2D bounding box
            x_min = np.clip(np.min(corners_img[0, :]), 0, im_size[0] - 1)
            y_min = np.clip(np.min(corners_img[1, :]), 0, im_size[1] - 1)
            x_max = np.clip(np.max(corners_img[0, :]), 0, im_size[0] - 1)
            y_max = np.clip(np.max(corners_img[1, :]), 0, im_size[1] - 1)
            
            # Filter out boxes that are too small to be meaningful
            if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                objects_2d.append({
                    'class': raw_name,
                    'class_id': self.class_to_id[raw_name],
                    'box_2d': [x_min, y_min, x_max, y_max]
                })
                
        return objects_2d

    def _map_category(self, category_name):
        """
        Map nuScenes fine-grained category names to our 10 research classes.
        
        nuScenes uses hierarchical names like 'vehicle.car', 'human.pedestrian.adult'.
        This method maps them to the simplified class set used in this research.
        """
        name_lower = category_name.lower()
        
        # Direct second-level match (e.g., 'vehicle.car' -> 'car')
        parts = category_name.split('.')
        if len(parts) > 1 and parts[1] in self.class_to_id:
            return parts[1]
        
        # Keyword-based fallback mapping
        mapping = {
            'pedestrian': 'pedestrian',
            'car': 'car',
            'truck': 'truck',
            'bus': 'bus',
            'trailer': 'trailer',
            'construction': 'construction_vehicle',
            'motorcycle': 'motorcycle',
            'bicycle': 'bicycle',
            'cone': 'traffic_cone',
            'barrier': 'barrier',
        }
        
        for keyword, cls in mapping.items():
            if keyword in name_lower:
                return cls
        
        return None  # Skip unmapped categories

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load real image
        img_path = sample_info['img_path']
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Project 3D annotations to 2D
        objects = self._project_3d_to_2d(sample_info['cam_token'])
        
        orig_h, orig_w = img.shape[:2]
        
        # Apply transform pipeline or default resize
        if self.transform:
            transformed = self.transform(
                image=img,
                bboxes=[obj['box_2d'] for obj in objects],
                class_labels=[obj['class_id'] for obj in objects]
            )
            img = transformed['image']
            boxes = np.array(transformed['bboxes']) if transformed['bboxes'] else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(transformed['class_labels']) if transformed['class_labels'] else np.zeros((0,), dtype=np.int64)
        else:
            # Resize image to target resolution
            img = cv2.resize(img, self.img_size)
            
            # Scale bounding boxes proportionally
            scale_x = self.img_size[0] / orig_w
            scale_y = self.img_size[1] / orig_h
            
            boxes = []
            labels = []
            for obj in objects:
                x1, y1, x2, y2 = obj['box_2d']
                boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
                labels.append(obj['class_id'])
                
            boxes = np.array(boxes) if boxes else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(labels) if labels else np.zeros((0,), dtype=np.int64)
            
        # Convert to tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': img_tensor,
            'bboxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'original_size': torch.tensor([orig_h, orig_w], dtype=torch.long)
        }

    def get_scene_stats(self):
        """Return summary statistics about the loaded dataset."""
        return {
            "version": self.version,
            "split": self.split,
            "camera": self.camera_name,
            "num_samples": len(self.samples),
        }


if __name__ == "__main__":
    print("=" * 60)
    print("NuScenes2DDataset - Real Data Verification")
    print("=" * 60)
    
    # Load training split
    dataset_train = NuScenes2DDataset(split="train", img_size=(640, 640))
    print(f"\n[TRAIN] Samples: {len(dataset_train)}")
    
    sample = dataset_train[0]
    print(f"[TRAIN] Image shape: {sample['image'].shape}")
    print(f"[TRAIN] Pixel range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"[TRAIN] Objects: {len(sample['bboxes'])}")
    if len(sample['bboxes']) > 0:
        print(f"[TRAIN] First box: {sample['bboxes'][0].tolist()}")
        print(f"[TRAIN] First label: {sample['labels'][0].item()} ({dataset_train.classes[sample['labels'][0].item()]})")
    
    # Load validation split
    dataset_val = NuScenes2DDataset(split="val", img_size=(640, 640))
    print(f"\n[VAL] Samples: {len(dataset_val)}")
    
    # Class distribution over first 10 training samples
    total_objs = 0
    class_counts = {}
    for i in range(min(10, len(dataset_train))):
        s = dataset_train[i]
        total_objs += len(s['bboxes'])
        for lbl in s['labels']:
            cls_name = dataset_train.classes[lbl.item()]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    print(f"\n[STATS] Objects in first 10 frames: {total_objs}")
    print(f"[STATS] Avg per frame: {total_objs / 10:.1f}")
    print(f"[STATS] Distribution: {dict(sorted(class_counts.items(), key=lambda x: -x[1]))}")
    
    print("\n" + "=" * 60)
    print("NuScenes loader verification PASSED")
    print("=" * 60)
