import os
import sys
import subprocess

def create_directories():
    """This function creates all standard project subdirectories."""
    directories = [
        "data/nuscenes",
        "data/gtsrb",
        "data/carla",
        "src/data_prep",
        "src/models",
        "src/training",
        "src/utils",
        "notebooks",
        "checkpoints"
    ]
    print("Creating Project Subdirectories")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created and verified: {directory}")
    print()

def check_dependencies():
    """ This function checks if required python packages are imported correctly."""
    print("Checking System Dependencies")
    
    dependencies = {
        "torch": "PyTorch (Deep Learning Core)",
        "torchvision": "Torchvision (Computer Vision Utility)",
        "transformers": "HuggingFace Transformers (ViT Models)",
        "timm": "Timm (PyTorch Image Models)",
        "ultralytics": "Ultralytics (YOLOv10)",
        "nuscenes": "nuScenes Devkit (Autonomous Dataset Toolkit)",
        "pyquaternion": "PyQuaternion (3D projections)",
        "albumentations": "Albumentations (EoT & Augmentation)",
        "cv2": "OpenCV (Image Processing)",
        "sklearn": "Scikit-Learn (Metrics)",
        "matplotlib": "Matplotlib (Visualization)"
    }
    
    missing_packages = []
    
    for package, description in dependencies.items():
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            print(f"{package:<15}: (version: {version:<10}) -> {description}")
        except ImportError:
            print(f"MISSING: {package:<15} - {description}")
            missing_packages.append(package)
            
    print()
    if missing_packages:
        print(f"Warning: {len(missing_packages)} package(s) missing. You can install them using:")
        print(f"pip install -r requirements.txt")
    else:
        print("All key dependencies are successfully installed!")
    print()

def check_cuda():
    """This function verifies PyTorch CUDA acceleration and VRAM status."""
    print("Checking GPU & CUDA Status")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Device Name: {device_name}")
            print(f"Total VRAM: {vram_total:.2f} GB")
            if vram_total < 8.0:
                print("ALERT!! VRAM is less than 8GB. Mixed precision (FP16) and gradient accumulation will be enabled.")
        else:
            print("WARNING CUDA is NOT available. Running on CPU will be extremely slow for fine-tuning!")
    except ImportError:
        print("ERROR PyTorch is not installed correctly. GPU check skipped.")
    print()

if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    os.chdir(project_root)
    print(f"Project Root: {project_root}\n")
    
    create_directories()
    check_dependencies()
    check_cuda()
    print("Setup script complete.")
