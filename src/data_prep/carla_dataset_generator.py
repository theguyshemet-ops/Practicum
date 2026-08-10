"""
CARLA Dataset Generator - Controlled Traffic Sign Rendering

Connects to a running CARLA server (CarlaUE4.exe) and generates a
controlled dataset of traffic sign images under systematic variations
of weather, distance, viewing angle, and sign type.

Generated dataset structure:
    data/carla/
    ├-- clear/
    |   ├-- carla_clear_d05_y00_sign14.png
    |   ├-- carla_clear_d05_y00_sign14.txt   (YOLO format)
    |   ├-- ...
    ├-- rain/
    |   └-- ...
    ├-- fog/
    |   └-- ...
    └-- metadata.json  (dataset manifest with all parameters)

YOLO label format: class_id centre_x centre_y width height (all normalised)

Requirements:
    - CARLA simulator running (CarlaUE4.exe)
    - CARLA PythonAPI on sys.path
    - Python 3.8+

Usage:
    1. Start CARLA: data\\carla\\CarlaUE4.exe
    2. Run: python src/data_prep/carla_dataset_generator.py --output data/carla
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

# -- Attempt to import CARLA PythonAPI ------------------------------------
# The CARLA PythonAPI path must be added before import.
_CARLA_API_PATHS = [
    r"E:\Python\VIT_vs_YOLO\Implementation\MainFolder\data\carla\PythonAPI\carla", #path should be replaced with the user's path
    r"E:\Python\VIT_vs_YOLO\Implementation\MainFolder\data\carla\PythonAPI",
]
for _p in _CARLA_API_PATHS:
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

try:
    import carla
    if not hasattr(carla, "Client"):
        print("\n" + "!" * 72)
        print(" [ERROR] You are running Python in the WRONG environment!")
        print("         The active environment has a dummy 'carla' package from PyPI.")
        print("         Please activate the correct environment by running:")
        print("             conda activate carla_env")
        print("!" * 72 + "\n")
        sys.exit(1)
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("[WARNING] CARLA PythonAPI not found. Install it or add to sys.path.")
    print("         Expected at: E:\\Python\\VIT_vs_YOLO\\Implementation\\MainFolder\\data\\carla\\PythonAPI")


# =======================================================================
# Configuration
# =======================================================================

# Weather presets: CARLA weather parameters
WEATHER_PRESETS = {
    "clear": {
        "cloudiness": 10.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 5.0,
        "sun_altitude_angle": 60.0,
        "fog_density": 0.0,
        "fog_distance": 0.0,
        "wetness": 0.0,
    },
    "rain": {
        "cloudiness": 80.0,
        "precipitation": 60.0,
        "precipitation_deposits": 50.0,
        "wind_intensity": 30.0,
        "sun_altitude_angle": 40.0,
        "fog_density": 10.0,
        "fog_distance": 50.0,
        "wetness": 70.0,
    },
    "fog": {
        "cloudiness": 90.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 5.0,
        "sun_altitude_angle": 30.0,
        "fog_density": 70.0,
        "fog_distance": 20.0,
        "wetness": 10.0,
    },
}

# Viewing distances (metres from camera to sign)
DISTANCES = [5, 10, 15, 20, 30]

# Camera yaw angles (degrees offset from straight-on)
YAW_ANGLES = [0, 15, 30, 45]

# Sign types mapped to GTSRB class IDs
# We use CARLA's built-in static mesh traffic signs
SIGN_CONFIGS = {
    14: {
        "name": "stop",
        "gtsrb_class": 14,
        "description": "Stop sign",
        "carla_bp": "static.prop.streetsign01",
        "face_local_z": 3.58,
    },
    1: {
        "name": "speed_limit_30",
        "gtsrb_class": 1,
        "description": "Speed Limit 30 km/h",
        "carla_bp": "static.prop.streetsign",
        "face_local_z": 1.85,
    },
    2: {
        "name": "speed_limit_50",
        "gtsrb_class": 2,
        "description": "Speed Limit 50 km/h",
        "carla_bp": "static.prop.streetsign",
        "face_local_z": 1.85,
    },
    13: {
        "name": "yield",
        "gtsrb_class": 13,
        "description": "Yield / Give Way",
        "carla_bp": "static.prop.streetsign04",
        "face_local_z": 2.47,
    },
    17: {
        "name": "no_entry",
        "gtsrb_class": 17,
        "description": "No Entry",
        "carla_bp": "static.prop.streetsign01",
        "face_local_z": 3.58,
    },
}

# Camera resolution (matches our detection pipeline)
IMG_WIDTH = 640
IMG_HEIGHT = 640
CAMERA_FOV = 90.0


# =======================================================================
# Helper: 3D -> 2D Projection for Bounding Box Computation
# =======================================================================

def project_3d_to_2d(location_3d, camera_transform, camera_fov, img_w, img_h):
    """
    Project a 3D world location to 2D pixel coordinates.
    
    Uses the pinhole camera model with CARLA's coordinate system.
    
    Parameters
    ----------
    location_3d : carla.Location
        3D world position of the point.
    camera_transform : carla.Transform
        Camera's world transform (position + rotation).
    camera_fov : float
        Camera field of view in degrees.
    img_w, img_h : int
        Image dimensions in pixels.
    
    Returns
    -------
    (u, v) : tuple of int or None
        Pixel coordinates, or None if behind camera.
    """
    # World -> Camera coordinate transform
    cam_loc = camera_transform.location
    cam_rot = camera_transform.rotation

    # Convert to numpy
    point = np.array([location_3d.x, location_3d.y, location_3d.z, 1.0])
    cam_pos = np.array([cam_loc.x, cam_loc.y, cam_loc.z])

    # Rotation matrix from Euler angles (CARLA uses UE4 convention)
    pitch = math.radians(cam_rot.pitch)
    yaw = math.radians(cam_rot.yaw)
    roll = math.radians(cam_rot.roll)

    # UE4 rotation matrices
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rot = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])

    # Transform point to camera space
    point_cam = rot.T @ (point[:3] - cam_pos)

    # Check if point is behind camera
    if point_cam[0] <= 0:
        return None

    # Focal length from FOV
    fov_rad = math.radians(camera_fov)
    focal = img_w / (2.0 * math.tan(fov_rad / 2.0))

    # Project to 2D (CARLA: x=forward, y=right, z=up)
    u = int(img_w / 2 + focal * point_cam[1] / point_cam[0])
    v = int(img_h / 2 - focal * point_cam[2] / point_cam[0])

    return (u, v)


def compute_sign_bbox_2d(sign_actor, face_local_z, camera_transform,
                          camera_fov, img_w, img_h):
    """
    Compute 2D bounding box of a traffic sign face by projecting its 3D corners.
    """
    if sign_actor is None:
        return None

    # Get actual transform of the actor
    sign_transform = sign_actor.get_transform()

    # Local corners of the sign face (0.6m x 0.6m)
    hw, hh = 0.3, 0.3
    local_corners = [
        carla.Location(x=0.0, y=-hw, z=face_local_z - hh),
        carla.Location(x=0.0, y=hw, z=face_local_z - hh),
        carla.Location(x=0.0, y=-hw, z=face_local_z + hh),
        carla.Location(x=0.0, y=hw, z=face_local_z + hh)
    ]

    us, vs = [], []
    for loc in local_corners:
        world_loc = sign_transform.transform(loc)
        pt = project_3d_to_2d(world_loc, camera_transform, camera_fov, img_w, img_h)
        if pt is not None:
            us.append(pt[0])
            vs.append(pt[1])

    if len(us) < 2:
        return None

    x1 = max(0, min(us))
    y1 = max(0, min(vs))
    x2 = min(img_w, max(us))
    y2 = min(img_h, max(vs))

    if x2 <= x1 or y2 <= y1:
        return None

    # Convert to YOLO normalised format
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    return (cx, cy, w, h)


# =======================================================================
# Main Generator Class
# =======================================================================

class CARLADatasetGenerator:
    """
    Generates a controlled traffic sign dataset from CARLA simulator.
    
    Connects to a running CARLA server, systematically varies weather,
    distance, angle, and sign type, and saves rendered frames with
    YOLO-format annotations.
    
    Args:
        host (str): CARLA server hostname. Default: 'localhost'.
        port (int): CARLA server port. Default: 2000.
        output_dir (str): Root directory for saved frames.
        timeout (float): Connection timeout in seconds. Default: 30.0.
    """

    def __init__(self, host="localhost", port=2000, output_dir="data/carla",
                 timeout=30.0):
        self.host = host
        self.port = port
        self.output_dir = output_dir
        self.timeout = timeout
        self.client = None
        self.world = None
        self.camera = None
        self.camera_bp = None
        self._latest_image = None
        self._image_ready = False

    def connect(self):
        """Connect to the CARLA server and configure the world."""
        if not CARLA_AVAILABLE:
            raise RuntimeError(
                "CARLA PythonAPI not available. Please install it or add to sys.path.\n"
                "Expected at: E:\\Python\\VIT_vs_YOLO\\Implementation\\MainFolder\\data\\carla\\PythonAPI"
            )

        print(f"[CARLA] Connecting to {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)

        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        print(f"[CARLA] Connected. Map: {self.world.get_map().name}")
        print(f"[CARLA] Synchronous mode enabled.")

    def _set_weather(self, weather_name):
        """Apply a weather preset to the CARLA world."""
        params = WEATHER_PRESETS[weather_name]
        weather = carla.WeatherParameters(
            cloudiness=params["cloudiness"],
            precipitation=params["precipitation"],
            precipitation_deposits=params["precipitation_deposits"],
            wind_intensity=params["wind_intensity"],
            sun_altitude_angle=params["sun_altitude_angle"],
            fog_density=params["fog_density"],
            fog_distance=params["fog_distance"],
            wetness=params["wetness"],
        )
        self.world.set_weather(weather)

    def _spawn_camera(self, transform):
        """Spawn an RGB camera sensor at the given transform."""
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(IMG_WIDTH))
        camera_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))

        self.camera = self.world.spawn_actor(camera_bp, transform)
        self._image_ready = False
        self.camera.listen(self._on_image)
        return self.camera

    def _on_image(self, image):
        """Callback: store the latest camera frame."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((IMG_HEIGHT, IMG_WIDTH, 4))  # BGRA
        self._latest_image = array[:, :, :3][:, :, ::-1]  # -> RGB
        self._image_ready = True

    def _capture_frame(self, max_ticks=20):
        """Tick the world and wait for a camera frame."""
        self._image_ready = False
        for _ in range(max_ticks):
            self.world.tick()
            time.sleep(0.05)
            if self._image_ready:
                return self._latest_image.copy()
        print("[WARNING] Frame capture timed out after max ticks.")
        return self._latest_image.copy() if self._latest_image is not None else None

    def _spawn_sign(self, sign_config, location, rotation=None):
        """
        Spawn a traffic sign prop at the given location.
        
        Returns the spawned actor or None if the blueprint is not found.
        """
        bp_lib = self.world.get_blueprint_library()

        # Try to find the blueprint
        bp_name = sign_config["carla_bp"]
        bp = None

        # Search strategies: exact match, then partial match
        try:
            bp = bp_lib.find(bp_name)
        except RuntimeError:
            # Try searching by partial name
            candidates = [b for b in bp_lib if bp_name.lower() in b.id.lower()]
            if candidates:
                bp = candidates[0]
                print(f"  [CARLA] Using fallback blueprint: {bp.id}")

        if bp is None:
            # Use any available static prop as fallback
            static_props = [b for b in bp_lib if "static.prop" in b.id.lower()
                           and "sign" in b.id.lower()]
            if static_props:
                bp = static_props[0]
                print(f"  [CARLA] Using generic sign prop: {bp.id}")
            else:
                print(f"  [WARNING] No suitable blueprint found for {bp_name}")
                return None

        if rotation is None:
            rotation = carla.Rotation(pitch=0, yaw=0, roll=0)

        transform = carla.Transform(location, rotation)
        actor = self.world.spawn_actor(bp, transform)
        return actor

    def _compute_sign_extent(self, distance):
        """
        Estimate the half-extent of a traffic sign face based on typical sizes.
        Real traffic signs are approximately 0.6m x 0.6m.
        """
        return (0.3, 0.3)  # half-width, half-height in metres

    def generate_dataset(self, weathers=None, distances=None, yaw_angles=None,
                          sign_ids=None):
        """
        Generate the full controlled dataset.
        
        Parameters
        ----------
        weathers : list of str
            Weather conditions. Default: ['clear', 'rain', 'fog'].
        distances : list of int
            Viewing distances in metres. Default: [5, 10, 15, 20, 30].
        yaw_angles : list of int
            Camera yaw offsets in degrees. Default: [0, 15, 30, 45].
        sign_ids : list of int
            GTSRB class IDs. Default: [14, 1, 2, 13, 17].
        
        Returns
        -------
        manifest : list of dict
            Metadata for each generated frame.
        """
        if weathers is None:
            weathers = list(WEATHER_PRESETS.keys())
        if distances is None:
            distances = DISTANCES
        if yaw_angles is None:
            yaw_angles = YAW_ANGLES
        if sign_ids is None:
            sign_ids = list(SIGN_CONFIGS.keys())

        total = len(weathers) * len(distances) * len(yaw_angles) * len(sign_ids)
        print(f"\n{'=' * 72}")
        print(f"  CARLA Dataset Generation")
        print(f"  Weathers: {weathers}")
        print(f"  Distances: {distances}m")
        print(f"  Yaw angles: {yaw_angles}°")
        print(f"  Sign types: {[SIGN_CONFIGS[s]['name'] for s in sign_ids]}")
        print(f"  Total frames: {total}")
        print(f"  Output: {self.output_dir}")
        print(f"{'=' * 72}\n")

        manifest = []
        frame_count = 0
        start_time = time.time()

        # Find a suitable spawn point for the camera
        spawn_points = self.world.get_map().get_spawn_points()
        base_spawn = spawn_points[0] if spawn_points else carla.Transform(
            carla.Location(x=0, y=0, z=2.0)
        )

        for weather in weathers:
            weather_dir = os.path.join(self.output_dir, weather)
            os.makedirs(weather_dir, exist_ok=True)

            print(f"\n  Weather: {weather}")
            self._set_weather(weather)
            # Let weather settle
            for _ in range(5):
                self.world.tick()

            for dist in distances:
                for yaw in yaw_angles:
                    for sign_id in sign_ids:
                        sign_config = SIGN_CONFIGS[sign_id]
                        frame_count += 1

                        # -- Position camera and sign --------------------
                        # Camera at base position
                        cam_x = base_spawn.location.x
                        cam_y = base_spawn.location.y
                        cam_z = base_spawn.location.z + 1.5  # eye height

                        # Yaw angle for camera viewing direction
                        cam_yaw = base_spawn.rotation.yaw + yaw

                        # Sign placed `dist` metres in front of camera
                        yaw_rad = math.radians(cam_yaw)
                        sign_x = cam_x + dist * math.cos(yaw_rad)
                        sign_y = cam_y + dist * math.sin(yaw_rad)
                        # Center the sign face at the camera's eye height
                        face_z = sign_config.get("face_local_z", 1.9)
                        sign_z = cam_z - face_z

                        sign_location = carla.Location(
                            x=sign_x, y=sign_y, z=sign_z
                        )
                        # Sign faces the camera
                        sign_rotation = carla.Rotation(
                            pitch=0, yaw=cam_yaw + 180, roll=0
                        )

                        # Spawn sign
                        sign_actor = self._spawn_sign(
                            sign_config, sign_location, sign_rotation
                        )

                        # Spawn camera
                        cam_transform = carla.Transform(
                            carla.Location(x=cam_x, y=cam_y, z=cam_z),
                            carla.Rotation(pitch=0, yaw=cam_yaw, roll=0),
                        )
                        self._spawn_camera(cam_transform)

                        # Let scene settle and capture
                        frame = self._capture_frame(max_ticks=10)

                        if frame is not None:
                            # -- Save image ------------------------------
                            fname = (f"carla_{weather}_d{dist:02d}_y{yaw:02d}"
                                     f"_sign{sign_id}")
                            img_path = os.path.join(weather_dir, f"{fname}.png")
                            lbl_path = os.path.join(weather_dir, f"{fname}.txt")

                            # -- Compute bounding box --------------------
                            bbox = compute_sign_bbox_2d(
                                sign_actor, sign_config.get("face_local_z", 1.9),
                                self.camera.get_transform() if self.camera else cam_transform,
                                CAMERA_FOV,
                                IMG_WIDTH, IMG_HEIGHT,
                            )

                            if bbox is not None:
                                # Overlay the GTSRB sign template onto the blank sign board
                                cx, cy, bw, bh = bbox
                                x1 = int((cx - bw / 2) * IMG_WIDTH)
                                y1 = int((cy - bh / 2) * IMG_HEIGHT)
                                x2 = int((cx + bw / 2) * IMG_WIDTH)
                                y2 = int((cy + bh / 2) * IMG_HEIGHT)
                                
                                # Clip bounds
                                x1 = max(0, min(x1, IMG_WIDTH - 1))
                                y1 = max(0, min(y1, IMG_HEIGHT - 1))
                                x2 = max(0, min(x2, IMG_WIDTH - 1))
                                y2 = max(0, min(y2, IMG_HEIGHT - 1))
                                
                                if (x2 - x1) > 2 and (y2 - y1) > 2:
                                    gtsrb_dir = os.path.join(
                                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                        "data", "gtsrb", "Meta"
                                    )
                                    template_path = os.path.join(gtsrb_dir, f"{sign_id}.png")
                                    if os.path.exists(template_path):
                                        try:
                                            import cv2
                                            template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
                                            if template is not None:
                                                resized_template = cv2.resize(template, (x2 - x1, y2 - y1))
                                                if resized_template.shape[2] == 4:
                                                    alpha = resized_template[:, :, 3:4] / 255.0
                                                    color = resized_template[:, :, :3]
                                                    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                                                    frame[y1:y2, x1:x2] = (1.0 - alpha) * frame[y1:y2, x1:x2] + alpha * color_rgb
                                                else:
                                                    template_rgb = cv2.cvtColor(resized_template, cv2.COLOR_BGR2RGB)
                                                    frame[y1:y2, x1:x2] = template_rgb
                                        except Exception as e:
                                            print(f"      [WARNING] Template overlay failed: {e}")

                            if bbox is None:
                                # Fallback: estimate bbox from distance
                                apparent_size = 0.6 / (dist * math.tan(
                                    math.radians(CAMERA_FOV / 2)) * 2 / IMG_WIDTH)
                                norm_size = apparent_size / IMG_WIDTH
                                bbox = (0.5, 0.5, max(0.02, norm_size),
                                        max(0.02, norm_size))

                            # Save as PNG
                            try:
                                import cv2
                                cv2.imwrite(img_path, frame[:, :, ::-1])  # RGB->BGR
                            except ImportError:
                                from PIL import Image
                                Image.fromarray(frame).save(img_path)

                            cx, cy, w, h = bbox

                            # Save YOLO label
                            with open(lbl_path, "w") as f:
                                f.write(f"{sign_id} {cx:.6f} {cy:.6f} "
                                        f"{w:.6f} {h:.6f}\n")
                            # -- Record metadata ------------------------
                            entry = {
                                "filename": f"{fname}.png",
                                "label_file": f"{fname}.txt",
                                "weather": weather,
                                "distance_m": dist,
                                "yaw_deg": yaw,
                                "sign_id": sign_id,
                                "sign_name": sign_config["name"],
                                "gtsrb_class": sign_config["gtsrb_class"],
                                "bbox_yolo": list(bbox),
                                "image_size": [IMG_WIDTH, IMG_HEIGHT],
                            }
                            manifest.append(entry)

                            elapsed = time.time() - start_time
                            fps = frame_count / max(elapsed, 1)
                            eta = (total - frame_count) / max(fps, 0.01)
                            print(f"    [{frame_count}/{total}] {fname} | "
                                  f"bbox=({cx:.3f},{cy:.3f},{w:.3f},{h:.3f}) | "
                                  f"ETA: {eta:.0f}s")
                        else:
                            print(f"    [{frame_count}/{total}] FAILED: "
                                  f"No frame captured")

                        # Cleanup actors
                        if self.camera is not None:
                            self.camera.stop()
                            self.camera.destroy()
                            self.camera = None
                        if sign_actor is not None:
                            sign_actor.destroy()

        # -- Save manifest -----------------------------------------------
        manifest_path = os.path.join(self.output_dir, "metadata.json")
        manifest_data = {
            "generator": "CARLADatasetGenerator",
            "timestamp": datetime.now().isoformat(),
            "total_frames": len(manifest),
            "weathers": weathers,
            "distances": distances,
            "yaw_angles": yaw_angles,
            "sign_ids": sign_ids,
            "image_size": [IMG_WIDTH, IMG_HEIGHT],
            "camera_fov": CAMERA_FOV,
            "frames": manifest,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

        elapsed = time.time() - start_time
        print(f"\n{'=' * 72}")
        print(f"  Generation Complete!")
        print(f"  Frames: {len(manifest)} / {total}")
        print(f"  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        print(f"  Manifest: {manifest_path}")
        print(f"{'=' * 72}")

        return manifest

    def cleanup(self):
        """Restore CARLA settings and destroy actors."""
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.destroy()
            except Exception:
                pass

        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        print("[CARLA] Cleanup complete.")


# =======================================================================
# CLI Entry Point
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CARLA Dataset Generator for ViT vs CNN Robustness Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full generation (requires running CARLA server):
    python src/data_prep/carla_dataset_generator.py --output data/carla

    # Custom parameters:
    python src/data_prep/carla_dataset_generator.py --output data/carla \\
        --weathers clear rain --distances 5 10 15
        """,
    )
    parser.add_argument("--output", type=str, default="data/carla",
                        help="Output directory (default: data/carla)")
    parser.add_argument("--host", type=str, default="localhost",
                        help="CARLA server host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000,
                        help="CARLA server port (default: 2000)")
    parser.add_argument("--weathers", nargs="+", default=None,
                        choices=["clear", "rain", "fog"],
                        help="Weather conditions (default: all)")
    parser.add_argument("--distances", nargs="+", type=int, default=None,
                        help="Distances in metres (default: 5 10 15 20 30)")
    parser.add_argument("--yaw_angles", nargs="+", type=int, default=None,
                        help="Yaw angles in degrees (default: 0 15 30 45)")

    args = parser.parse_args()

    # Resolve output path relative to project root
    project_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", ".."
    ))
    if not os.path.isabs(args.output):
        args.output = os.path.join(project_root, args.output)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 72)
    print("  CARLA Live Dataset Generator")
    print("  (Requires running CARLA server)")
    print("=" * 72)

    generator = CARLADatasetGenerator(
        host=args.host,
        port=args.port,
        output_dir=args.output,
    )

    try:
        generator.connect()
        generator.generate_dataset(
            weathers=args.weathers,
            distances=args.distances,
            yaw_angles=args.yaw_angles,
        )
    except Exception as e:
        print(f"\n[ERROR] CARLA generation failed: {e}")
        sys.exit(1)
    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()
