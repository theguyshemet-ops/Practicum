"""
generate_carla_renders.py - Offline dataset generator for CARLA physical domain.

Connects to a running CARLA simulator, spawns a target actor, positions
an RGB camera sensor at a grid of distances and angles, projects the 3D
bounding box into 2D camera coordinates, and saves the image and YOLO-format label.

Usage:
    # Make sure CARLA simulator is running on port 2000 first, then:
    python src/data_prep/generate_carla_renders.py --output_dir data/carla --port 2000
"""

import os
import sys
import argparse
import time
import math
import numpy as np
import cv2

# We place the carla import inside a try-except to allow the script to be parsed
# on machines where the CARLA client package is not installed.
try:
    import carla
except ImportError:
    carla = None


def get_2d_bbox(actor, camera, w, h, fov):
    """
    Project 3D bounding box corners of the actor into 2D camera coordinates.
    Returns normalized YOLO coordinates [x_center, y_center, width, height] or None.
    """
    bbox = actor.bounding_box
    transform = actor.get_transform()
    bb_loc = bbox.location
    bb_ext = bbox.extent

    # 8 corner offsets relative to actor transform center
    corners = [
        carla.Location(x=-bb_ext.x, y=-bb_ext.y, z=-bb_ext.z),
        carla.Location(x=bb_ext.x, y=-bb_ext.y, z=-bb_ext.z),
        carla.Location(x=bb_ext.x, y=bb_ext.y, z=-bb_ext.z),
        carla.Location(x=-bb_ext.x, y=bb_ext.y, z=-bb_ext.z),
        carla.Location(x=-bb_ext.x, y=-bb_ext.y, z=bb_ext.z),
        carla.Location(x=bb_ext.x, y=-bb_ext.y, z=bb_ext.z),
        carla.Location(x=bb_ext.x, y=bb_ext.y, z=bb_ext.z),
        carla.Location(x=-bb_ext.x, y=bb_ext.y, z=bb_ext.z),
    ]

    # Convert corners to world coordinates
    world_corners = []
    for corner in corners:
        # Adjust for bbox offset, then transform to world space
        corner_world = transform.transform(corner + bb_loc)
        world_corners.append(corner_world)

    # Get camera inverse transform to translate world corners to camera local space
    cam_transform = camera.get_transform()
    cam_inv = cam_transform.inverse()

    xs = []
    ys = []

    # Focal length formula
    f = w / (2.0 * math.tan(fov * math.pi / 360.0))

    for corner in world_corners:
        # Transform from world space to camera local space
        p = cam_inv.transform(corner)

        # CARLA coordinate system: X=forward, Y=right, Z=up
        # If point is behind or on the camera plane, skip
        if p.x <= 0.1:
            continue

        # Project 3D coordinate to 2D image coordinate
        # Local Y is right, local -Z is down (in standard image coordinate space)
        u = (p.y * f / p.x) + w / 2.0
        v = (-p.z * f / p.x) + h / 2.0

        xs.append(u)
        ys.append(v)

    if not xs:
        return None

    # Clip to image boundary
    x_min = max(0.0, min(xs))
    x_max = min(float(w), max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(float(h), max(ys))

    # Reject tiny/invalid boxes
    if (x_max - x_min) < 2 or (y_max - y_min) < 2:
        return None

    # Compute normalized YOLO values
    x_center = (x_min + x_max) / (2.0 * w)
    y_center = (y_min + y_max) / (2.0 * h)
    bbox_w = (x_max - x_min) / w
    bbox_h = (y_max - y_min) / h

    return [x_center, y_center, bbox_w, bbox_h]


def main():
    parser = argparse.ArgumentParser(description="CARLA Dataset Generator")
    parser.add_argument("--output_dir", type=str, default="data/carla", help="Directory to save renders")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=640, help="Render height")
    parser.add_argument("--fov", type=float, default=90.0, help="Camera FOV")
    args = parser.parse_args()

    if carla is None:
        print("[ERROR] The 'carla' python package is not installed.")
        print("Please install carla API or run this script in an environment where it is available.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Distances (meters) and angles (degrees) sweeps
    distances = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    angles = [0.0, 15.0, 30.0, 45.0]
    weathers = [
        ("clear", carla.WeatherParameters.ClearNoon),
        ("rain", carla.WeatherParameters.HardRainNoon),
        ("fog", carla.WeatherParameters.FoggyNoon)
    ]

    actor_list = []
    import queue

    try:
        print(f"Connecting to CARLA server at {args.host}:{args.port}...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()

        # Set world to synchronous mode to guarantee frame alignments
        settings = world.get_settings()
        original_sync_mode = settings.synchronous_mode
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # Find sign prop or fall back to a standard traffic sign / obstacle prop
        prop_bp = None
        # Common blueprints for static street props in CARLA
        blueprints_to_try = [
            "static.prop.stop_sign",
            "static.prop.streetbarrier",
            "static.prop.trafficcone",
            "vehicle.tesla.model3"  # final fallback
        ]
        for bp_name in blueprints_to_try:
            bps = blueprint_library.filter(bp_name)
            if bps:
                prop_bp = bps[0]
                print(f"Using blueprint target: {bp_name}")
                break

        if prop_bp is None:
            raise RuntimeError("Could not find any suitable target blueprints in CARLA library.")

        # Spawn target actor at map center (or standard spawn point)
        spawn_pts = world.get_map().get_spawn_points()
        spawn_pt = spawn_pts[0] if spawn_pts else carla.Transform()
        target_transform = carla.Transform(
            spawn_pt.location + carla.Location(z=0.5),
            carla.Rotation(yaw=0)
        )
        target_actor = world.spawn_actor(prop_bp, target_transform)
        actor_list.append(target_actor)
        print(f"Spawned target actor at: {target_actor.get_location()}")

        # Set up camera sensor
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))
        camera_bp.set_attribute("fov", str(args.fov))

        # Position camera looking at target
        # Initially spawn camera attached to spectator/world
        camera_transform = carla.Transform(
            target_transform.location + carla.Location(x=-5.0, z=1.0),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )
        camera_sensor = world.spawn_actor(camera_bp, camera_transform)
        actor_list.append(camera_sensor)

        # Synchronous queue for camera frames
        image_queue = queue.Queue()
        camera_sensor.listen(image_queue.put)

        # Let the simulator warm up for a few steps
        for _ in range(10):
            world.tick()
            try:
                image_queue.get(timeout=0.5)
            except queue.Empty:
                pass

        total_captured = 0

        # Run campaigns
        for weather_name, weather_params in weathers:
            print(f"Setting weather: {weather_name}")
            world.set_weather(weather_params)

            for dist in distances:
                for yaw_deg in angles:
                    # Compute camera transform relative to target
                    # yaw_deg determines the horizontal rotation angle of the camera around the sign
                    yaw_rad = math.radians(yaw_deg)
                    
                    # Camera Position (spherical coordinates relative to sign)
                    cam_x = target_transform.location.x - dist * math.cos(yaw_rad)
                    cam_y = target_transform.location.y - dist * math.sin(yaw_rad)
                    cam_z = target_transform.location.z + 0.8  # slightly elevated camera height
                    
                    # Face camera towards target sign
                    cam_yaw = yaw_deg
                    cam_pitch = -3.0  # slight downward angle
                    
                    new_transform = carla.Transform(
                        carla.Location(x=cam_x, y=cam_y, z=cam_z),
                        carla.Rotation(pitch=cam_pitch, yaw=cam_yaw, roll=0.0)
                    )
                    camera_sensor.set_transform(new_transform)

                    # Tick simulator to update physics/transform and render frame
                    world.tick()
                    
                    try:
                        # Fetch the camera frame
                        image = image_queue.get(timeout=2.0)
                        
                        # Project bounding box
                        bbox_yolo = get_2d_bbox(target_actor, camera_sensor, args.width, args.height, args.fov)
                        
                        if bbox_yolo is None:
                            # Sign was outside view, skip
                            continue

                        # Save image
                        # Image format in CARLA is BGRA, we convert to BGR for OpenCV saving
                        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                        array = np.reshape(array, (image.height, image.width, 4))
                        bgr_img = array[:, :, :3]

                        base_name = f"carla_{weather_name}_d{int(dist)}_y{int(yaw_deg)}"
                        img_path = os.path.join(args.output_dir, f"{base_name}.png")
                        txt_path = os.path.join(args.output_dir, f"{base_name}.txt")

                        cv2.imwrite(img_path, bgr_img)

                        # Write label
                        # Format: class_id, x_center, y_center, width, height (class_id=14 for GTSRB Stop Sign)
                        class_id = 14  
                        with open(txt_path, "w") as f:
                            f.write(f"{class_id} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n")

                        total_captured += 1
                        print(f"Captured: {base_name}.png -> Bbox: {bbox_yolo}")

                    except queue.Empty:
                        print(f"[WARN] Timeout waiting for frame at dist={dist}, yaw={yaw_deg}")
                    except Exception as e:
                        print(f"[ERROR] Failed to process frame at dist={dist}, yaw={yaw_deg}: {e}")

        print(f"Completed! Generated {total_captured} frames in: {args.output_dir}")

    finally:
        # Restore original synchronous mode settings
        # We wrap in try-except because world might not be initialized if connection failed
        try:
            settings = world.get_settings()
            settings.synchronous_mode = original_sync_mode
            world.apply_settings(settings)
        except Exception:
            pass

        # Destroy actors to clean up simulator world
        print("Cleaning up spawned actors...")
        for actor in actor_list:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
