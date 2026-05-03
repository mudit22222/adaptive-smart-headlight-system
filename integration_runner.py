import carla
import cv2
import numpy as np
import time
import random
import threading

from ego_vehicle import setup_ego_vehicle, spawn_npcs
from camera_feed import attach_camera
from yolo_detection import load_yolo_model
from led_grid import led_grid

# ── Shared state (protected by a lock) ────────────────────────────────────────
npc_list: list = []
collision_sensors: dict = {}
state_lock = threading.Lock()

# Fixed width (px) for the LED-grid sidebar
LED_SIDEBAR_WIDTH = 300


# ── NPC helpers ───────────────────────────────────────────────────────────────

def spawn_single_npc(world, blueprint_library):
    """Spawn one random NPC vehicle and enable autopilot. Returns actor or None."""
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("⚠ No spawn points available.")
        return None

    bp = random.choice(blueprint_library.filter("vehicle.*"))
    transform = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(bp, transform)
    if vehicle:
        vehicle.set_autopilot(True)
    return vehicle


def attach_collision_sensor(world, vehicle, blueprint_library):
    """
    Attach a collision sensor to *vehicle*.
    On collision: destroy the NPC + sensor, spawn a fresh replacement.
    Returns the sensor actor.
    """
    col_bp = blueprint_library.find("sensor.other.collision")
    col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)

    # Pin objects into default args to avoid stale-closure bugs
    def on_collision(event, _vehicle=vehicle, _col_sensor=col_sensor):
        print(f"💥 Collision detected for NPC ID {_vehicle.id}")
        try:
            with state_lock:
                # --- Destroy old sensor first, then the NPC ---
                if _col_sensor.is_alive:
                    _col_sensor.stop()
                    _col_sensor.destroy()
                collision_sensors.pop(_vehicle.id, None)

                if _vehicle.is_alive:
                    _vehicle.destroy()
                if _vehicle in npc_list:
                    npc_list.remove(_vehicle)

            # Spawn replacement (outside lock – world.try_spawn_actor can block)
            new_npc = spawn_single_npc(world, blueprint_library)
            if new_npc:
                new_sensor = attach_collision_sensor(world, new_npc, blueprint_library)
                with state_lock:
                    npc_list.append(new_npc)
                    collision_sensors[new_npc.id] = new_sensor
                print(f"🚗 Respawned NPC ID {new_npc.id}")
        except Exception as e:
            print(f"⚠ Error handling collision for NPC {_vehicle.id}: {e}")

    col_sensor.listen(on_collision)
    return col_sensor


# ── Camera helper ─────────────────────────────────────────────────────────────

def attach_camera_safe(world, ego):
    """
    Wraps attach_camera() so that frame_data also carries a lock,
    preventing torn reads from the camera callback thread.
    """
    camera, frame_data = attach_camera(world, ego)
    frame_data.setdefault("lock", threading.Lock())
    return camera, frame_data


# ── YOLO classes that map to COCO labels ──────────────────────────────────────
# Note: COCO uses "motorcycle", not "motorbike"
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global npc_list

    # ── Connect to CARLA ──
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    print(f"✅ Connected to CARLA world: {world.get_map().name}")

    ego = None
    camera = None

    try:
        # ── Ego vehicle ──
        ego = setup_ego_vehicle(world)
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        # ── NPC vehicles ──
        with state_lock:
            npc_list = spawn_npcs(world, count=20)
            for npc in npc_list:
                collision_sensors[npc.id] = attach_collision_sensor(
                    world, npc, blueprint_library
                )

        # ── Camera ──
        camera, frame_data = attach_camera_safe(world, ego)

        # ── YOLO ──
        model, device = load_yolo_model()
        print("🚦 Starting YOLO detection loop… (press Q to quit)")

        while True:
            # Thread-safe frame read
            frame_lock = frame_data.get("lock")
            if frame_lock:
                with frame_lock:
                    frame = frame_data.get("frame")
                    frame = frame.copy() if frame is not None else None
            else:
                frame = frame_data.get("frame")

            if frame is None:
                print("📸 Waiting for camera feed…")
                time.sleep(0.1)
                continue

            # ── YOLO inference ──
            results = model(frame)
            detections = results[0].boxes.data.cpu().numpy()
            names = model.names

            structured_detections = []
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                cls_name = names[int(cls_id)]
                structured_detections.append({
                    "name": cls_name,
                    "confidence": float(conf),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                })

            vehicle_detections = [
                d for d in structured_detections if d["name"] in VEHICLE_CLASSES
            ]

            # ── LED grid (fixed sidebar width) ──
            grid_raw = led_grid(vehicle_detections, frame.shape[1])
            grid_padded = cv2.copyMakeBorder(
                grid_raw, 20, 20, 20, 20,
                cv2.BORDER_CONSTANT, value=(40, 40, 40),
            )
            # Resize sidebar to match frame height, keep fixed width
            grid_sidebar = cv2.resize(
                grid_padded, (LED_SIDEBAR_WIDTH, frame.shape[0])
            )

            # ── Draw bounding boxes ──
            plotted = frame.copy()
            for det in vehicle_detections:
                x1, y1, x2, y2 = det["box"]
                label = f"{det['name']} {det['confidence']:.2f}"
                cv2.rectangle(plotted, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    plotted, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

            # ── Combine & display ──
            combined = np.hstack((plotted, grid_sidebar))
            cv2.imshow("CARLA | YOLO | Auto-Respawn | LED Grid", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("🛑 Quit signal received.")
                break

            # Yield CPU when frame is processed
            time.sleep(0.01)

    finally:
        print("🧹 Cleaning up…")

        # Stop & destroy camera
        if camera is not None:
            try:
                camera.stop()
                camera.destroy()
            except Exception as e:
                print(f"⚠ Camera cleanup error: {e}")

        # Destroy all collision sensors
        with state_lock:
            sensors_snapshot = list(collision_sensors.values())
            npcs_snapshot = list(npc_list)

        for sensor in sensors_snapshot:
            try:
                if sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except Exception as e:
                print(f"⚠ Sensor destroy error: {e}")

        # Destroy all NPC vehicles
        for npc in npcs_snapshot:
            try:
                if npc.is_alive:
                    npc.destroy()
            except Exception as e:
                print(f"⚠ NPC destroy error: {e}")

        # Destroy ego last
        if ego is not None:
            try:
                ego.destroy()
            except Exception as e:
                print(f"⚠ Ego destroy error: {e}")

        cv2.destroyAllWindows()
        print(" All actors destroyed cleanly.")


if __name__ == "__main__":
    main()
