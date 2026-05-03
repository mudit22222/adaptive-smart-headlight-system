import random
import carla


def setup_ego_vehicle(world):
    """
    Spawn the ego vehicle (Tesla Model 3) at a random spawn point.

    Fixes vs original:
    - blueprint_library.filter('model3') returns an empty list on many CARLA
      versions / maps — use the full type_id 'vehicle.tesla.model3' instead.
    - filter() can still return an empty list if the asset isn't present;
      added a clear error rather than an IndexError on [0].
    - world.spawn_actor() raises on collision; wrapped in try/except so a
      bad spawn point doesn't crash the whole program.
    - Shuffled spawn points so repeated runs pick different locations.

    Returns:
        ego vehicle actor, or raises RuntimeError if spawning fails.
    """
    blueprint_library = world.get_blueprint_library()

    # Full type_id is required — 'model3' alone often matches nothing
    vehicle_bps = blueprint_library.filter('vehicle.tesla.model3')
    if not vehicle_bps:
        raise RuntimeError(
            "Blueprint 'vehicle.tesla.model3' not found. "
            "Check your CARLA asset pack or use a different vehicle."
        )
    vehicle_bp = vehicle_bps[0]

    # Optionally make the ego vehicle a distinct colour
    if vehicle_bp.has_attribute('color'):
        vehicle_bp.set_attribute('color', '255,0,0')   # red — easy to spot

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available on this map.")

    random.shuffle(spawn_points)

    ego_vehicle = None
    for sp in spawn_points:
        try:
            ego_vehicle = world.spawn_actor(vehicle_bp, sp)
            break
        except RuntimeError:
            # Spawn point occupied — try next one
            continue

    if ego_vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle at any spawn point.")

    ego_vehicle.set_autopilot(True)
    print(f"🚘 Ego vehicle spawned: {ego_vehicle.type_id} (id={ego_vehicle.id})")
    return ego_vehicle


def spawn_npcs(world, count=20):
    """
    Spawn up to *count* NPC vehicles with autopilot enabled.

    Fixes vs original:
    - Duplicate import of random removed (it's at the top of the module).
    - spawn_points list can be shorter than count — added a guard so we
      never request more vehicles than there are unique spawn points.
    - Used random.sample() instead of random.choice() in a loop so the
      same spawn point is never tried twice, reducing failed spawns.
    - Printed how many were *requested* vs *actually spawned* for clarity.

    Returns:
        list of successfully spawned NPC vehicle actors.
    """
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        print("⚠  No spawn points available — no NPCs spawned.")
        return []

    # Never request more spawn points than the map provides
    count = min(count, len(spawn_points))
    selected_points = random.sample(spawn_points, count)

    vehicle_bps = blueprint_library.filter('vehicle.*')
    if not vehicle_bps:
        print("⚠  No vehicle blueprints found — no NPCs spawned.")
        return []

    vehicles = []
    for transform in selected_points:
        bp = random.choice(vehicle_bps)

        # Some blueprints require specific attributes; set a safe colour
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        npc = world.try_spawn_actor(bp, transform)
        if npc:
            npc.set_autopilot(True)
            vehicles.append(npc)

    print(f"🚗 Spawned {len(vehicles)}/{count} NPC vehicles")
    return vehicles
