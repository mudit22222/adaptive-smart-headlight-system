import numpy as np
import cv2

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def led_grid(detections, frame_width=640, frame_height=480):
    """
    Render an LED matrix grid showing vehicle positions.

    Args:
        detections:   list of dicts with keys "name", "confidence", "box" ([x1,y1,x2,y2])
        frame_width:  actual camera frame width  (pixels)
        frame_height: actual camera frame height (pixels)

    Returns:
        BGR image (numpy array) of the LED matrix
    """
    grid_cols = 16          # horizontal LEDs
    grid_rows = 12          # vertical LEDs (matches 4:3 camera aspect ratio)
    cell_size = 20
    spacing   = 5

    # Colors
    led_on_color     = (0, 220, 0)      # bright green — free cell
    led_off_color    = (0, 0, 60)       # dark blue-red — occupied cell
    background_color = (10, 10, 10)

    img_w = spacing + grid_cols * (cell_size + spacing)
    img_h = spacing + grid_rows * (cell_size + spacing) + 20   # +20 for label row

    grid_img = np.full((img_h, img_w, 3), background_color, dtype=np.uint8)

    # All LEDs ON by default
    led_state = np.ones((grid_rows, grid_cols), dtype=bool)

    # ── Map each detection onto the grid ──────────────────────────────────────
    for det in (detections or []):
        # Support both dict format {"name", "box"} and raw list [x1,y1,x2,y2,cls,...]
        if isinstance(det, dict):
            if det.get("name") not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = det["box"]
        elif isinstance(det, (list, tuple, np.ndarray)) and len(det) >= 5:
            if det[4] not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        else:
            continue

        # Scale bounding box → grid coordinates
        col1 = int((x1 / frame_width)  * grid_cols)
        row1 = int((y1 / frame_height) * grid_rows)
        col2 = int((x2 / frame_width)  * grid_cols)
        row2 = int((y2 / frame_height) * grid_rows)

        # Clamp to valid grid bounds
        col1 = max(0, min(grid_cols - 1, col1))
        col2 = max(0, min(grid_cols,     col2))
        row1 = max(0, min(grid_rows - 1, row1))
        row2 = max(0, min(grid_rows,     row2))

        # Ensure at least 1 cell is always marked
        if col2 <= col1:
            col2 = col1 + 1
        if row2 <= row1:
            row2 = row1 + 1

        led_state[row1:row2, col1:col2] = False

    # ── Draw LEDs ─────────────────────────────────────────────────────────────
    for row in range(grid_rows):
        for col in range(grid_cols):
            cx = spacing + col * (cell_size + spacing) + cell_size // 2
            cy = spacing + row * (cell_size + spacing) + cell_size // 2
            radius = cell_size // 2

            color = led_on_color if led_state[row, col] else led_off_color

            # Filled circle (LED body)
            cv2.circle(grid_img, (cx, cy), radius, color, -1)

            # Rim: bright when ON, subtle when OFF
            rim = (180, 255, 180) if led_state[row, col] else (40, 40, 40)
            cv2.circle(grid_img, (cx, cy), radius, rim, 1)

    # ── Label ─────────────────────────────────────────────────────────────────
    cv2.putText(
        grid_img, "LED MATRIX",
        (spacing, img_h - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (100, 255, 100), 1, cv2.LINE_AA,
    )

    return grid_img
