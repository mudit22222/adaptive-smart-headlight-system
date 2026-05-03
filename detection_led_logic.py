import cv2
import numpy as np

GRID_WIDTH, GRID_HEIGHT = 32, 24
LED_SIZE, LED_GAP = 20, 5
LED_ON_COLOR = (255, 200, 100)
LED_OFF_COLOR = (20, 20, 20)

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def create_led_grid(led_states):
    grid_h_pixels = GRID_HEIGHT * (LED_SIZE + LED_GAP) - LED_GAP
    grid_w_pixels = GRID_WIDTH * (LED_SIZE + LED_GAP) - LED_GAP
    led_display = np.zeros((grid_h_pixels, grid_w_pixels, 3), dtype=np.uint8)

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            color = LED_ON_COLOR if led_states[y, x] == 1 else LED_OFF_COLOR
            start_x = x * (LED_SIZE + LED_GAP)
            start_y = y * (LED_SIZE + LED_GAP)
            cv2.rectangle(
                led_display,
                (start_x, start_y),
                (start_x + LED_SIZE, start_y + LED_SIZE),
                color,
                -1,
            )
    return led_display


def led_grid(detections, frame_width=640, frame_height=480):
    """
    Build and return an LED grid image from structured detection dicts.

    Args:
        detections:   list of dicts with keys "name", "confidence", "box" ([x1,y1,x2,y2])
        frame_width:  actual camera frame width  (pixels) — used for scaling
        frame_height: actual camera frame height (pixels) — used for scaling
    """
    # Start with all LEDs ON (background)
    led_states = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)

    for det in detections:
        # Accept both dict format (from main.py) and raw list format [x1,y1,x2,y2,cls,conf]
        if isinstance(det, dict):
            if det.get("name") not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = det["box"]
        elif isinstance(det, (list, tuple, np.ndarray)) and len(det) >= 5:
            if det[4] not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = det[:4]
        else:
            continue

        # Scale bounding box coords → LED grid coords
        led_x1 = int((x1 / frame_width)  * GRID_WIDTH)
        led_y1 = int((y1 / frame_height) * GRID_HEIGHT)
        led_x2 = int((x2 / frame_width)  * GRID_WIDTH)
        led_y2 = int((y2 / frame_height) * GRID_HEIGHT)

        # Clamp to grid bounds
        led_x1 = max(0, min(GRID_WIDTH,  led_x1))
        led_y1 = max(0, min(GRID_HEIGHT, led_y1))
        led_x2 = max(0, min(GRID_WIDTH,  led_x2))
        led_y2 = max(0, min(GRID_HEIGHT, led_y2))

        # Mark occupied cells as OFF
        if led_x2 > led_x1 and led_y2 > led_y1:
            led_states[led_y1:led_y2, led_x1:led_x2] = 0

    return create_led_grid(led_states)
