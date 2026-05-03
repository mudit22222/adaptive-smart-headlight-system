import torch
from ultralytics import YOLO


def load_yolo_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🧠 Using device: {device}")
    model = YOLO('yolov8n.pt').to(device)
    return model, device


def run_yolo_detection(model, frame):
    results = model.predict(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()
    plotted = results[0].plot()
    return plotted, detections
