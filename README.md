# 🚗 Adaptive Smart Headlight Simulation System

## 🚀 Overview
This project simulates an **adaptive automotive headlight system** using computer vision and real-time decision logic.

It detects vehicles using **YOLOv8** and dynamically controls a virtual **LED matrix** to reduce glare for oncoming traffic. The system is integrated with the **CARLA simulator** for realistic driving scenarios and live visualization.

---

## ⚡ Key Features
- Real-time vehicle detection using YOLOv8  
- Adaptive LED matrix control for glare reduction  
- CARLA-based simulation environment  
- Dynamic vehicle filtering (car, bus, truck, etc.)  
- Real-time visualization (camera + LED grid)  
- Auto-respawn logic for NPC vehicles after collision  

---

## 🏗️ System Architecture
CARLA Simulator
↓
Camera Feed (Ego Vehicle)
↓
YOLOv8 Detection
↓
Vehicle Filtering
↓
LED Control Logic
↓
Visualization (Camera + LED Grid)

---

## 🧠 Working Principle
- A camera is attached to the ego vehicle in CARLA  
- YOLOv8 detects objects in real-time  
- Only **vehicle classes** are considered  
- The LED matrix dynamically **turns OFF regions** corresponding to detected vehicles  
- This simulates an **anti-glare adaptive headlight system**  

---

## 🔧 Tech Stack
- Python  
- OpenCV  
- YOLOv8 (Ultralytics)  
- CARLA Simulator  
- NumPy  
- PyTorch  

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
