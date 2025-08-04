# 🚦 Traffic Violation Detection System

An AI-based system that detects multiple types of traffic violations using computer vision and machine learning. It supports processing of both static images and video files with real-time visualization and violation logging.

---

## 🧩 Key Features

- **YOLOv8** for object detection (vehicles, persons, helmets, traffic lights)
- **EasyOCR** for Indian license plate recognition
- **Helmet Violation Detection**
- **Red Light Violation Detection** using HSV analysis
- **Speed Violation Detection** using pixel-to-meter conversion
- **Automatic Violation Line Detection**
- **Modular Architecture** for easy maintenance and extension
- **Web Interface** powered by Gradio

---

## 📁 Project Structure

├── config/ # Configuration settings
├── core/ # Video and image processing logic
├── data/ # CSV logging and dashboard data
├── detectors/ # Violation detection modules
├── models/ # YOLO and OCR model loading
├── ui/ # Gradio-based user interface
├── main.py # Entry point for app
└── README.md

---

## ⚙️ How to Run

### 1. Clone the Repository

git clone https://github.com/Omkar4140/traffic_violation_hugging_face_v3.git
cd traffic_violation_hugging_face_v3

## 2. Install Dependencies
pip install -r requirements.txt

## 3. Run the Application
python main.py

## 🧠 Detection Modules
Type	Description
Helmet	Checks for helmets on riders and passengers
Red Light	Detects red-light crossing using HSV + violation line
Speed	Calculates vehicle speed from frame difference
License Plate	Recognizes Indian plates and flags invalid/missing ones

⚙️ Configurable Parameters

vehicle_confidence: 0.5
helmet_confidence: 0.4
traffic_light_confidence: 0.3
speed_limit_kmph: 40
line_tolerance: 15
pixel_to_meter_ratio: 0.05
🛠️ Technologies Used
YOLOv8 – Object Detection

EasyOCR – License Plate Recognition

OpenCV – Image/Video Processing

Gradio – Web Interface

Pandas, Matplotlib – Data Handling and Visualization


### 🌐 Live Demo
🧪 Try the Web App:
👉 https://huggingface.co/spaces/Omkar4141/traffic_violation_v3
📌 Note
This project is under active development and may be extended with cloud support, mobile app integration, and improved real-time processing.
