VisionBin ♻️

AI-Powered Smart Waste Classification & Automated Segregation

VisionBin is an intelligent waste-management system that combines computer vision, deep learning, and IoT integration to automatically identify and segregate waste in real time.

The system uses a hybrid YOLOv8 + LeViT architecture: YOLOv8 detects waste objects from a live camera stream, while a lightweight LeViT classifier categorizes each detected object into one of nine waste categories. The classification result can then be consumed by an IoT controller such as an ESP32 to drive the physical segregation mechanism.

⸻

🚀 Key Features

* Real-time waste detection using YOLOv8
* Fine-grained waste classification using a lightweight LeViT model
* Supports 9 waste categories
* Live camera processing using OpenCV
* Flask-based web interface for real-time video streaming
* REST endpoint for exposing the latest classification result
* Designed for edge deployment and IoT-based automated sorting
* Multiple deep-learning architectures evaluated during development

Supported Waste Categories

Category	Examples
Cardboard	Boxes, packaging
E-Waste	Electronic components
Foam / Rubber	Styrofoam, rubber materials
Glass	Bottles, containers
Medical	Syringes, tablets, PPE
Metal	Cans, metal objects
Organic	Food and biodegradable waste
Paper	Sheets, newspapers
Plastic	Bottles, plastic packaging

⸻

🧠 System Architecture

                ┌─────────────────────┐
                │   Camera / Webcam   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      YOLOv8         │
                │  Object Detection   │
                └──────────┬──────────┘
                           │
                    Detected Bounding Box
                           │
                           ▼
                ┌─────────────────────┐
                │   Image Cropping    │
                │   & Preprocessing   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      LeViT          │
                │ Waste Classification│
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Waste Category    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Flask /latest     │
                │    REST Endpoint    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ ESP32 / IoT Control │
                │ Automated Sorting   │
                └─────────────────────┘

⸻

🔬 Hybrid AI Pipeline

VisionBin separates object detection from fine-grained classification.

1. YOLOv8 — Object Detection

YOLOv8 processes frames from the camera and identifies regions containing waste objects.

The detected bounding boxes are extracted and used to crop individual objects before classification.

2. LeViT — Waste Classification

Each detected crop is transformed into a 224 × 224 image and passed through a TorchScript LeViT model.

The classifier predicts one of the nine supported waste classes.

This two-stage approach allows the system to first locate the object and then perform specialized waste classification.

⸻

🌐 Flask Backend

The real-time inference pipeline is exposed through a Flask server.

Endpoints

Endpoint	Description
/	Web interface
/video_feed	Live MJPEG video stream
/latest	Returns the latest detected waste category

Example response:

{
  "class": "plastic"
}

The /latest endpoint enables an external microcontroller such as an ESP32 to periodically retrieve the detected class and trigger the corresponding mechanical sorting action.

⸻

🛠️ Technology Stack

AI / Computer Vision

* Python
* PyTorch
* TorchVision
* Ultralytics YOLOv8
* LeViT
* OpenCV
* Pillow

Backend

* Flask
* REST API
* MJPEG video streaming

Edge / IoT

* Raspberry Pi-oriented TorchScript model
* ESP32 integration
* Camera-based sensing
* Automated waste segregation

Development & Experimentation

* Jupyter Notebook
* PowerPoint automation with python-pptx

⸻

📁 Project Structure

VisionBin/
│
├── app.py                    # Main Flask + real-time inference application
├── app1.py                   # EfficientNetV2 inference prototype
├── demo.py                   # YOLOv8 + LeViT local inference demo
│
├── Models/
│   ├── CoAtNet-2.ipynb       # Model experimentation
│   ├── EfficientNet_V2.ipynb # Model experimentation
│   └── LeViT.ipynb           # LeViT model experimentation
│
├── templates/
│   └── index.html            # Flask web interface
│
├── index.html                # Frontend prototype
├── visionbin.html            # VisionBin UI
│
├── yolov8n.pt                # YOLOv8 detection model
├── levit_trashnext_rpi.pt    # TorchScript LeViT classifier
│
├── generate_ppt.py            # Presentation generator
├── edit_ppt.py                # Presentation editing utility
│
└── requirements.txt           # Python dependencies

The repository also contains earlier model experiments and prototypes using EfficientNetV2 and CoAtNet, reflecting the model-selection and experimentation phase of development.

⸻

⚙️ Installation

Clone the repository:

git clone https://github.com/Vikramsrini/Trash.git
cd Trash

Create a virtual environment:

python -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

The project requires:

ultralytics
torch
torchvision
opencv-python

⸻

▶️ Running VisionBin

Start the Flask application:

python app.py

The server runs on:

http://localhost:5000

Open the web interface in your browser to view the live camera feed and detected waste classifications.

⸻

📊 Model Development

During development, several architectures were investigated before selecting the lightweight LeViT-based classifier for deployment.

Experiments in the repository include:

* LeViT
* EfficientNetV2
* CoAtNet-2

The repository’s presentation tooling documents the final system as achieving approximately 90% classification accuracy across 9 waste classes, with the architecture optimized for practical edge inference.

Performance numbers should be interpreted as project benchmark results and may vary depending on hardware, dataset, lighting conditions, and camera setup.

⸻

💡 Why VisionBin?

Traditional waste bins rely on manual segregation, which can be inefficient and error-prone.

VisionBin aims to move waste segregation closer to the point of disposal by combining:

Computer Vision → AI Classification → IoT Control → Physical Sorting

This enables a single system to identify waste automatically and communicate the classification to a downstream sorting mechanism.

⸻

🔮 Future Improvements

* Improve robustness under different lighting and camera conditions
* Add confidence-aware classification and rejection of uncertain predictions
* Optimize inference further for Raspberry Pi / dedicated edge accelerators
* Connect directly to an ESP32-controlled sorting mechanism
* Add telemetry and waste-category analytics
* Expand the training dataset and supported waste categories
* Containerize and deploy the inference service

⸻

👨‍💻 Author

Vikram S

Computer Science / Software Engineering Student

GitHub

⸻

⭐ Project Highlights

Hybrid AI Architecture
YOLOv8 handles object localization while LeViT performs specialized waste classification.

Real-Time Processing
Camera frames are processed continuously and the latest classification is exposed through a lightweight API.

Edge-Oriented Design
TorchScript models and lightweight architectures make the system suitable for resource-constrained deployments.

AI + IoT Integration
The classification API provides a bridge between computer vision inference and physical automated waste segregation.
