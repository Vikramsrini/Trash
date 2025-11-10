from flask import Flask, render_template, Response, jsonify
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import time

app = Flask(__name__)

# ------------------- Configuration -------------------
classes = ["cardboard", "e-waste", "foam_rubber", "glass", "medical",
           "metal", "organic", "paper", "plastic"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------- Load Models -------------------
yolo = YOLO("yolov8n.pt")
classifier = torch.jit.load("levit_trashnext_rpi.pt", map_location="cpu")
classifier.eval()

# ------------------- Initialize Webcam -------------------
camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not camera.isOpened():
    camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Camera not available.")

# ------------------- Global Variables -------------------
latest_class = None
last_sent_time = 0  # throttle detection updates

# ------------------- Helper Functions -------------------
def run_classification(crop_bgr):
    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb)
        tensor = transform(pil).unsqueeze(0)
        with torch.no_grad():
            out = classifier(tensor)
            logits = out["logits"] if isinstance(out, dict) else out
            pred_idx = logits.argmax(dim=1).item()
            label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
            return label
    except Exception:
        return None

# ------------------- Video Streaming -------------------
def generate_frames():
    global latest_class, last_sent_time
    while True:
        success, frame = camera.read()
        if not success:
            continue

        try:
            yolo_results = yolo(frame)
            detections = yolo_results[0].boxes.xyxy.cpu().numpy()
        except Exception:
            detections = []

        for box in detections:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            label = run_classification(cropped)

            if label:
                # Draw box and label on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Update latest class (throttle 1/sec)
                if time.time() - last_sent_time > 1.0:
                    latest_class = label
                    last_sent_time = time.time()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ------------------- Flask Routes -------------------
@app.route('/')
def index():
    return render_template('index.html')  # Simple HTML to display video

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest', methods=['GET'])
def latest():
    """Return latest detected class for ESP32"""
    global latest_class
    return jsonify({"class": latest_class if latest_class else ""})

# ------------------- Run Flask -------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
