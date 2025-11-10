from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image

# 🔹 Load YOLOv8 (pretrained for detection)
yolo_model = YOLO("yolov8n.pt")  # n = nano (fastest)

# 🔹 Load your trained LeViT classifier
levit_model = torch.jit.load("levit_trashnext_rpi.pt", map_location="cpu")
levit_model.eval()

# 🔹 TrashNext classes
classes = ["cardboard", "e-waste", "foam_rubber", "glass", "medical",
           "metal", "organic", "paper", "plastic"]

# 🔹 Image transformation for LeViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 🔹 Open Mac camera (built-in)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ Camera not detected! Check System Settings → Privacy & Security → Camera")
    exit()

print("📷 Starting hybrid YOLO + LeViT... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔹 YOLOv8 detection
    yolo_results = yolo_model(frame)
    detections = yolo_results[0].boxes.xyxy.cpu().numpy()  # bounding boxes (x1, y1, x2, y2)

    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box[:4])

        # Crop detected region
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # Convert to PIL & preprocess for LeViT
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        img_tensor = transform(pil_img).unsqueeze(0)

        # 🔹 LeViT classification
        with torch.no_grad():
            outputs = levit_model(img_tensor)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            pred_idx = logits.argmax(dim=1).item()
            pred_class = classes[pred_idx]

        # Draw box + class on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, pred_class, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show video
    cv2.imshow("VisionBin YOLO + LeViT", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
