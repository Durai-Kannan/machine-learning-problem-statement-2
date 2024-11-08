import cv2
import torch
import numpy as np
from time import time

# Load YOLOv5 model (automatically downloads weights if not present)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the detection confidence threshold
model.conf = 0.4  # Confidence threshold (adjust as needed)

# Traffic jam threshold
traffic_jam_threshold = 12

# Capture video
cap = cv2.VideoCapture('video.mp4')

# Process video at 1 FPS
fps = 1
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current timestamp in seconds
    current_time = time()
    
    # Only process the frame if 1 second has passed
    if current_time - prev_time >= 1.0:
        prev_time = current_time

        # Resize frame for faster processing (optional)
        frame = cv2.resize(frame, (800, 600))

        # Perform vehicle detection with YOLO
        results = model(frame)
        detections = results.pred[0]

        # Count vehicles in the current frame
        vehicle_count = 0

        for det in detections:
            # Extract bounding box and confidence
            x1, y1, x2, y2, confidence, cls = map(int, det[:6])

            # Filter for vehicles (e.g., cars, trucks)
            if cls in [2, 5, 7]:  # Class IDs for cars, buses, and trucks in COCO dataset
                vehicle_count += 1
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Vehicle {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check for traffic jam
        if vehicle_count > traffic_jam_threshold:
            traffic_status = "Traffic Jam"
        else:
            traffic_status = "Normal Traffic"

        # Display vehicle count and traffic status
        cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, traffic_status, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the result
        cv2.imshow("Vehicle Detection and Counting", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
