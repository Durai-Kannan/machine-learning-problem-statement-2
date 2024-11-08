import cv2
import numpy as np
from time import sleep

# Parameters for contour filtering
min_width = 80   # Minimum width of the rectangle for a vehicle
min_height = 80  # Minimum height of the rectangle for a vehicle
delay = 60       # FPS of the video

# Traffic jam threshold
traffic_jam_threshold = 20

# Load video
cap = cv2.VideoCapture('video.mp4')

# Use MOG2 for background subtraction with shadow detection enabled
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)

def process_single_frame(frame):
    """
    Processes a single frame to detect vehicles and returns processed frame and vehicle count.

    Args:
        frame (numpy.ndarray): The current video frame.

    Returns:
        processed_frame (numpy.ndarray): Frame with bounding boxes and labels.
        cars_in_frame (int): The count of detected vehicles in the frame.
        traffic_status (str): Traffic status message ("Traffic Jam" or "Normal Traffic").
    """
    height, width, _ = frame.shape

    # Apply background subtraction and noise reduction
    fg_mask = background_subtractor.apply(frame)
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8))

    # Find contours in the mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize car count for this frame
    cars_in_frame = 0

    # Process each contour to detect vehicles
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # Filter contours by size
        if w >= min_width and h >= min_height:
            # Increment car count
            cars_in_frame += 1

            # Draw bounding box around each detected vehicle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Determine traffic status based on car count
    traffic_status = "Traffic Jam" if cars_in_frame > traffic_jam_threshold else "Normal Traffic"

    # Display vehicle count and traffic status on the frame
    cv2.putText(frame, "VEHICLE COUNT: " + str(cars_in_frame), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.putText(frame, "TRAFFIC STATUS: " + traffic_status, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
    
    return frame, cars_in_frame, traffic_status

# Main video loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Control the frame rate
    sleep(1 / delay)

    # Process each frame independently
    processed_frame, cars_in_frame, traffic_status = process_single_frame(frame)

    # Show the processed frame
    cv2.imshow("Traffic Detection", processed_frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:  # 27 is the ASCII code for 'Esc'
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

