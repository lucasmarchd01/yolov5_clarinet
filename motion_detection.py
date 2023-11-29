import cv2
import numpy as np
import time

# Load the pre-trained YOLO model
net = cv2.dnn.readNetFromONNX("yolov5.onnx")
layer_names = net.getUnconnectedOutLayersNames()

# Initialize video capture
cap = cv2.VideoCapture(
    0
)  # Use 0 for default webcam, you can change it to the path of a video file if needed

# Initialize the clarinet tracker
tracker = cv2.TrackerCSRT_create()

# Variables for movement analysis
prev_position = None
prev_time = time.time()
movement_threshold = 20  # Adjust this threshold based on your requirements

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO object detection
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Get class IDs, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if (
                confidence > 0.5 and class_id == 1
            ):  # Assuming class_id 1 corresponds to the clarinet in your YOLO model
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Update the tracker with the clarinet bounding box
    if len(indices) > 0:
        bbox = boxes[indices[0][0]]
        tracker.init(frame, (bbox[0], bbox[1], bbox[2], bbox[3]))
        prev_position = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        prev_time = time.time()

    # Update the tracker
    success, bbox = tracker.update(frame)

    if success:
        # Draw bounding box
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

        # Get current clarinet position
        current_position = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        current_time = time.time()

        # Calculate 2D tangential velocity
        delta_time = current_time - prev_time
        if delta_time > 0:
            velocity = (current_position - prev_position) / delta_time
            tangential_velocity = np.dot(
                velocity, np.array([-np.sin(np.pi / 4), np.cos(np.pi / 4)])
            )  # Assuming 45-degree angle for tangential direction
            print(f"Tangential Velocity: {tangential_velocity}")

        # Update previous position and time for the next iteration
        prev_position = current_position
        prev_time = current_time

        # Check for up/down movement based on y-coordinate
        if current_position[1] < prev_position[1] - movement_threshold:
            print("Clarinet moved up!")
        elif current_position[1] > prev_position[1] + movement_threshold:
            print("Clarinet moved down!")

    # Display the frame
    cv2.imshow("Clarinet Tracking", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
