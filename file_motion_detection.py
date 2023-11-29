import os
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

# Directory containing YOLOv5 detection results (.txt files)
results_directory = "yolov5/runs/detect/exp2/labels"

# Placeholder for tracking information
clarinet_positions = []

# Loop through each result file
for filename in natsorted(os.listdir(results_directory)):
    if filename.endswith(".txt"):
        frame_number = int(filename.split("_")[-1][:-4])

        # Read YOLOv5 detection results for the current frame
        with open(os.path.join(results_directory, filename), "r") as file:
            lines = file.readlines()

        # Check if clarinet is detected in the frame
        if lines:
            # Assuming YOLO format: class x_center y_center width height
            clarinet_data = [
                list(map(float, line.strip().split()[1:])) for line in lines
            ]
            clarinet_positions.append((frame_number, clarinet_data))
        else:
            clarinet_positions.append((frame_number, None))

# Extract relevant tracking metrics
up_down_movement = []
for frame_number, clarinet_data in clarinet_positions:
    if clarinet_data:
        # Calculate relative variation in y_center (up/down movement)
        y_center_variation = np.diff(np.array(clarinet_data)[:, 1])
        up_down_movement.append((frame_number, y_center_variation))

# Example: Plotting up/down movement over time
if up_down_movement:
    frame_numbers, y_center_variation = zip(*up_down_movement)
    plt.plot(frame_numbers, y_center_variation)
    plt.xlabel("Frame Number")
    plt.ylabel("Y-Center Variation")
    plt.title("Clarinet Up/Down Movement Analysis")
    plt.show()
else:
    print("No clarinet detected in any frame.")
