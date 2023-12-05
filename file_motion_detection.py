import os
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import cv2
import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency


def read_clarinet_positions(results_directory):
    clarinet_positions = []
    for filename in natsorted(os.listdir(results_directory)):
        if filename.endswith(".txt"):
            frame_number = int(filename.split("_")[-1][:-4])

            with open(os.path.join(results_directory, filename), "r") as file:
                lines = file.readlines()
            if lines:
                clarinet_data = [
                    list(map(float, line.strip().split()[1:])) for line in lines
                ]
                clarinet_positions.append((frame_number, clarinet_data))
            else:
                clarinet_positions.append((frame_number, None))
    return clarinet_positions


def read_person_positions(results_directory):
    person_positions = []

    for filename in natsorted(os.listdir(results_directory)):
        if filename.endswith(".txt"):
            frame_number = int(filename.split("_")[-1][:-4])

            with open(os.path.join(results_directory, filename), "r") as file:
                lines = file.readlines()
            if lines:
                person_data = [
                    list(map(float, line.strip().split()[1:])) for line in lines
                ]
                person_positions.append((frame_number, person_data))
            else:
                person_positions.append((frame_number, None))

    return person_positions


def track_up_down_movement(clarinet_positions):
    up_down_movement = []

    for i in range(1, len(clarinet_positions)):
        current_frame, current_clarinet_data = clarinet_positions[i]
        prev_frame, prev_clarinet_data = clarinet_positions[i - 1]

        if current_clarinet_data and prev_clarinet_data:
            current_y_center = current_clarinet_data[0][1]
            prev_y_center = prev_clarinet_data[0][1]

            y_center_variation = current_y_center - prev_y_center

            up_down_movement.append((current_frame, y_center_variation))

    return up_down_movement


def plot_up_down_movement(up_down_movement):
    if up_down_movement:
        frame_numbers, y_center_variation = zip(*up_down_movement)
        plt.plot(frame_numbers, y_center_variation)
        plt.xlabel("Frame Number")
        plt.ylabel("Y-Center Variation")
        plt.title("Clarinet Up/Down Movement Analysis")
        plt.show()
    else:
        print("No clarinet detected in any frame.")


def plot_2d_trajectory(clarinet_positions):
    if clarinet_positions:
        x_centers = [clarinet_data[0][0] for _, clarinet_data in clarinet_positions]
        y_centers = [clarinet_data[0][1] for _, clarinet_data in clarinet_positions]
        normalized_x = [
            (x - min(x_centers)) / (max(x_centers) - min(x_centers)) for x in x_centers
        ]
        normalized_y = [
            (y - min(y_centers)) / (max(y_centers) - min(y_centers)) for y in y_centers
        ]

        plt.plot(normalized_x, normalized_y, marker="o", linestyle="-", markersize=5)
        plt.xlabel("Normalized X")
        plt.ylabel("Normalized Y")
        plt.title("2D Trajectory of Normalized Clarinet Position")
        plt.show()
    else:
        print("No clarinet detected in any frame.")


def get_clarinet_trajectory(
    clarinet_positions, max_frame_difference=5, width_change_threshold=0
):
    clarinet_trajectory = []

    for i in range(1, len(clarinet_positions)):
        current_frame, current_box_list = clarinet_positions[i]
        prev_frame, prev_box_list = clarinet_positions[i - 1]

        width_change = current_box_list[0][2] - prev_box_list[0][2]

        if width_change > width_change_threshold:
            movement_direction = "Upward"
        elif width_change < -width_change_threshold:
            movement_direction = "Downward"
        else:
            movement_direction = "No Movement"

        if current_frame - prev_frame <= max_frame_difference:
            clarinet_trajectory.append((current_frame, movement_direction))
        else:
            clarinet_trajectory.append((current_frame, "No Movement"))

    return clarinet_trajectory


def get_bending_trajectory(
    person_positions, max_frame_difference=5, height_change_threshold=0
):
    # Assuming 'person_positions' is a list of person bounding box coordinates [(frame, [[x_center, y_center, width, height], ...]), ...]

    person_movement_trajectory = []

    # Loop through each frame to track the person's movement
    for i in range(1, len(person_positions)):
        current_frame, current_box_list = person_positions[i]
        prev_frame, prev_box_list = person_positions[i - 1]

        current_height = current_box_list[0][3]
        prev_height = prev_box_list[0][3]

        height_change = current_height - prev_height

        if height_change > height_change_threshold:
            movement_direction = "Straightening"
        elif height_change < -height_change_threshold:
            movement_direction = "Bending"
        else:
            movement_direction = "No Movement"

        # Check if the frames are at most 5 frames apart
        if current_frame - prev_frame <= max_frame_difference:
            person_movement_trajectory.append((current_frame, movement_direction))
        else:
            person_movement_trajectory.append((current_frame, "No Movement"))
    return person_movement_trajectory


def get_forward_backward_movement(bounding_boxes, max_frame_difference=5, threshold=0):
    # Placeholder for tracking information
    movement_trajectory = []

    # Loop through each frame to track the forward/backward movement
    for i in range(1, len(bounding_boxes)):
        current_frame, current_box_list = bounding_boxes[i]
        prev_frame, prev_box_list = bounding_boxes[i - 1]

        # Extract the x-center values for the current and previous frames
        x_center_current = current_box_list[0][0]
        x_center_prev = prev_box_list[0][0]

        # Calculate the change in x-center position
        x_center_change = x_center_current - x_center_prev

        # Determine the direction of movement based on x-center change
        if x_center_change > threshold:
            movement_direction = "Backward"
        elif x_center_change < -threshold:
            movement_direction = "Forward"
        else:
            movement_direction = "No Movement"

            # Check if the frames are at most 5 frames apart
        if current_frame - prev_frame <= max_frame_difference:
            movement_trajectory.append((current_frame, movement_direction))
        else:
            movement_trajectory.append((current_frame, "No Movement"))

    return movement_trajectory


def overlay_movement_direction(video_path, output_path, trajectory):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_number, movement_direction in trajectory:
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < frame_number - 1:
            ret, _ = cap.read()
            if not ret:
                break
            out.write(_)

        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (255, 0, 0)
        text_position = (width - 150, height - 10)

        cv2.putText(
            frame_rgb,
            movement_direction,
            text_position,
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    cap.release()
    out.release()
    print(f"Overlay video saved at: {output_path}")


def visualization(clarinet_results_directory, person_results_directory):
    clarinet_positions = read_clarinet_positions(clarinet_results_directory)
    clarinet_bell_movement = get_clarinet_trajectory(clarinet_positions)

    person_positions = read_person_positions(person_results_directory)

    knee_bending_movement = get_bending_trajectory(person_positions)
    forward_backward_movement = get_forward_backward_movement(person_positions)

    # Extract frame numbers and movements
    clarinet_frames, clarinet_movements = zip(*clarinet_bell_movement)
    knee_frames, knee_movements = zip(*knee_bending_movement)
    forward_backward_frames, forward_backward_movements = zip(
        *forward_backward_movement
    )

    # Combine the movements into a single list for each frame
    all_frames = set(clarinet_frames + knee_frames + forward_backward_frames)

    # Create dictionaries to map frame numbers to movements
    clarinet_mapping = dict(zip(clarinet_frames, clarinet_movements))
    knee_mapping = dict(zip(knee_frames, knee_movements))
    forward_backward_mapping = dict(
        zip(forward_backward_frames, forward_backward_movements)
    )

    # Extract movements for each frame, defaulting to "No Movement" if not present
    combined_movements = [
        (
            clarinet_mapping.get(frame, "No Movement"),
            knee_mapping.get(frame, "No Movement"),
            forward_backward_mapping.get(frame, "No Movement"),
        )
        for frame in all_frames
    ]

    # Extract individual movements for confusion matrix
    clarinet_labels, knee_labels, forward_backward_labels = zip(*combined_movements)

    # Specify the unique labels for each movement type
    clarinet_unique_labels = ["Upward", "Downward", "No Movement"]
    knee_unique_labels = ["Straightening", "Bending", "No Movement"]
    forward_backward_unique_labels = ["Forward", "Backward", "No Movement"]

    # Create confusion matrix
    conf_matrix = np.zeros(
        (
            len(clarinet_unique_labels),
            len(knee_unique_labels),
            len(forward_backward_unique_labels),
        )
    )

    # Populate the confusion matrix
    for clarinet, knee, forward_backward in combined_movements:
        clarinet_index = clarinet_unique_labels.index(clarinet)
        knee_index = knee_unique_labels.index(knee)
        forward_backward_index = forward_backward_unique_labels.index(forward_backward)
        conf_matrix[clarinet_index, knee_index, forward_backward_index] += 1

    # Create contingency table for chi-squared test
    contingency_table = conf_matrix.sum(axis=2)

    # Perform chi-squared test
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

    print(f"Chi-squared statistic: {chi2_stat}")
    print(f"P-value: {p_value}")

    # Plot 3D confusion matrix using a series of 2D heatmaps
    fig = plt.figure(figsize=(15, 10))

    for i in range(conf_matrix.shape[2]):
        ax = fig.add_subplot(2, 2, i + 1)
        sns.heatmap(
            conf_matrix[:, :, i],
            annot=True,
            fmt=".0f",
            cmap="Blues",
            xticklabels=knee_unique_labels,
            yticklabels=clarinet_unique_labels,
            ax=ax,
        )
        ax.set_xlabel("Knee Movement")
        ax.set_ylabel("Clarinet Movement")
        ax.set_title(f"Forward/Backward Movement: {forward_backward_unique_labels[i]}")

    plt.tight_layout()
    plt.show()


# Directory containing YOLOv5 detection results (.txt files)
results_directory = (
    "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp3/labels"
)

clarinet_positions = read_clarinet_positions(results_directory)
up_down_movement = track_up_down_movement(clarinet_positions)
# plot_up_down_movement(up_down_movement)
# plot_2d_trajectory(clarinet_positions)


video_path = (
    "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp2/Paul_brahms.mp4"
)
output_path = "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp2/Paul_brahms_movement_threshold.mp4"
clarinet_bell_trajectory = get_clarinet_trajectory(clarinet_positions)
# overlay_movement_direction(video_path, output_path, clarinet_bell_trajectory)


# Directory containing person detection results (.txt files)
person_results_directory = (
    "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp6/labels"
)
person_video_path = (
    "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp6/Romi_brahms.mp4"
)
output_path = "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp6/Romi_brahms_movement_forback.mp4"


person_positions = read_person_positions(person_results_directory)

knee_bending = get_bending_trajectory(person_positions, height_change_threshold=0)

forward_backward_trajectory = get_forward_backward_movement(person_positions)

# overlay_movement_direction(person_video_path, output_path, forward_backward_trajectory)


clarinet_results_directory = (
    "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp4/labels"
)

person_results_directory = (
    "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp7/labels"
)

visualization(clarinet_results_directory, person_results_directory)
