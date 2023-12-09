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


def get_clarinet_trajectory(
    clarinet_positions, window_size=2, width_change_threshold=0
):
    clarinet_trajectory = []
    averages_with_frames = []

    for i in range(len(clarinet_positions)):
        current_frame, current_box_list = clarinet_positions[i]

        # Collect information for the window
        window_widths = []
        window_frames = []

        # Collect previous frames within the threshold
        for j in range(i, -1, -1):
            if current_frame - clarinet_positions[j][0] <= window_size:
                window_widths.append(clarinet_positions[j][1][0][2])
                window_frames.append(clarinet_positions[j][0])
            else:
                break

        # Collect next frames within the threshold
        for j in range(i + 1, len(clarinet_positions)):
            if clarinet_positions[j][0] - current_frame <= window_size:
                window_widths.append(clarinet_positions[j][1][0][2])
                window_frames.append(clarinet_positions[j][0])
            else:
                break

        # Calculate the average width within the window
        if window_widths:
            avg_width = sum(window_widths) / len(window_widths)
        else:
            avg_width = 0

        # Calculate the width change relative to the previously calculated frame average
        if averages_with_frames:
            prev_frame, prev_avg = averages_with_frames[-1]
            if prev_frame in window_frames:
                width_change = avg_width - prev_avg
            else:
                width_change = 0
        else:
            width_change = 0

        averages_with_frames.append((current_frame, avg_width))

        # Determine movement direction based on the width change
        if width_change > width_change_threshold:
            movement_direction = "Upward"
        elif width_change < -width_change_threshold:
            movement_direction = "Downward"
        else:
            movement_direction = "No Movement"

        # Append the result to the trajectory
        clarinet_trajectory.append((current_frame, movement_direction))

    return clarinet_trajectory


def get_bending_trajectory(
    person_positions, frame_threshold=2, height_change_threshold=0
):
    person_movement_trajectory = []
    averages_with_frames = []

    for i in range(len(person_positions)):
        current_frame, current_box_list = person_positions[i]

        # Collect information for the window
        window_heights = []
        window_frames = []

        # Collect previous frames within the threshold
        for j in range(i, -1, -1):
            if current_frame - person_positions[j][0] <= frame_threshold:
                window_heights.append(person_positions[j][1][0][3])
                window_frames.append(person_positions[j][0])
            else:
                break

        # Collect next frames within the threshold
        for j in range(i + 1, len(person_positions)):
            if person_positions[j][0] - current_frame <= frame_threshold:
                window_heights.append(person_positions[j][1][0][3])
                window_frames.append(person_positions[j][0])
            else:
                break

        # Calculate the average height within the window
        if window_heights:
            avg_height = sum(window_heights) / len(window_heights)
        else:
            avg_height = 0

        # Calculate the height change relative to the previously calculated frame average
        if averages_with_frames:
            prev_frame, prev_avg = averages_with_frames[-1]
            if (
                prev_frame in window_frames
            ):  # Check if the previous frame is within the window
                height_change = avg_height - prev_avg
            else:
                height_change = 0
        else:
            height_change = 0

        averages_with_frames.append((current_frame, avg_height))

        # Determine movement direction based on the height change
        if height_change > height_change_threshold:
            movement_direction = "Straightening"
        elif height_change < -height_change_threshold:
            movement_direction = "Bending"
        else:
            movement_direction = "No Movement"

        # Append the result to the trajectory
        person_movement_trajectory.append((current_frame, movement_direction))

    return person_movement_trajectory


def get_forward_backward_movement(bounding_boxes, window_size=2, threshold=0):
    # Placeholder for tracking information
    movement_trajectory = []
    averages_with_frames = []

    # Loop through each frame to track the forward/backward movement
    for i in range(len(bounding_boxes)):
        current_frame, current_box_list = bounding_boxes[i]

        # Collect information for the window
        window_x_centers = []
        window_frames = []

        # Collect previous frames within the window
        for j in range(i, max(0, i - window_size) - 1, -1):
            window_x_centers.append(bounding_boxes[j][1][0][0])
            window_frames.append(bounding_boxes[j][0])

        # Collect next frames within the window
        for j in range(i + 1, min(len(bounding_boxes), i + window_size + 1)):
            window_x_centers.append(bounding_boxes[j][1][0][0])
            window_frames.append(bounding_boxes[j][0])

        # Calculate the average x-center position within the window
        if window_x_centers:
            avg_x_center = sum(window_x_centers) / len(window_x_centers)
        else:
            avg_x_center = 0

        # Calculate the x-center change relative to the previously calculated frame average
        if averages_with_frames:
            prev_frame, prev_avg = averages_with_frames[-1]
            if prev_frame in window_frames:
                x_center_change = avg_x_center - prev_avg
            else:
                x_center_change = 0
        else:
            x_center_change = 0

        averages_with_frames.append((current_frame, avg_x_center))

        # Determine movement direction based on the x-center change
        if x_center_change > threshold:
            movement_direction = "Backward"
        elif x_center_change < -threshold:
            movement_direction = "Forward"
        else:
            movement_direction = "No Movement"

        # Append the result to the trajectory
        movement_trajectory.append((current_frame, movement_direction))

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


def get_frame_timestamps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_timestamps = [
        frame_number / fps for frame_number in range(1, total_frames + 1)
    ]

    cap.release()
    return frame_timestamps


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def convert_to_min_sec(timestamp):
    minutes, seconds = divmod(timestamp, 60)
    return f"{int(minutes):02}:{int(seconds):02}"


def map_movement_to_value_clarinet(movement):
    if movement == "Upward":
        return 1
    elif movement == "Downward":
        return -1
    else:
        return 0


def map_movement_to_value_knee(movement):
    if movement == "Straightening":
        return 1
    elif movement == "Bending":
        return -1
    else:
        return 0


def map_movement_to_value_forback(movement):
    if movement == "Forward":
        return 1
    elif movement == "Backward":
        return -1
    else:
        return 0


def visualization(clarinet_results_directory, person_results_directory, video_path):
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
    all_frames = sorted(set(clarinet_frames + knee_frames + forward_backward_frames))

    # Visualization 1: Grouped Bar Chart
    timestamps = get_frame_timestamps(video_path)
    timestamps_detected_clarinet = [
        timestamps[frame_number - 1] for frame_number in clarinet_frames
    ]
    timestamps_detected_clarinet_formatted = [
        convert_to_min_sec(timestamp) for timestamp in timestamps_detected_clarinet
    ]
    # Filter out frames with no movement
    filtered_data = [
        (ts, movement)
        for ts, movement in zip(
            timestamps_detected_clarinet_formatted, clarinet_movements
        )
        if movement != "No Movement"
    ]

    filtered_timestamps, filtered_movements = zip(*filtered_data)
    hue_order = ["Downward", "Upward"]
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title("My Window Title")
    sns.histplot(
        x=filtered_timestamps,
        hue=filtered_movements,
        hue_order=hue_order,
        multiple="stack",
        bins=50,
        palette="viridis",
    )

    # Set x-axis ticks at 15-second intervals
    plt.xticks(
        ticks=plt.xticks()[0][::10],
        rotation=45,  # Rotate labels for better readability
    )

    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.title("Grouped Bar Chart of Clarinet Movements Over Time")
    fig.canvas.manager.set_window_title(
        "Grouped Bar Chart of Clarinet Movements Over Time"
    )
    plt.show()

    # Create a DataFrame with the required data
    data = {
        "Timestamps": filtered_timestamps,
        "Movements": filtered_movements,
    }
    df = pd.DataFrame(data)

    # Plot the clustered bar chart
    fig = plt.figure(figsize=(12, 8))
    sns.countplot(x="Timestamps", hue="Movements", data=df, palette="viridis")
    plt.xticks(rotation=45)
    fig.canvas.manager.set_window_title(
        "Clustered Bar Chart of Clarinet Movements Over Time"
    )
    plt.title("Clustered Bar Chart of Clarinet Movements Over Time")
    plt.show()

    # Visualization 3: Separate Heatmaps for Each Movement Type (excluding "No Movement" for both knee and forward/backward)
    fig, axes = plt.subplots(1, conf_matrix.shape[2] - 1, figsize=(15, 5), sharey=True)

    for i in range(conf_matrix.shape[2] - 1):
        # Exclude "No Movement" class for both knee and forward/backward
        filtered_conf_matrix = conf_matrix[:-1, :-1, i]

        sns.heatmap(
            filtered_conf_matrix,
            cmap="viridis",
            annot=True,
            fmt=".0f",
            xticklabels=knee_unique_labels[:-1],  # Exclude "No Movement" label for knee
            yticklabels=clarinet_unique_labels[
                :-1
            ],  # Exclude "No Movement" label for clarinet
            ax=axes[i],
        )
        axes[i].set_xlabel("Knee Movement")
        axes[i].set_ylabel("Clarinet Movement")
        axes[i].set_title(
            f"Forward/Backward Movement: {forward_backward_unique_labels[i]}"
        )

    plt.tight_layout()
    fig.canvas.manager.set_window_title("Heatmaps for Each Movement Type")
    plt.show()

    # Map clarinet movements to values
    clarinet_values = np.array(
        list(map(map_movement_to_value_clarinet, clarinet_movements))
    )

    # Line Plot
    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(
        x=timestamps_detected_clarinet_formatted,
        y=np.cumsum(clarinet_values),
        errorbar=None,
        marker="o",
        color="blue",
        label="Clarinet Movement",
    )

    # Set x-axis ticks at 15-second intervals
    plt.xticks(
        ticks=plt.xticks()[0][::10],
        rotation=45,  # Rotate labels for better readability
    )

    plt.xlabel("Time")
    plt.ylabel("Cumulative Clarinet Movement")
    plt.title("Line Plot of Cumulative Clarinet Movement Over Time")
    fig.canvas.manager.set_window_title(
        "Line Plot of Cumulative Clarinet Movement Over Time"
    )
    plt.show()

    # Assuming you have data similar to clarinet example for knee bending
    timestamps_detected_knee = [
        timestamps[frame_number - 1] for frame_number in knee_frames
    ]
    timestamps_detected_knee_formatted = [
        convert_to_min_sec(timestamp) for timestamp in timestamps_detected_knee
    ]

    # Filter out frames with no movement
    filtered_data = [
        (ts, movement)
        for ts, movement in zip(timestamps_detected_knee_formatted, knee_movements)
        if movement != "No Movement"
    ]

    filtered_timestamps, filtered_movements = zip(*filtered_data)
    fig = plt.figure(figsize=(12, 8))
    sns.histplot(
        x=filtered_timestamps,
        hue=filtered_movements,
        multiple="stack",
        bins=50,
        palette="viridis",
    )

    plt.xticks(
        ticks=plt.xticks()[0][::10],
        rotation=45,
    )

    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.title("Grouped Bar Chart of Knee Bending Movements Over Time")
    fig.canvas.manager.set_window_title(
        "Grouped Bar Chart of Knee Bending Movements Over Time"
    )
    plt.show()
    # Assuming you have data similar to clarinet example for knee bending
    data_knee = {
        "Timestamps": filtered_timestamps,
        "Movements": filtered_movements,
    }
    df_knee = pd.DataFrame(data_knee)

    fig = plt.figure(figsize=(12, 8))
    sns.countplot(x="Timestamps", hue="Movements", data=df_knee, palette="viridis")
    plt.xticks(rotation=45)
    fig.canvas.manager.set_window_title(
        "Clustered Bar Chart of Knee Movements Over Time"
    )
    plt.title("Clustered Bar Chart of Knee Movements Over Time")
    plt.show()

    # Assuming you have data similar to clarinet example for knee bending
    knee_values = np.array(list(map(map_movement_to_value_knee, knee_movements)))

    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(
        x=timestamps_detected_knee_formatted,
        y=np.cumsum(knee_values),
        errorbar=None,
        marker="o",
        color="blue",
        label="Knee Bending Movement",
    )

    plt.xticks(
        ticks=plt.xticks()[0][::10],
        rotation=45,
    )

    plt.xlabel("Time")
    plt.ylabel("Cumulative Knee Bending Movement")
    plt.title("Line Plot of Cumulative Knee Bending Movement Over Time")
    fig.canvas.manager.set_window_title(
        "Line Plot of Cumulative Knee Bending Movement Over Time"
    )
    plt.show()

    # Assuming you have data similar to clarinet example for forward/backward movements
    timestamps_detected_forward_backward = [
        timestamps[frame_number - 1] for frame_number in forward_backward_frames
    ]
    timestamps_detected_forward_backward_formatted = [
        convert_to_min_sec(timestamp)
        for timestamp in timestamps_detected_forward_backward
    ]

    # Filter out frames with no movement
    filtered_data_forward_backward = [
        (ts, movement)
        for ts, movement in zip(
            timestamps_detected_forward_backward_formatted, forward_backward_movements
        )
        if movement != "No Movement"
    ]

    filtered_timestamps_fb, filtered_movements_fb = zip(*filtered_data_forward_backward)

    fig = plt.figure(figsize=(12, 8))
    sns.histplot(
        x=filtered_timestamps_fb,
        hue=filtered_movements_fb,
        multiple="stack",
        bins=50,
        palette="viridis",
    )

    plt.xticks(
        ticks=plt.xticks()[0][::10],
        rotation=45,
    )

    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.title("Grouped Bar Chart of Forward/Backward Movements Over Time")
    fig.canvas.manager.set_window_title(
        "Grouped Bar Chart of Forward/Backward Movements Over Time"
    )
    plt.show()
    # Assuming you have data similar to clarinet example for forward/backward movements
    data_forward_backward = {
        "Timestamps": filtered_timestamps_fb,
        "Movements": filtered_movements_fb,
    }
    df_forward_backward = pd.DataFrame(data_forward_backward)

    fig = plt.figure(figsize=(12, 8))
    sns.countplot(
        x="Timestamps", hue="Movements", data=df_forward_backward, palette="viridis"
    )
    plt.xticks(rotation=45)
    fig.canvas.manager.set_window_title(
        "Clustered Bar Chart of Forward/Backward Movements Over Time"
    )
    plt.title("Clustered Bar Chart of Forward/Backward Movements Over Time")
    plt.show()

    # Assuming you have data similar to clarinet example for forward/backward movements
    forward_backward_values = np.array(
        list(map(map_movement_to_value_forback, filtered_movements_fb))
    )

    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(
        x=filtered_timestamps_fb,
        y=np.cumsum(forward_backward_values),
        errorbar=None,
        marker="o",
        color="blue",
        label="Forward/Backward Movement",
    )

    plt.xticks(
        ticks=plt.xticks()[0][::10],
        rotation=45,
    )

    plt.xlabel("Time")
    plt.ylabel("Cumulative Forward/Backward Movement")
    plt.title("Line Plot of Cumulative Forward/Backward Movement Over Time")
    fig.canvas.manager.set_window_title(
        "Line Plot of Cumulative Forward/Backward Movement Over Time"
    )
    plt.show()


# Directory containing YOLOv5 detection results (.txt files)
clarinet_results_directory = (
    "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp10/labels"
)

clarinet_positions = read_clarinet_positions(clarinet_results_directory)
up_down_movement = track_up_down_movement(clarinet_positions)


clarinet_video_path = "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp10/Brahms 112BMP Trial 003.mp4"
clarinet_output_path = "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp10/003_brahms_clarinet_smooth.mp4"
clarinet_bell_trajectory = get_clarinet_trajectory(clarinet_positions)
overlay_movement_direction(
    clarinet_video_path, clarinet_output_path, clarinet_bell_trajectory
)


# Directory containing person detection results (.txt files)
person_results_directory = (
    "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp9/labels"
)
person_video_path = "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp9/Brahms 112BMP Trial 003.mp4"
person_knee_output_path = "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp9/003_brahms_knee_smooth.mp4"
person_forback_output_path = "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp9/003_brahms_forback_smooth.mp4"


person_positions = read_person_positions(person_results_directory)

knee_bending = get_bending_trajectory(person_positions, frame_threshold=2)

forward_backward_trajectory = get_forward_backward_movement(person_positions)

overlay_movement_direction(person_video_path, person_knee_output_path, knee_bending)

overlay_movement_direction(
    person_video_path, person_forback_output_path, forward_backward_trajectory
)


clarinet_results_directory = (
    "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp10/labels"
)

person_results_directory = (
    "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp9/labels"
)

video_path = "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp9/Brahms 112BMP Trial 003.mp4"

visualization(
    clarinet_results_directory, person_results_directory, video_path=video_path
)
