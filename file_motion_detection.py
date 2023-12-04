import os
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import cv2


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


def track_bottom_left_marker(clarinet_positions, max_frame_difference=5):
    bottom_left_marker_trajectory = []

    for i in range(1, len(clarinet_positions)):
        current_frame, current_box_list = clarinet_positions[i]
        prev_frame, prev_box_list = clarinet_positions[i - 1]

        bottom_left_current = (
            current_box_list[0][0] - current_box_list[0][2] / 2,
            current_box_list[0][1] + current_box_list[0][3] / 2,
        )

        bottom_left_prev = (
            prev_box_list[0][0] - prev_box_list[0][2] / 2,
            prev_box_list[0][1] + prev_box_list[0][3] / 2,
        )

        width_change = current_box_list[0][2] - prev_box_list[0][2]

        if width_change > 0:
            movement_direction = "Upward"
        elif width_change < 0:
            movement_direction = "Downward"
        else:
            movement_direction = "No Movement"

        if current_frame - prev_frame <= max_frame_difference:
            bottom_left_marker_trajectory.append(
                (current_frame, bottom_left_current, movement_direction)
            )
        else:
            bottom_left_marker_trajectory.append(
                (current_frame, bottom_left_current, "No Movement")
            )

    return bottom_left_marker_trajectory


def overlay_movement_direction(video_path, output_path, bottom_left_marker_trajectory):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_number, (x, y), movement_direction in bottom_left_marker_trajectory:
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


# Directory containing YOLOv5 detection results (.txt files)
results_directory = (
    "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp4/labels"
)

clarinet_positions = read_clarinet_positions(results_directory)
up_down_movement = track_up_down_movement(clarinet_positions)
plot_up_down_movement(up_down_movement)
plot_2d_trajectory(clarinet_positions)


video_path = (
    "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp4/Sean_brahms.mp4"
)
output_path = "/Users/lucasmarch/Projects/MUMT620_Project/yolov5/runs/detect/exp4/Sean_brahms_movement.mp4"
bottom_left_marker_trajectory = track_bottom_left_marker(clarinet_positions)
overlay_movement_direction(video_path, output_path, bottom_left_marker_trajectory)
