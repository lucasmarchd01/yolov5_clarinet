import torch
import cv2
import numpy as np
import time

# Load your locally trained YOLOv5 model
model = torch.hub.load(
    "yolov5",
    "custom",
    path="yolov5/runs/train/exp4/weights/best.pt",
    source="local",
)
model.eval()

# Dummy input (adjust according to your model input)
dummy_input = torch.randn(1, 3, 640, 640)

# Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "yolov5.onnx",
    verbose=True,
    input_names=["input"],
    output_names=["output"],
)
