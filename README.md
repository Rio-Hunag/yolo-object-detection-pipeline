YOLOv8 Object Detection & Training Pipeline

An end-to-end workflow for training, evaluating, and running custom YOLOv8 object detection models using PyTorch and Ultralytics.

Features

End-to-End Workflow: Covers data preprocessing, dataset setup, model fine-tuning, and performance evaluation.

Model Training: Fine-tuned YOLOv8 (using yolov8l.pt as base) on custom labeled datasets.

Evaluation: Evaluated performance using Precision, Recall, mAP@0.5, and confusion matrices.

Requirements

Bash
pip install ultralytics torch
