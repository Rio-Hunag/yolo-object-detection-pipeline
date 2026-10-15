import os
import argparse
from ultralytics import YOLO

def predict():
    parser = argparse.ArgumentParser(description="YOLO Model Inference Script")
    parser.add_argument("--weights", type=str, default="runs/detect/custom_yolo_experiment/weights/best.pt", help="Path to trained weights file")
    parser.add_argument("--source", type=str, required=True, help="Path to input test image or video")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"[ERROR] Trained weights file not found at {args.weights}")
        return

    # Load trained custom model and run inference
    model = YOLO(args.weights)
    print(f"[INFO] Running object detection on {args.source}...")
    results = model.predict(source=args.source, save=True)
    print(f"[INFO] Bounding box predictions saved to {results[0].save_dir}")

if __name__ == "__main__":
    predict()
